//! Soft length-aware prefill selection over a shared worker pool.
//!
//! Unlike `prefill_bucket`, this policy never hard-partitions workers. Every
//! available worker is eligible. Selection minimizes:
//!
//! ```text
//! score = load_weight * load_norm + mismatch_weight * length_mismatch
//! ```
//!
//! `length_mismatch` is the absolute log2 distance between the request prompt
//! length and the worker's preferred length (static label / config map, else
//! EWMA of recently assigned lengths). When the pool is load-imbalanced by the
//! absolute and relative thresholds, the mismatch term is zeroed and selection
//! falls back to least-loaded.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::Mutex;

use super::prompt_len::prompt_tokens;
use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;

const LABEL_PREFERRED_INPUT_TOKENS: &str = "preferred_input_tokens";

/// Tunables for soft length-aware scoring.
#[derive(Debug, Clone)]
pub struct SoftLengthAwareConfig {
    pub load_weight: f32,
    pub mismatch_weight: f32,
    /// Bytes→tokens estimator used when headers omit a token count.
    /// Equivalent to `1 / chars_per_token` (default 0.25 ≈ 4 chars/token).
    pub tokens_per_byte: f64,
    pub ewma_alpha: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    /// Optional static preferred lengths keyed by worker URL.
    pub preferred_input_tokens: HashMap<String, usize>,
}

impl Default for SoftLengthAwareConfig {
    fn default() -> Self {
        Self {
            load_weight: 1.0,
            mismatch_weight: 0.75,
            tokens_per_byte: 0.25,
            ewma_alpha: 0.2,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            preferred_input_tokens: HashMap::new(),
        }
    }
}

/// Snapshot used by the pure selection helper (and by the device-free sim).
#[derive(Debug, Clone)]
pub struct SoftLengthWorkerView {
    pub index: usize,
    pub url: String,
    pub available: bool,
    pub load: usize,
    /// Static preference from labels / config. `None` falls back to EWMA.
    pub preferred_input_tokens: Option<usize>,
}

#[derive(Debug)]
pub struct SoftLengthAwarePolicy {
    config: SoftLengthAwareConfig,
    /// Per-worker EWMA of assigned prompt lengths.
    ewma: DashMap<String, f64>,
    /// Tiny mutex so selection + EWMA update stay consistent under reserve.
    update_lock: Mutex<()>,
}

impl SoftLengthAwarePolicy {
    pub fn new(config: SoftLengthAwareConfig) -> Self {
        Self {
            config,
            ewma: DashMap::new(),
            update_lock: Mutex::new(()),
        }
    }

    pub fn config(&self) -> &SoftLengthAwareConfig {
        &self.config
    }

    pub fn ewma_for(&self, url: &str) -> Option<f64> {
        self.ewma.get(url).map(|value| *value)
    }

    fn preferred_for_view(&self, worker: &SoftLengthWorkerView) -> f64 {
        if let Some(tokens) = worker.preferred_input_tokens.filter(|tokens| *tokens > 0) {
            return tokens as f64;
        }
        if let Some(tokens) = self
            .config
            .preferred_input_tokens
            .get(&worker.url)
            .copied()
            .filter(|tokens| *tokens > 0)
        {
            return tokens as f64;
        }
        self.ewma
            .get(&worker.url)
            .map(|value| (*value).max(1.0))
            .unwrap_or(1.0)
    }

    fn record_assignment(&self, url: &str, prompt_tokens: usize) {
        let observed = (prompt_tokens.max(1)) as f64;
        let alpha = self.config.ewma_alpha.clamp(0.0, 1.0) as f64;
        self.ewma
            .entry(url.to_string())
            .and_modify(|prev| *prev = alpha * observed + (1.0 - alpha) * *prev)
            .or_insert(observed);
    }
}

/// Absolute log2 distance between request length and preferred length.
pub fn length_mismatch(prompt_tokens: usize, preferred_tokens: f64) -> f32 {
    let request = (prompt_tokens.max(1) as f64).ln();
    let preferred = preferred_tokens.max(1.0).ln();
    ((request - preferred).abs() / std::f64::consts::LN_2) as f32
}

/// Pool is imbalanced when the busiest worker is both absolutely and
/// relatively heavier than the lightest available worker.
pub fn pool_imbalanced(loads: &[usize], abs_threshold: usize, rel_threshold: f32) -> bool {
    let Some(min_load) = loads.iter().copied().min() else {
        return false;
    };
    let Some(max_load) = loads.iter().copied().max() else {
        return false;
    };
    if max_load.saturating_sub(min_load) < abs_threshold {
        return false;
    }
    (max_load as f32) > (min_load as f32) * rel_threshold
}

pub fn worker_score(
    worker: &SoftLengthWorkerView,
    prompt_tokens: usize,
    preferred_tokens: f64,
    mismatch_weight: f32,
    config: &SoftLengthAwareConfig,
) -> f32 {
    // `load_norm` is the raw in-flight count: one queued request trades against
    // roughly `1 / mismatch_weight` log2-length steps under the default weights.
    let load_norm = worker.load as f32;
    let mismatch = length_mismatch(prompt_tokens, preferred_tokens);
    config.load_weight * load_norm + mismatch_weight * mismatch
}

/// Pure shared-pool selection. Returns the worker index or `None` if empty.
pub fn select_soft_length_worker(
    workers: &[SoftLengthWorkerView],
    prompt_tokens: usize,
    preferred_fn: impl Fn(&SoftLengthWorkerView) -> f64,
    config: &SoftLengthAwareConfig,
) -> Option<usize> {
    let eligible: Vec<&SoftLengthWorkerView> = workers
        .iter()
        .filter(|worker| worker.available)
        .collect();
    if eligible.is_empty() {
        return None;
    }

    let loads: Vec<usize> = eligible.iter().map(|worker| worker.load).collect();
    let imbalanced = pool_imbalanced(
        &loads,
        config.balance_abs_threshold,
        config.balance_rel_threshold,
    );
    let mismatch_weight = if imbalanced { 0.0 } else { config.mismatch_weight };

    eligible
        .into_iter()
        .min_by(|left, right| {
            let left_score = worker_score(
                left,
                prompt_tokens,
                preferred_fn(left),
                mismatch_weight,
                config,
            );
            let right_score = worker_score(
                right,
                prompt_tokens,
                preferred_fn(right),
                mismatch_weight,
                config,
            );
            left_score
                .partial_cmp(&right_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.load.cmp(&right.load))
                .then_with(|| left.url.cmp(&right.url))
        })
        .map(|worker| worker.index)
}

fn preferred_from_worker(worker: &dyn Worker) -> Option<usize> {
    worker
        .metadata()
        .labels
        .get(LABEL_PREFERRED_INPUT_TOKENS)
        .and_then(|value| value.parse().ok())
        .filter(|tokens| *tokens > 0)
}

impl LoadBalancingPolicy for SoftLengthAwarePolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        if get_healthy_worker_indices(workers).is_empty() {
            return None;
        }
        let views: Vec<SoftLengthWorkerView> = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| SoftLengthWorkerView {
                index,
                url: worker.url().to_string(),
                available: worker.is_available(),
                load: worker.load(),
                preferred_input_tokens: preferred_from_worker(worker.as_ref()),
            })
            .collect();
        let prompt = prompt_tokens(request_text, headers, self.config.tokens_per_byte);
        let _guard = self.update_lock.lock();
        let chosen = select_soft_length_worker(
            &views,
            prompt,
            |worker| self.preferred_for_view(worker),
            &self.config,
        )?;
        self.record_assignment(&views[chosen].url, prompt);
        Some(chosen)
    }

    fn name(&self) -> &'static str {
        "soft_length_aware"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn tracks_request_load(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, WorkerType};
    use std::collections::HashMap;

    fn views(specs: &[(&str, usize, Option<usize>)]) -> Vec<SoftLengthWorkerView> {
        specs
            .iter()
            .enumerate()
            .map(|(index, (url, load, preferred))| SoftLengthWorkerView {
                index,
                url: (*url).to_string(),
                available: true,
                load: *load,
                preferred_input_tokens: *preferred,
            })
            .collect()
    }

    #[test]
    fn log_mismatch_is_zero_for_equal_lengths() {
        assert!((length_mismatch(4096, 4096.0)).abs() < 1e-6);
        assert!((length_mismatch(8192, 4096.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn prefers_matching_length_when_loads_are_equal() {
        let config = SoftLengthAwareConfig::default();
        let workers = views(&[("short", 0, Some(4096)), ("long", 0, Some(30_000))]);
        let chosen = select_soft_length_worker(
            &workers,
            4096,
            |worker| worker.preferred_input_tokens.unwrap_or(1) as f64,
            &config,
        )
        .unwrap();
        assert_eq!(workers[chosen].url, "short");

        let chosen_long = select_soft_length_worker(
            &workers,
            28_000,
            |worker| worker.preferred_input_tokens.unwrap_or(1) as f64,
            &config,
        )
        .unwrap();
        assert_eq!(workers[chosen_long].url, "long");
    }

    #[test]
    fn imbalance_zeros_mismatch_and_picks_least_loaded() {
        let config = SoftLengthAwareConfig {
            balance_abs_threshold: 2,
            balance_rel_threshold: 1.1,
            ..SoftLengthAwareConfig::default()
        };
        // short matches length but is much busier; imbalance must ignore affinity.
        let workers = views(&[("short", 8, Some(4096)), ("long", 0, Some(30_000))]);
        let chosen = select_soft_length_worker(
            &workers,
            4096,
            |worker| worker.preferred_input_tokens.unwrap_or(1) as f64,
            &config,
        )
        .unwrap();
        assert_eq!(workers[chosen].url, "long");
    }

    #[test]
    fn shared_pool_allows_short_worker_to_take_long_prompt_under_load() {
        // Absolute load scoring: a few extra in-flight requests outweigh a
        // multi-bit length mismatch, so the idle short worker can absorb a
        // long prompt without waiting for the imbalance gate.
        let config = SoftLengthAwareConfig::default();
        let workers = views(&[("short", 0, Some(4096)), ("long", 4, Some(30_000))]);
        let chosen = select_soft_length_worker(
            &workers,
            28_000,
            |worker| worker.preferred_input_tokens.unwrap_or(1) as f64,
            &config,
        )
        .unwrap();
        assert_eq!(workers[chosen].url, "short");
    }

    #[test]
    fn ewma_moves_toward_assigned_lengths() {
        let policy = SoftLengthAwarePolicy::new(SoftLengthAwareConfig {
            ewma_alpha: 0.5,
            ..SoftLengthAwareConfig::default()
        });
        policy.record_assignment("http://p0", 1000);
        assert!((policy.ewma_for("http://p0").unwrap() - 1000.0).abs() < 1e-9);
        policy.record_assignment("http://p0", 3000);
        assert!((policy.ewma_for("http://p0").unwrap() - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn policy_reads_prompt_header_and_worker_label() {
        let mut preferred = HashMap::new();
        preferred.insert("http://long:8000".to_string(), 30_000);
        let policy = SoftLengthAwarePolicy::new(SoftLengthAwareConfig {
            preferred_input_tokens: preferred,
            ..SoftLengthAwareConfig::default()
        });

        let mut short_labels = HashMap::new();
        short_labels.insert(LABEL_PREFERRED_INPUT_TOKENS.to_string(), "4096".to_string());
        let short = BasicWorker::new(
            "http://short:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        )
        .with_labels(short_labels);
        let long = BasicWorker::new(
            "http://long:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(short), Arc::new(long)];
        let mut headers = RequestHeaders::new();
        headers.insert("x-prompt-tokens".to_string(), "4000".to_string());
        let chosen = policy
            .select_worker_with_headers(&workers, None, Some(&headers))
            .unwrap();
        assert_eq!(workers[chosen].url(), "http://short:8000");
        assert_eq!(policy.name(), "soft_length_aware");
        assert!(policy.ewma_for("http://short:8000").is_some());
    }

    #[test]
    fn accepts_x_input_tokens_alias() {
        let policy = SoftLengthAwarePolicy::new(SoftLengthAwareConfig::default());
        let worker = BasicWorker::new(
            "http://only:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(worker)];
        let mut headers = RequestHeaders::new();
        headers.insert("x-input-tokens".to_string(), "1234".to_string());
        assert_eq!(
            policy.select_worker_with_headers(&workers, None, Some(&headers)),
            Some(0)
        );
    }
}
