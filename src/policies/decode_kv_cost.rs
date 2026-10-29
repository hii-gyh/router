//! Decode placement by KV pressure, queue depth, and incoming length.
//!
//! The pool is not partitioned. Two eligible workers are sampled and the one
//! with the lower cost is kept. A fresh observation older than three poll
//! intervals is ignored and the worker is scored from the local in-flight count.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use rand::Rng;

use super::prompt_len::{output_allowance, prompt_tokens};
use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;

#[derive(Debug, Clone)]
pub struct DecodeKvCostConfig {
    pub reject_usage: f64,
    pub reject_waiting: usize,
    pub running_ref: f64,
    pub length_ref: f64,
    pub default_output_tokens: usize,
    pub output_cap: usize,
    pub weight_usage: f64,
    pub weight_queue: f64,
    pub weight_admission: f64,
    pub tokens_per_byte: f64,
    pub stale_after: Duration,
    /// Known KV capacity in tokens. When set, admission is remaining-capacity share.
    pub capacity_tokens: Option<usize>,
}

impl Default for DecodeKvCostConfig {
    fn default() -> Self {
        Self {
            reject_usage: 0.92,
            reject_waiting: 8,
            running_ref: 16.0,
            length_ref: 4096.0,
            default_output_tokens: 256,
            output_cap: 4096,
            weight_usage: 1.0,
            weight_queue: 0.35,
            weight_admission: 0.50,
            tokens_per_byte: 0.25,
            stale_after: Duration::from_secs(3),
            capacity_tokens: None,
        }
    }
}

#[derive(Debug, Clone)]
struct StoredObservation {
    kv_usage: Option<f64>,
    running: Option<usize>,
    waiting: Option<usize>,
    at: Instant,
}

#[derive(Debug, Default)]
pub struct DecodeMetricsInbox {
    observations: RwLock<HashMap<String, StoredObservation>>,
}

impl DecodeMetricsInbox {
    pub fn ingest_prometheus(&self, worker_url: &str, body: &str) {
        self.ingest_prometheus_at(worker_url, body, Instant::now());
    }

    pub fn ingest_prometheus_at(&self, worker_url: &str, body: &str, at: Instant) {
        let engine = dp_engine(worker_url);
        let Some(parsed) = parse_engine_metrics(body).get(&engine).cloned() else {
            return;
        };
        self.observe(
            worker_url,
            DecodeObservation {
                kv_usage: parsed.0,
                running: parsed.1,
                waiting: parsed.2,
                at,
            },
        );
    }

    pub fn observe(&self, worker_url: &str, observation: DecodeObservation) {
        self.observations.write().unwrap().insert(
            worker_url.to_string(),
            StoredObservation {
                kv_usage: observation.kv_usage,
                running: observation.running,
                waiting: observation.waiting,
                at: observation.at,
            },
        );
    }
}

#[derive(Debug, Clone)]
pub struct DecodeObservation {
    pub kv_usage: Option<f64>,
    pub running: Option<usize>,
    pub waiting: Option<usize>,
    pub at: Instant,
}

#[derive(Debug, Clone)]
pub struct DecodeWorkerView {
    pub index: usize,
    pub available: bool,
    pub local_load: usize,
    pub kv_usage: Option<f64>,
    pub running: Option<usize>,
    pub waiting: Option<usize>,
    pub fresh: bool,
}

#[derive(Debug)]
pub struct DecodeKvCostPolicy {
    config: DecodeKvCostConfig,
    inbox: Arc<DecodeMetricsInbox>,
}

impl DecodeKvCostPolicy {
    pub fn new(config: DecodeKvCostConfig) -> Self {
        Self {
            config,
            inbox: Arc::new(DecodeMetricsInbox::default()),
        }
    }

    pub fn config(&self) -> &DecodeKvCostConfig {
        &self.config
    }

    pub fn inbox(&self) -> Arc<DecodeMetricsInbox> {
        Arc::clone(&self.inbox)
    }
}

/// `1.0 * U + 0.35 * q + 0.50 * a`, with `q` and the length ratio capped at 2.
pub fn decode_worker_cost(
    demand_tokens: usize,
    worker: &DecodeWorkerView,
    config: &DecodeKvCostConfig,
) -> f64 {
    let local_usage = (worker.local_load as f64 / config.running_ref).min(1.0);
    let (usage, running, waiting, admission_usage) = if worker.fresh {
        (
            worker.kv_usage.unwrap_or(local_usage),
            worker.running.unwrap_or(worker.local_load) as f64,
            worker.waiting.unwrap_or(0) as f64,
            worker.kv_usage.unwrap_or(local_usage),
        )
    } else {
        (local_usage, worker.local_load as f64, 0.0, 0.5)
    };
    let queue = ((running + waiting) / config.running_ref).min(2.0);
    let admission = match (worker.fresh, config.capacity_tokens) {
        (true, Some(capacity)) => {
            let free = (capacity as f64 * (1.0 - admission_usage)).max(1.0);
            (demand_tokens as f64 / free).min(2.0)
        }
        _ => {
            let length_ratio = (demand_tokens as f64 / config.length_ref).min(2.0);
            length_ratio * (0.25 + admission_usage)
        }
    };
    config.weight_usage * usage + config.weight_queue * queue + config.weight_admission * admission
}

pub fn select_decode_worker(
    workers: &[DecodeWorkerView],
    demand_tokens: usize,
    config: &DecodeKvCostConfig,
    rng: &mut impl Rng,
) -> Option<usize> {
    let available: Vec<&DecodeWorkerView> =
        workers.iter().filter(|worker| worker.available).collect();
    if available.is_empty() {
        return None;
    }
    let eligible: Vec<&DecodeWorkerView> = available
        .iter()
        .copied()
        .filter(|worker| !blocked(worker, config))
        .collect();
    let pool: Vec<&DecodeWorkerView> = if eligible.is_empty() {
        available
    } else {
        eligible
    };
    Some(power_of_two(&pool, demand_tokens, config, rng).index)
}

fn blocked(worker: &DecodeWorkerView, config: &DecodeKvCostConfig) -> bool {
    if !worker.fresh {
        return false;
    }
    let usage_blocked = worker
        .kv_usage
        .is_some_and(|usage| usage >= config.reject_usage);
    let waiting_blocked = worker
        .waiting
        .is_some_and(|waiting| waiting >= config.reject_waiting);
    usage_blocked || waiting_blocked
}

fn power_of_two<'a>(
    workers: &[&'a DecodeWorkerView],
    demand_tokens: usize,
    config: &DecodeKvCostConfig,
    rng: &mut impl Rng,
) -> &'a DecodeWorkerView {
    if workers.len() == 1 {
        return workers[0];
    }
    let left = rng.random_range(0..workers.len());
    let mut right = rng.random_range(0..workers.len());
    while right == left {
        right = rng.random_range(0..workers.len());
    }
    let first = workers[left];
    let second = workers[right];
    let first_cost = decode_worker_cost(demand_tokens, first, config);
    let second_cost = decode_worker_cost(demand_tokens, second, config);
    if first_cost < second_cost
        || ((first_cost - second_cost).abs() <= f64::EPSILON && first.index <= second.index)
    {
        first
    } else {
        second
    }
}

fn dp_engine(worker_url: &str) -> usize {
    worker_url
        .rsplit_once('@')
        .and_then(|(_, rank)| rank.parse().ok())
        .unwrap_or(0)
}

fn metrics_base(worker_url: &str) -> &str {
    match worker_url.rsplit_once('@') {
        Some((base, rank)) if rank.parse::<usize>().is_ok() => base,
        _ => worker_url,
    }
}

fn parse_engine_metrics(body: &str) -> HashMap<usize, (Option<f64>, Option<usize>, Option<usize>)> {
    let mut engines: HashMap<usize, (Option<f64>, Option<usize>, Option<usize>)> = HashMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((series, raw_value)) = line.rsplit_once(char::is_whitespace) else {
            continue;
        };
        let engine = series
            .split("engine=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .and_then(|engine| engine.parse().ok())
            .unwrap_or(0);
        let entry = engines.entry(engine).or_default();
        if series.starts_with("vllm:kv_cache_usage_perc") {
            entry.0 = raw_value.parse().ok();
        } else if series.starts_with("vllm:num_requests_running") {
            entry.1 = raw_value.parse::<f64>().ok().map(|value| value as usize);
        } else if series.starts_with("vllm:num_requests_waiting") {
            entry.2 = raw_value.parse::<f64>().ok().map(|value| value as usize);
        }
    }
    engines
}

pub fn spawn_decode_metrics_poller(
    client: reqwest::Client,
    worker_urls: Vec<String>,
    inbox: Arc<DecodeMetricsInbox>,
    interval: Duration,
) {
    if worker_urls.is_empty() {
        return;
    }
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            for url in &worker_urls {
                let metrics_url = format!("{}/metrics", metrics_base(url));
                let Ok(response) = client.get(&metrics_url).send().await else {
                    continue;
                };
                if !response.status().is_success() {
                    continue;
                }
                let Ok(body) = response.text().await else {
                    continue;
                };
                inbox.ingest_prometheus(url, &body);
            }
        }
    });
}

impl LoadBalancingPolicy for DecodeKvCostPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        if get_healthy_worker_indices(workers).is_empty() {
            return None;
        }
        let now = Instant::now();
        let stored = self.inbox.observations.read().unwrap();
        let views: Vec<DecodeWorkerView> = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| {
                let observation = stored.get(worker.url());
                let fresh = observation
                    .map(|observation| {
                        now.saturating_duration_since(observation.at) <= self.config.stale_after
                    })
                    .unwrap_or(false);
                DecodeWorkerView {
                    index,
                    available: worker.is_available(),
                    local_load: worker.load(),
                    kv_usage: observation.and_then(|observation| observation.kv_usage),
                    running: observation.and_then(|observation| observation.running),
                    waiting: observation.and_then(|observation| observation.waiting),
                    fresh,
                }
            })
            .collect();
        drop(stored);
        let prompt = prompt_tokens(request_text, headers, self.config.tokens_per_byte);
        let output = output_allowance(
            request_text,
            self.config.default_output_tokens,
            self.config.output_cap,
        );
        select_decode_worker(&views, prompt + output, &self.config, &mut rand::rng())
    }

    fn name(&self) -> &'static str {
        "decode_kv_cost"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn metrics_inbox(&self) -> Option<Arc<DecodeMetricsInbox>> {
        Some(Arc::clone(&self.inbox))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
