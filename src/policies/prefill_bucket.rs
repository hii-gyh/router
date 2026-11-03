//! Length-aware prefill selection for differentiated worker configs.
//!
//! Workers are assigned to length tiers (e.g. a short pool with a smaller
//! `max-model-len`, and an open pool that can take any prompt). The policy
//! keeps two rules separate:
//!
//! 1. **Hard capacity.** A prompt only lands on a worker whose tier can hold
//!    it. Long prompts never enter a short-only config.
//! 2. **Soft preference.** Among eligible workers, prefer the tightest tier
//!    that covers the prompt, but pay a load-equivalent penalty rather than
//!    refusing the larger tiers. When the preferred tier is busy, traffic
//!    borrows capacity from the open pool instead of queueing behind a hard
//!    partition.
//!
//! Session identity hashes inside the preferred tier while that worker is
//! available. Without a session, two eligible workers are sampled and the
//! lower-cost one is kept (`cost = load + preference_weight × tier_distance`).
//!
//! Set `hard_partition = true` to restore the old spill-threshold behavior
//! (preferred tier only, until `spill_abs` / `spill_rel` fire).

use std::collections::HashMap;
use std::sync::Arc;

use rand::Rng;

use super::prompt_len::{prompt_tokens, session_key};
use super::{get_healthy_worker_indices, LoadBalancingPolicy, RequestHeaders};
use crate::core::Worker;
use crate::policies::ConsistentHashPolicy;

const VIRTUAL_NODES_PER_WORKER: u32 = 160;

/// Upper bounds of the finite buckets, in ascending token order.
/// One open-ended bucket always follows the last finite ceiling.
#[derive(Debug, Clone)]
pub struct PrefillBucketConfig {
    pub bucket_max_tokens: Vec<usize>,
    pub spill_abs: usize,
    pub spill_rel: f32,
    /// Worker URL to bucket index. Missing URLs belong to the open bucket.
    pub worker_buckets: HashMap<String, usize>,
    pub tokens_per_byte: f64,
    /// Extra load-equivalent cost per tier step away from the preferred bucket.
    /// `2.0` means an idle open worker beats a preferred worker with 3+ in flight.
    pub preference_weight: f32,
    /// When true, stay on the preferred tier until the spill thresholds fire.
    pub hard_partition: bool,
}

impl Default for PrefillBucketConfig {
    fn default() -> Self {
        Self {
            bucket_max_tokens: vec![2048, 8192, 32768],
            spill_abs: 8,
            spill_rel: 1.5,
            worker_buckets: HashMap::new(),
            tokens_per_byte: 0.25,
            preference_weight: 2.0,
            hard_partition: false,
        }
    }
}

impl PrefillBucketConfig {
    pub fn bucket_count(&self) -> usize {
        self.bucket_max_tokens.len() + 1
    }

    pub fn bucket_for_tokens(&self, prompt_tokens: usize) -> usize {
        self.bucket_max_tokens
            .iter()
            .position(|ceiling| prompt_tokens <= *ceiling)
            .unwrap_or(self.bucket_max_tokens.len())
    }

    pub fn bucket_of_worker(&self, url: &str) -> usize {
        self.worker_buckets
            .get(url)
            .copied()
            .unwrap_or(self.bucket_max_tokens.len())
            .min(self.bucket_count().saturating_sub(1))
    }

    pub fn worker_cost(&self, worker: &PrefillWorkerView, preferred: usize) -> f32 {
        let tier = self.bucket_of_worker(&worker.url);
        let distance = tier.saturating_sub(preferred) as f32;
        worker.load as f32 + self.preference_weight * distance
    }
}

#[derive(Debug, Clone)]
pub struct PrefillWorkerView {
    pub index: usize,
    pub url: String,
    pub available: bool,
    pub load: usize,
}

#[derive(Debug)]
pub struct PrefillBucketPolicy {
    config: PrefillBucketConfig,
}

impl PrefillBucketPolicy {
    pub fn new(config: PrefillBucketConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PrefillBucketConfig {
        &self.config
    }
}

pub fn select_prefill_worker(
    workers: &[PrefillWorkerView],
    prompt_tokens: usize,
    session: Option<&str>,
    config: &PrefillBucketConfig,
    rng: &mut impl Rng,
) -> Option<usize> {
    if workers.is_empty() || config.bucket_max_tokens.is_empty() {
        return None;
    }
    let preferred = config.bucket_for_tokens(prompt_tokens);
    if config.hard_partition {
        return select_hard(workers, preferred, session, config, rng);
    }
    select_soft(workers, preferred, session, config, rng)
}

fn select_soft(
    workers: &[PrefillWorkerView],
    preferred: usize,
    session: Option<&str>,
    config: &PrefillBucketConfig,
    rng: &mut impl Rng,
) -> Option<usize> {
    // Hard capacity: never place onto a tighter tier than the prompt needs.
    let eligible: Vec<&PrefillWorkerView> = workers
        .iter()
        .filter(|worker| {
            worker.available && config.bucket_of_worker(&worker.url) >= preferred
        })
        .collect();
    if eligible.is_empty() {
        return None;
    }
    if let Some(session) = session.filter(|session| !session.is_empty()) {
        let preferred_members: Vec<&PrefillWorkerView> = workers
            .iter()
            .filter(|worker| config.bucket_of_worker(&worker.url) == preferred)
            .collect();
        if !preferred_members.is_empty() {
            let chosen = consistent_member(&preferred_members, session);
            if chosen.available {
                return Some(chosen.index);
            }
        }
        return least_cost(&eligible, preferred, config).map(|worker| worker.index);
    }
    // Compare one preferred-tier worker against one non-preferred eligible
    // worker when both sides exist. That keeps idle shorts on the short tier
    // (cost 0 beats cost preference_weight) and borrows open capacity as soon
    // as preferred load exceeds the open cost — without depending on whether
    // power-of-two happened to sample the right pair.
    let preferred_eligible: Vec<&PrefillWorkerView> = eligible
        .iter()
        .copied()
        .filter(|worker| config.bucket_of_worker(&worker.url) == preferred)
        .collect();
    let other_eligible: Vec<&PrefillWorkerView> = eligible
        .iter()
        .copied()
        .filter(|worker| config.bucket_of_worker(&worker.url) != preferred)
        .collect();
    if preferred_eligible.is_empty() {
        return power_of_two_cost(&eligible, preferred, config, rng).map(|worker| worker.index);
    }
    if other_eligible.is_empty() {
        return power_of_two_load(&preferred_eligible, rng).map(|worker| worker.index);
    }
    let first = preferred_eligible[rng.random_range(0..preferred_eligible.len())];
    let second = other_eligible[rng.random_range(0..other_eligible.len())];
    let first_cost = config.worker_cost(first, preferred);
    let second_cost = config.worker_cost(second, preferred);
    Some(if (first_cost - second_cost).abs() <= f32::EPSILON {
        if first.load <= second.load {
            first.index
        } else {
            second.index
        }
    } else if first_cost < second_cost {
        first.index
    } else {
        second.index
    })
}

fn select_hard(
    workers: &[PrefillWorkerView],
    preferred: usize,
    session: Option<&str>,
    config: &PrefillBucketConfig,
    rng: &mut impl Rng,
) -> Option<usize> {
    for bucket in preferred..config.bucket_count() {
        if bucket_should_spill(workers, bucket, config) {
            continue;
        }
        return Some(pick_within_bucket(workers, bucket, session, config, rng));
    }
    None
}

fn bucket_should_spill(
    workers: &[PrefillWorkerView],
    bucket: usize,
    config: &PrefillBucketConfig,
) -> bool {
    let available = available_in_bucket(workers, bucket, config);
    if available.is_empty() {
        return true;
    }
    let Some(next) = available_in_bucket(workers, bucket + 1, config)
        .into_iter()
        .map(|worker| worker.load)
        .min()
    else {
        return false;
    };
    let min_load = available
        .iter()
        .map(|worker| worker.load)
        .min()
        .unwrap_or(0);
    min_load >= config.spill_abs && (min_load as f32) > next as f32 * config.spill_rel
}

fn available_in_bucket<'a>(
    workers: &'a [PrefillWorkerView],
    bucket: usize,
    config: &PrefillBucketConfig,
) -> Vec<&'a PrefillWorkerView> {
    if bucket >= config.bucket_count() {
        return Vec::new();
    }
    workers
        .iter()
        .filter(|worker| worker.available && config.bucket_of_worker(&worker.url) == bucket)
        .collect()
}

fn pick_within_bucket(
    workers: &[PrefillWorkerView],
    bucket: usize,
    session: Option<&str>,
    config: &PrefillBucketConfig,
    rng: &mut impl Rng,
) -> usize {
    let members: Vec<&PrefillWorkerView> = workers
        .iter()
        .filter(|worker| config.bucket_of_worker(&worker.url) == bucket)
        .collect();
    let available: Vec<&PrefillWorkerView> = members
        .iter()
        .copied()
        .filter(|worker| worker.available)
        .collect();
    if let Some(session) = session.filter(|session| !session.is_empty()) {
        let chosen = consistent_member(&members, session);
        if chosen.available {
            return chosen.index;
        }
        return least_loaded(&available)
            .map(|worker| worker.index)
            .unwrap_or(chosen.index);
    }
    power_of_two_load(&available, rng)
        .map(|worker| worker.index)
        .unwrap_or(members[0].index)
}

fn consistent_member<'a>(
    members: &[&'a PrefillWorkerView],
    session: &str,
) -> &'a PrefillWorkerView {
    let mut ring = Vec::with_capacity(members.len() * VIRTUAL_NODES_PER_WORKER as usize);
    for worker in members {
        for vnode in 0..VIRTUAL_NODES_PER_WORKER {
            let hash = ConsistentHashPolicy::fbi_hash(&format!("{}#{vnode}", worker.url));
            ring.push((hash, *worker));
        }
    }
    ring.sort_by_key(|(hash, _)| *hash);
    let key_hash = ConsistentHashPolicy::fbi_hash(session);
    ring.iter()
        .find(|(hash, _)| *hash >= key_hash)
        .map(|(_, worker)| *worker)
        .unwrap_or(ring[0].1)
}

fn least_loaded<'a>(workers: &[&'a PrefillWorkerView]) -> Option<&'a PrefillWorkerView> {
    workers.iter().copied().min_by(|left, right| {
        left.load
            .cmp(&right.load)
            .then_with(|| left.url.cmp(&right.url))
    })
}

fn least_cost<'a>(
    workers: &[&'a PrefillWorkerView],
    preferred: usize,
    config: &PrefillBucketConfig,
) -> Option<&'a PrefillWorkerView> {
    workers.iter().copied().min_by(|left, right| {
        config
            .worker_cost(left, preferred)
            .partial_cmp(&config.worker_cost(right, preferred))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.load.cmp(&right.load))
            .then_with(|| left.url.cmp(&right.url))
    })
}

fn power_of_two_load<'a>(
    workers: &[&'a PrefillWorkerView],
    rng: &mut impl Rng,
) -> Option<&'a PrefillWorkerView> {
    match workers.len() {
        0 => None,
        1 => Some(workers[0]),
        len => {
            let left = rng.random_range(0..len);
            let mut right = rng.random_range(0..len);
            while right == left {
                right = rng.random_range(0..len);
            }
            let first = workers[left];
            let second = workers[right];
            Some(if first.load <= second.load {
                first
            } else {
                second
            })
        }
    }
}

fn power_of_two_cost<'a>(
    workers: &[&'a PrefillWorkerView],
    preferred: usize,
    config: &PrefillBucketConfig,
    rng: &mut impl Rng,
) -> Option<&'a PrefillWorkerView> {
    match workers.len() {
        0 => None,
        1 => Some(workers[0]),
        len => {
            let left = rng.random_range(0..len);
            let mut right = rng.random_range(0..len);
            while right == left {
                right = rng.random_range(0..len);
            }
            let first = workers[left];
            let second = workers[right];
            let first_cost = config.worker_cost(first, preferred);
            let second_cost = config.worker_cost(second, preferred);
            Some(if (first_cost - second_cost).abs() <= f32::EPSILON {
                if first.load <= second.load {
                    first
                } else {
                    second
                }
            } else if first_cost < second_cost {
                first
            } else {
                second
            })
        }
    }
}

impl LoadBalancingPolicy for PrefillBucketPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        if get_healthy_worker_indices(workers).is_empty() {
            return None;
        }
        let views: Vec<PrefillWorkerView> = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| PrefillWorkerView {
                index,
                url: worker.url().to_string(),
                available: worker.is_available(),
                load: worker.load(),
            })
            .collect();
        let prompt = prompt_tokens(request_text, headers, self.config.tokens_per_byte);
        let session = session_key(request_text, headers);
        select_prefill_worker(
            &views,
            prompt,
            session.as_deref(),
            &self.config,
            &mut rand::rng(),
        )
    }

    fn name(&self) -> &'static str {
        "prefill_bucket"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
