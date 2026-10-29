//! Length-bucket prefill selection.
//!
//! Workers are statically assigned to a bucket. A request enters the tightest
//! bucket whose ceiling covers the estimated prompt, and spills only toward a
//! larger bucket when the tight one has no available worker or is busier than
//! the next bucket by the configured margin. Session identity is hashed inside
//! the chosen bucket. Without a session, two available workers are sampled and
//! the one with fewer in-flight requests is kept.

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
}

impl Default for PrefillBucketConfig {
    fn default() -> Self {
        Self {
            bucket_max_tokens: vec![2048, 8192, 32768],
            spill_abs: 8,
            spill_rel: 1.5,
            worker_buckets: HashMap::new(),
            tokens_per_byte: 0.25,
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
    let start = config.bucket_for_tokens(prompt_tokens);
    for bucket in start..config.bucket_count() {
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
    power_of_two(&available, rng)
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

fn power_of_two<'a>(
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
