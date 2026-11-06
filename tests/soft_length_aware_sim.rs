//! Device-free correctness + scheduler pre-eval for `soft_length_aware`.
//!
//! No GPU/NPU/vLLM process is required. Service time is one tick per prompt
//! token. The sim compares soft length affinity against round-robin and against
//! hard `prefill_bucket` under the same mixed arrivals.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vllm_router_rs::config::PolicyConfig;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{
    select_prefill_worker, select_soft_length_worker, LoadBalancingPolicy, PolicyFactory,
    PolicyRegistry, PrefillBucketConfig, PrefillWorkerView, RequestHeaders, SoftLengthAwareConfig,
    SoftLengthAwarePolicy, SoftLengthWorkerView,
};

#[derive(Clone, Copy)]
enum PrefillPlacement {
    SoftLength,
    RoundRobin,
    HardBucket,
}

#[derive(Debug, Clone)]
struct PrefillSimResult {
    mean_wait: f64,
    p90_wait: u64,
    load_stddev: f64,
    /// Fraction of requests that landed on the closer preferred worker.
    affinity_hit_rate: f64,
    policy: &'static str,
}

fn preferred_of(url: &str) -> usize {
    match url {
        "short" => 4_096,
        "long" => 30_000,
        _ => 1,
    }
}

fn closer_worker(prompt: usize) -> &'static str {
    let short_mismatch = ((prompt as f64).ln() - (4_096f64).ln()).abs();
    let long_mismatch = ((prompt as f64).ln() - (30_000f64).ln()).abs();
    if short_mismatch <= long_mismatch {
        "short"
    } else {
        "long"
    }
}

fn soft_config() -> SoftLengthAwareConfig {
    SoftLengthAwareConfig {
        load_weight: 1.0,
        mismatch_weight: 0.75,
        tokens_per_byte: 0.25,
        ewma_alpha: 0.2,
        balance_abs_threshold: 4,
        balance_rel_threshold: 1.25,
        preferred_input_tokens: HashMap::from([
            ("short".to_string(), 4_096),
            ("long".to_string(), 30_000),
        ]),
    }
}

fn hard_bucket_config() -> PrefillBucketConfig {
    PrefillBucketConfig {
        bucket_max_tokens: vec![8_192],
        spill_abs: 4,
        spill_rel: 1.2,
        worker_buckets: HashMap::from([("short".to_string(), 0), ("long".to_string(), 1)]),
        tokens_per_byte: 0.25,
        preference_weight: 2.0,
        hard_partition: true,
    }
}

fn sample_prompt(rng: &mut StdRng) -> usize {
    // Mixed length profile from the session export (input-side).
    let draw: f64 = rng.random();
    if draw < 0.25 {
        4_096
    } else if draw < 0.50 {
        7_168
    } else if draw < 0.75 {
        16_384
    } else {
        30_000
    }
}

fn simulate(placement: PrefillPlacement, jobs: usize, seed: u64) -> PrefillSimResult {
    let soft = soft_config();
    let hard = hard_bucket_config();
    let urls = ["short", "long"];
    let mut free_at = [0u64; 2];
    let mut inflight = [0usize; 2];
    let mut finishes: Vec<(u64, usize)> = Vec::new();
    let mut waits = Vec::with_capacity(jobs);
    let mut affinity_hits = 0usize;
    let mut load_samples = Vec::with_capacity(jobs);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rr_cursor = 0usize;
    // Online EWMA state for soft selection (mirrors the policy).
    let mut ewma = [0.0f64; 2];
    let mut ewma_init = [false; 2];

    for step in 0..jobs {
        let now = step as u64 * 40;
        finishes.retain(|(done, worker)| {
            if *done <= now {
                inflight[*worker] = inflight[*worker].saturating_sub(1);
                false
            } else {
                true
            }
        });
        let prompt = sample_prompt(&mut rng);
        let chosen = match placement {
            PrefillPlacement::RoundRobin => {
                let index = rr_cursor % 2;
                rr_cursor += 1;
                index
            }
            PrefillPlacement::HardBucket => {
                let views: Vec<PrefillWorkerView> = urls
                    .iter()
                    .enumerate()
                    .map(|(index, url)| PrefillWorkerView {
                        index,
                        url: (*url).to_string(),
                        available: true,
                        load: inflight[index],
                    })
                    .collect();
                select_prefill_worker(&views, prompt, None, &hard, &mut rng).expect("hard pick")
            }
            PrefillPlacement::SoftLength => {
                let views: Vec<SoftLengthWorkerView> = urls
                    .iter()
                    .enumerate()
                    .map(|(index, url)| SoftLengthWorkerView {
                        index,
                        url: (*url).to_string(),
                        available: true,
                        load: inflight[index],
                        preferred_input_tokens: Some(preferred_of(url)),
                    })
                    .collect();
                let pick = select_soft_length_worker(
                    &views,
                    prompt,
                    |worker| {
                        if let Some(tokens) = worker.preferred_input_tokens {
                            return tokens as f64;
                        }
                        if ewma_init[worker.index] {
                            ewma[worker.index].max(1.0)
                        } else {
                            1.0
                        }
                    },
                    &soft,
                )
                .expect("soft pick");
                let alpha = soft.ewma_alpha as f64;
                let observed = prompt as f64;
                if ewma_init[pick] {
                    ewma[pick] = alpha * observed + (1.0 - alpha) * ewma[pick];
                } else {
                    ewma[pick] = observed;
                    ewma_init[pick] = true;
                }
                pick
            }
        };

        if urls[chosen] == closer_worker(prompt) {
            affinity_hits += 1;
        }
        let wait = free_at[chosen].saturating_sub(now);
        waits.push(wait);
        let start = free_at[chosen].max(now);
        free_at[chosen] = start + prompt as u64;
        inflight[chosen] += 1;
        finishes.push((free_at[chosen], chosen));
        let mean = (inflight[0] + inflight[1]) as f64 / 2.0;
        let variance =
            ((inflight[0] as f64 - mean).powi(2) + (inflight[1] as f64 - mean).powi(2)) / 2.0;
        load_samples.push(variance.sqrt());
    }

    waits.sort_unstable();
    let mean_wait = waits.iter().sum::<u64>() as f64 / waits.len() as f64;
    let p90_wait = waits[(waits.len() as f64 * 0.9) as usize];
    let load_stddev = load_samples.iter().sum::<f64>() / load_samples.len() as f64;
    PrefillSimResult {
        mean_wait,
        p90_wait,
        load_stddev,
        affinity_hit_rate: affinity_hits as f64 / jobs as f64,
        policy: match placement {
            PrefillPlacement::SoftLength => "soft_length_aware",
            PrefillPlacement::RoundRobin => "round_robin",
            PrefillPlacement::HardBucket => "prefill_bucket_hard",
        },
    }
}

#[test]
fn soft_length_beats_round_robin_affinity_under_mixed_lengths() {
    let soft = simulate(PrefillPlacement::SoftLength, 800, 20251103);
    let rr = simulate(PrefillPlacement::RoundRobin, 800, 20251103);
    eprintln!(
        "pre-eval soft={:?}\npre-eval rr={:?}",
        soft, rr
    );
    assert!(
        soft.affinity_hit_rate > rr.affinity_hit_rate + 0.08,
        "soft affinity {} should beat RR affinity {}",
        soft.affinity_hit_rate,
        rr.affinity_hit_rate
    );
}

#[test]
fn soft_length_keeps_wait_competitive_with_round_robin() {
    let soft = simulate(PrefillPlacement::SoftLength, 800, 7);
    let rr = simulate(PrefillPlacement::RoundRobin, 800, 7);
    let hard = simulate(PrefillPlacement::HardBucket, 800, 7);
    eprintln!("wait soft={:?}\nrr={:?}\nhard={:?}", soft, rr, hard);
    // Mean wait is the primary pre-eval gate: soft should stay near RR while
    // buying affinity. Tail wait may rise vs RR, but should beat hard buckets.
    assert!(
        soft.mean_wait < rr.mean_wait * 1.10,
        "soft mean wait {} vs rr {}",
        soft.mean_wait,
        rr.mean_wait
    );
    assert!(
        soft.p90_wait <= hard.p90_wait,
        "soft p90 {} should not exceed hard-bucket p90 {}",
        soft.p90_wait,
        hard.p90_wait
    );
    assert!(
        soft.load_stddev <= rr.load_stddev * 2.0,
        "soft load stddev {} vs rr {}",
        soft.load_stddev,
        rr.load_stddev
    );
}

#[test]
fn hard_bucket_affinity_is_high_but_soft_avoids_its_wait_regression() {
    let soft = simulate(PrefillPlacement::SoftLength, 800, 99);
    let hard = simulate(PrefillPlacement::HardBucket, 800, 99);
    let rr = simulate(PrefillPlacement::RoundRobin, 800, 99);
    eprintln!("compare soft={:?}\nhard={:?}\nrr={:?}", soft, hard, rr);
    // Hard partition can look "affinity-perfect" on short prompts but pays in waits
    // when short buckets saturate; soft should stay closer to RR waits.
    assert!(
        soft.mean_wait <= hard.mean_wait * 1.05
            || soft.load_stddev + 0.05 < hard.load_stddev,
        "soft should not be clearly worse than hard on both wait and balance"
    );
}

#[test]
fn factory_and_registry_expose_soft_length_aware() {
    let config = PolicyConfig::SoftLengthAware {
        load_weight: 1.0,
        mismatch_weight: 0.75,
        tokens_per_byte: 0.25,
        ewma_alpha: 0.2,
        balance_abs_threshold: 32,
        balance_rel_threshold: 1.1,
        preferred_input_tokens: HashMap::new(),
    };
    let policy = PolicyFactory::create_from_config(&config, None);
    assert_eq!(policy.name(), "soft_length_aware");
    assert_eq!(
        PolicyFactory::create_by_name("soft_length_aware")
            .expect("by name")
            .name(),
        "soft_length_aware"
    );

    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
    registry.set_prefill_policy(policy);
    assert_eq!(registry.get_prefill_policy().name(), "soft_length_aware");
    assert_eq!(
        registry
            .on_worker_added("m", Some("soft_length_aware"))
            .name(),
        "soft_length_aware"
    );
}

#[test]
fn policy_selects_labeled_worker_for_matching_prompt() {
    let policy = SoftLengthAwarePolicy::new(soft_config());
    let mut short_labels = HashMap::new();
    short_labels.insert("preferred_input_tokens".to_string(), "4096".to_string());
    let mut long_labels = HashMap::new();
    long_labels.insert("preferred_input_tokens".to_string(), "30000".to_string());
    let workers: Vec<std::sync::Arc<dyn Worker>> = vec![
        std::sync::Arc::new(
            BasicWorker::new(
                "http://short:8000".to_string(),
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            )
            .with_labels(short_labels),
        ),
        std::sync::Arc::new(
            BasicWorker::new(
                "http://long:8000".to_string(),
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            )
            .with_labels(long_labels),
        ),
    ];
    let mut headers = RequestHeaders::new();
    headers.insert("x-prompt-tokens".to_string(), "4096".to_string());
    let chosen = policy
        .select_worker_with_headers(&workers, None, Some(&headers))
        .unwrap();
    assert_eq!(workers[chosen].url(), "http://short:8000");
}

#[test]
fn multi_seed_pre_eval_summary_prints_stable_ranking() {
    let mut rows = Vec::new();
    for seed in [1u64, 2, 3, 5, 8, 13, 21] {
        let soft = simulate(PrefillPlacement::SoftLength, 600, seed);
        let rr = simulate(PrefillPlacement::RoundRobin, 600, seed);
        let hard = simulate(PrefillPlacement::HardBucket, 600, seed);
        rows.push((seed, soft, rr, hard));
    }
    let soft_affinity: f64 = rows.iter().map(|r| r.1.affinity_hit_rate).sum::<f64>() / rows.len() as f64;
    let rr_affinity: f64 = rows.iter().map(|r| r.2.affinity_hit_rate).sum::<f64>() / rows.len() as f64;
    let soft_wait: f64 = rows.iter().map(|r| r.1.mean_wait).sum::<f64>() / rows.len() as f64;
    let rr_wait: f64 = rows.iter().map(|r| r.2.mean_wait).sum::<f64>() / rows.len() as f64;
    let hard_wait: f64 = rows.iter().map(|r| r.3.mean_wait).sum::<f64>() / rows.len() as f64;
    eprintln!(
        "multi-seed summary: soft_affinity={soft_affinity:.3} rr_affinity={rr_affinity:.3} \
         soft_wait={soft_wait:.1} rr_wait={rr_wait:.1} hard_wait={hard_wait:.1}"
    );
    assert!(soft_affinity > rr_affinity);
    // Soft should remain in the RR wait ballpark across seeds.
    assert!(soft_wait < rr_wait * 1.3);
}
