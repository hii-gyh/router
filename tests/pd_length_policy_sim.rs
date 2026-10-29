//! Correctness tests and a device-free queue simulation for length-bucket prefill
//! selection and KV-cost decode selection.
//!
//! Service time is modeled as one tick per token. No GPU, NPU, or vLLM process
//! is involved. The simulation compares the new policies with a shared pool and
//! with round-robin under the same arrivals.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use vllm_router_rs::core::{BasicWorker, Worker, WorkerType};
use vllm_router_rs::policies::{
    decode_worker_cost, select_decode_worker, select_prefill_worker, DecodeKvCostConfig,
    DecodeKvCostPolicy, DecodeWorkerView, LoadBalancingPolicy, PrefillBucketConfig,
    PrefillBucketPolicy, PrefillWorkerView, RequestHeaders,
};

fn prefill_views(loads: &[(&str, usize, bool)]) -> Vec<PrefillWorkerView> {
    loads
        .iter()
        .enumerate()
        .map(|(index, (url, load, available))| PrefillWorkerView {
            index,
            url: (*url).to_string(),
            available: *available,
            load: *load,
        })
        .collect()
}

fn bucket_config(assignments: &[(&str, usize)]) -> PrefillBucketConfig {
    let mut worker_buckets = HashMap::new();
    for (url, bucket) in assignments {
        worker_buckets.insert((*url).to_string(), *bucket);
    }
    PrefillBucketConfig {
        bucket_max_tokens: vec![2048, 8192],
        spill_abs: 8,
        spill_rel: 1.5,
        worker_buckets,
        tokens_per_byte: 0.25,
    }
}

fn decode_view(
    index: usize,
    usage: f64,
    running: usize,
    waiting: usize,
    fresh: bool,
) -> DecodeWorkerView {
    DecodeWorkerView {
        index,
        available: true,
        local_load: running,
        kv_usage: Some(usage),
        running: Some(running),
        waiting: Some(waiting),
        fresh,
    }
}

#[test]
fn decode_cost_matches_the_length_and_pressure_example() {
    let config = DecodeKvCostConfig::default();
    let hot = decode_view(0, 0.85, 20, 4, true);
    let cool = decode_view(1, 0.30, 4, 0, true);
    let hot_cost = decode_worker_cost(8192, &hot, &config);
    let cool_cost = decode_worker_cost(8192, &cool, &config);
    assert!((hot_cost - 2.475).abs() < 1e-9, "hot={hot_cost}");
    assert!((cool_cost - 0.9375).abs() < 1e-9, "cool={cool_cost}");

    let short_on_hot = decode_worker_cost(512, &hot, &config);
    assert!(
        (short_on_hot - 1.44375).abs() < 1e-9,
        "short_on_hot={short_on_hot}"
    );
    assert!(cool_cost < hot_cost);
    assert!(short_on_hot > decode_worker_cost(512, &cool, &config));
}

#[test]
fn prefill_bucket_keeps_long_prompts_out_of_shorter_buckets() {
    let config = bucket_config(&[("short", 0), ("mid", 1), ("open", 2)]);
    let workers = prefill_views(&[("short", 0, true), ("mid", 0, true), ("open", 0, true)]);
    let mut rng = StdRng::seed_from_u64(1);
    let short = select_prefill_worker(&workers, 512, Some("session-a"), &config, &mut rng);
    let mid = select_prefill_worker(&workers, 4096, Some("session-a"), &config, &mut rng);
    let long = select_prefill_worker(&workers, 9000, Some("session-a"), &config, &mut rng);
    assert_eq!(workers[short.unwrap()].url, "short");
    assert_eq!(workers[mid.unwrap()].url, "mid");
    assert_eq!(workers[long.unwrap()].url, "open");
}

#[test]
fn session_stays_on_the_same_prefill_until_that_worker_is_unavailable() {
    let config = bucket_config(&[("p0", 0), ("p1", 0)]);
    let mut rng = StdRng::seed_from_u64(2);
    let healthy = prefill_views(&[("p0", 1, true), ("p1", 0, true)]);
    let first = select_prefill_worker(&healthy, 1000, Some("chat-7"), &config, &mut rng).unwrap();
    for _ in 0..20 {
        let again =
            select_prefill_worker(&healthy, 1000, Some("chat-7"), &config, &mut rng).unwrap();
        assert_eq!(again, first);
    }

    let mut degraded = healthy.clone();
    degraded[first].available = false;
    let fallback =
        select_prefill_worker(&degraded, 1000, Some("chat-7"), &config, &mut rng).unwrap();
    assert_ne!(fallback, first);
    assert_eq!(degraded[fallback].url, if first == 0 { "p1" } else { "p0" });
}

#[test]
fn hot_short_bucket_spills_upward_and_a_similarly_hot_next_bucket_does_not() {
    let config = bucket_config(&[("s0", 0), ("s1", 0), ("l0", 1)]);
    let mut rng = StdRng::seed_from_u64(3);
    let spilling = prefill_views(&[("s0", 10, true), ("s1", 12, true), ("l0", 0, true)]);
    let spilled = select_prefill_worker(&spilling, 256, Some("s"), &config, &mut rng).unwrap();
    assert_eq!(spilling[spilled].url, "l0");

    let held = prefill_views(&[("s0", 10, true), ("s1", 12, true), ("l0", 9, true)]);
    let stayed = select_prefill_worker(&held, 256, Some("s"), &config, &mut rng).unwrap();
    assert_ne!(held[stayed].url, "l0");
}

#[test]
fn unmapped_prefill_workers_belong_to_the_open_bucket() {
    let config = PrefillBucketConfig::default();
    assert_eq!(config.bucket_of_worker("http://unused:8000"), 3);
    assert_eq!(config.bucket_for_tokens(2048), 0);
    assert_eq!(config.bucket_for_tokens(2049), 1);
    assert_eq!(config.bucket_for_tokens(40_000), 3);
}

#[test]
fn decode_gate_skips_a_full_worker_and_uses_it_only_when_every_worker_is_full() {
    let config = DecodeKvCostConfig::default();
    let mut rng = StdRng::seed_from_u64(4);
    let mixed = vec![
        decode_view(0, 0.97, 20, 0, true),
        decode_view(1, 0.40, 3, 0, true),
    ];
    assert_eq!(
        select_decode_worker(&mixed, 1024, &config, &mut rng),
        Some(1)
    );

    let both_full = vec![
        decode_view(0, 0.97, 20, 0, true),
        decode_view(1, 0.93, 12, 9, true),
    ];
    assert_eq!(
        select_decode_worker(&both_full, 1024, &config, &mut rng),
        Some(1)
    );
}

#[test]
fn stale_kv_usage_does_not_block_a_worker() {
    let config = DecodeKvCostConfig::default();
    let mut rng = StdRng::seed_from_u64(5);
    let workers = vec![
        DecodeWorkerView {
            index: 0,
            available: true,
            local_load: 0,
            kv_usage: Some(0.99),
            running: Some(30),
            waiting: Some(20),
            fresh: false,
        },
        decode_view(1, 0.20, 2, 0, true),
    ];
    let chosen = select_decode_worker(&workers, 512, &config, &mut rng).unwrap();
    assert_eq!(chosen, 0);
}

#[test]
fn prometheus_sample_is_visible_to_decode_selection() {
    let policy = DecodeKvCostPolicy::new(DecodeKvCostConfig::default());
    let body = r#"
# HELP vllm:kv_cache_usage_perc GPU KV cache usage. 1 means 100 percent usage.
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.20
vllm:num_requests_running{engine="0",model_name="m"} 2
vllm:num_requests_waiting{engine="0",model_name="m"} 0
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.96
vllm:num_requests_running{engine="1",model_name="m"} 18
vllm:num_requests_waiting{engine="1",model_name="m"} 3
"#;
    policy
        .inbox()
        .ingest_prometheus("http://decode:8000@1", body);
    let workers: Vec<std::sync::Arc<dyn Worker>> = vec![
        std::sync::Arc::new(BasicWorker::new(
            "http://decode:8000@1".to_string(),
            WorkerType::Decode,
        )),
        std::sync::Arc::new(BasicWorker::new(
            "http://decode:8000@0".to_string(),
            WorkerType::Decode,
        )),
    ];
    let mut headers = RequestHeaders::new();
    headers.insert("x-prompt-tokens".to_string(), "4096".to_string());
    let chosen = policy
        .select_worker_with_headers(&workers, None, Some(&headers))
        .unwrap();
    assert_eq!(workers[chosen].url(), "http://decode:8000@0");
}

#[test]
fn prefill_policy_reads_prompt_tokens_and_max_tokens_headers_are_not_required_for_bucket() {
    let mut worker_buckets = HashMap::new();
    worker_buckets.insert("http://short:8000".to_string(), 0);
    worker_buckets.insert("http://long:8000".to_string(), 1);
    let policy = PrefillBucketPolicy::new(PrefillBucketConfig {
        bucket_max_tokens: vec![2048],
        worker_buckets,
        spill_abs: 8,
        ..PrefillBucketConfig::default()
    });
    let workers: Vec<std::sync::Arc<dyn Worker>> = vec![
        std::sync::Arc::new(BasicWorker::new(
            "http://short:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        )),
        std::sync::Arc::new(BasicWorker::new(
            "http://long:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        )),
    ];
    let mut headers = RequestHeaders::new();
    headers.insert("x-prompt-tokens".to_string(), "32000".to_string());
    headers.insert("x-session-id".to_string(), "conversation".to_string());
    let chosen = policy
        .select_worker_with_headers(
            &workers,
            Some(r#"{"max_tokens":128,"messages":[{"content":"hi"}]}"#),
            Some(&headers),
        )
        .unwrap();
    assert_eq!(workers[chosen].url(), "http://long:8000");
    assert_eq!(policy.name(), "prefill_bucket");
}

#[test]
fn bimodal_prefill_short_requests_wait_less_than_in_a_shared_pool() {
    let bucketed = simulate_prefill(true);
    let shared = simulate_prefill(false);
    assert!(
        bucketed.short_mean_wait * 5.0 < shared.short_mean_wait,
        "bucketed short wait {} should be well below shared short wait {}",
        bucketed.short_mean_wait,
        shared.short_mean_wait
    );
    assert!(
        bucketed.short_p50_wait < shared.short_p50_wait,
        "bucketed p50 {} shared p50 {}",
        bucketed.short_p50_wait,
        shared.short_p50_wait
    );
    assert_eq!(
        bucketed.misplaced_long, 0,
        "a long prompt entered the short bucket"
    );
}

#[test]
fn high_kv_pressure_decode_overflows_less_often_than_round_robin() {
    let cost = simulate_decode(Placement::KvCost, ArrivalProfile::HighPressure);
    let round_robin = simulate_decode(Placement::RoundRobin, ArrivalProfile::HighPressure);
    assert!(
        cost.overflows * 2 < round_robin.overflows,
        "cost overflows {} round-robin overflows {}",
        cost.overflows,
        round_robin.overflows
    );
    assert!(
        cost.usage_stddev < round_robin.usage_stddev,
        "cost stddev {} round-robin stddev {}",
        cost.usage_stddev,
        round_robin.usage_stddev
    );
}

#[test]
fn low_kv_pressure_does_not_create_overflow() {
    let cost = simulate_decode(Placement::KvCost, ArrivalProfile::LowPressure);
    let round_robin = simulate_decode(Placement::RoundRobin, ArrivalProfile::LowPressure);
    assert_eq!(cost.overflows, 0);
    assert_eq!(round_robin.overflows, 0);
}

#[test]
fn stale_observation_timestamp_is_ignored_by_the_policy() {
    let policy = DecodeKvCostPolicy::new(DecodeKvCostConfig {
        stale_after: Duration::from_secs(3),
        ..DecodeKvCostConfig::default()
    });
    policy.inbox().ingest_prometheus_at(
        "http://old:8000",
        r#"
vllm:kv_cache_usage_perc{engine="0"} 0.99
vllm:num_requests_running{engine="0"} 40
vllm:num_requests_waiting{engine="0"} 12
"#,
        Instant::now() - Duration::from_secs(30),
    );
    policy.inbox().ingest_prometheus(
        "http://fresh:8000",
        r#"
vllm:kv_cache_usage_perc{engine="0"} 0.55
vllm:num_requests_running{engine="0"} 6
vllm:num_requests_waiting{engine="0"} 1
"#,
    );
    let old = BasicWorker::new("http://old:8000".to_string(), WorkerType::Decode);
    let fresh = BasicWorker::new("http://fresh:8000".to_string(), WorkerType::Decode);
    let workers: Vec<std::sync::Arc<dyn Worker>> =
        vec![std::sync::Arc::new(old), std::sync::Arc::new(fresh)];
    let mut headers = RequestHeaders::new();
    headers.insert("x-prompt-tokens".to_string(), "128".to_string());
    let chosen = policy
        .select_worker_with_headers(&workers, None, Some(&headers))
        .unwrap();
    assert_eq!(workers[chosen].url(), "http://old:8000");
}

struct PrefillSim {
    short_mean_wait: f64,
    short_p50_wait: u64,
    misplaced_long: usize,
}

fn simulate_prefill(bucketed: bool) -> PrefillSim {
    let mut config = PrefillBucketConfig {
        bucket_max_tokens: vec![2048],
        spill_abs: 10_000,
        spill_rel: 1.5,
        worker_buckets: HashMap::new(),
        tokens_per_byte: 0.25,
    };
    if bucketed {
        config.worker_buckets.insert("p0".to_string(), 0);
        config.worker_buckets.insert("p1".to_string(), 0);
    }
    let urls = ["p0", "p1", "p2", "p3"];
    let mut free_at = [0u64; 4];
    let mut inflight = [0usize; 4];
    let mut finishes: Vec<(u64, usize)> = Vec::new();
    let mut short_waits = Vec::new();
    let mut misplaced_long = 0usize;
    let mut rng = StdRng::seed_from_u64(11);

    // Period 3 against 4 workers, so round-robin mixes long and short work
    // onto the same worker. A period-2 pattern lines up with the worker
    // count and accidentally keeps them apart.
    for step in 0..400u64 {
        let now = step * 30;
        finishes.retain(|(done, worker)| {
            if *done <= now {
                inflight[*worker] = inflight[*worker].saturating_sub(1);
                false
            } else {
                true
            }
        });
        let tokens = if step % 3 == 0 { 8_192 } else { 256 };
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
        let chosen = if bucketed {
            select_prefill_worker(&views, tokens, None, &config, &mut rng).expect("prefill worker")
        } else {
            (step % 4) as usize
        };
        if tokens > 2048 && config.bucket_of_worker(&views[chosen].url) == 0 {
            misplaced_long += 1;
        }
        let wait = free_at[chosen].saturating_sub(now);
        if tokens <= 2048 {
            short_waits.push(wait);
        }
        let start = free_at[chosen].max(now);
        free_at[chosen] = start + tokens as u64;
        inflight[chosen] += 1;
        finishes.push((free_at[chosen], chosen));
    }
    short_waits.sort_unstable();
    let mean = short_waits.iter().sum::<u64>() as f64 / short_waits.len() as f64;
    PrefillSim {
        short_mean_wait: mean,
        short_p50_wait: short_waits[short_waits.len() / 2],
        misplaced_long,
    }
}

#[derive(Clone, Copy)]
enum Placement {
    KvCost,
    RoundRobin,
}

#[derive(Clone, Copy)]
enum ArrivalProfile {
    HighPressure,
    LowPressure,
}

struct DecodeSim {
    overflows: usize,
    usage_stddev: f64,
}

fn simulate_decode(placement: Placement, profile: ArrivalProfile) -> DecodeSim {
    let (jobs, interarrival, lifetime, capacity) = match profile {
        // Even requests are long. Round-robin therefore stacks every long
        // request on one worker; the cost function spreads those longs apart.
        ArrivalProfile::HighPressure => (80usize, 10u64, 40u64, 20_000usize),
        ArrivalProfile::LowPressure => (120usize, 80u64, 8u64, 16_384usize),
    };
    let config = DecodeKvCostConfig::default();
    let mut used = [0i64; 2];
    let mut running = [0usize; 2];
    let mut finishes: Vec<(u64, usize, i64)> = Vec::new();
    let mut usage_samples = Vec::new();
    let mut overflows = 0usize;
    let mut rng = StdRng::seed_from_u64(19);
    let mut cursor = 0usize;

    for step in 0..jobs {
        let now = step as u64 * interarrival;
        finishes.retain(|(done, worker, kv)| {
            if *done <= now {
                used[*worker] -= kv;
                running[*worker] = running[*worker].saturating_sub(1);
                false
            } else {
                true
            }
        });
        let demand = match profile {
            ArrivalProfile::HighPressure if step % 2 == 0 => 12_000,
            ArrivalProfile::HighPressure => 1_000,
            ArrivalProfile::LowPressure => 1_024,
        };
        let chosen = match placement {
            Placement::RoundRobin => {
                let index = cursor % 2;
                cursor += 1;
                index
            }
            Placement::KvCost => {
                let views = vec![
                    usage_view(0, used[0], running[0], capacity),
                    usage_view(1, used[1], running[1], capacity),
                ];
                select_decode_worker(&views, demand, &config, &mut rng).unwrap()
            }
        };
        if used[chosen] + demand as i64 > capacity as i64 {
            overflows += 1;
        }
        used[chosen] += demand as i64;
        running[chosen] += 1;
        finishes.push((now + lifetime, chosen, demand as i64));
        let mean = (used[0] + used[1]) as f64 / 2.0;
        let variance = ((used[0] as f64 - mean).powi(2) + (used[1] as f64 - mean).powi(2)) / 2.0;
        usage_samples.push(variance.sqrt());
    }
    let usage_stddev = usage_samples.iter().sum::<f64>() / usage_samples.len() as f64;
    DecodeSim {
        overflows,
        usage_stddev,
    }
}

fn usage_view(index: usize, used: i64, running: usize, capacity: usize) -> DecodeWorkerView {
    DecodeWorkerView {
        index,
        available: true,
        local_load: running,
        kv_usage: Some((used.max(0) as f64 / capacity as f64).min(1.5)),
        running: Some(running),
        waiting: Some(0),
        fresh: true,
    }
}
