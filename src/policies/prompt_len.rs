//! Prompt length and session identity shared by the PD length policies.
//!
//! `x-prompt-tokens` wins when the caller already knows the token count.
//! Otherwise the body is measured in bytes and scaled by a tokens-per-byte
//! coefficient. That coefficient starts at the same 0.25 default as
//! `MomentumTokenEstimator`.

use super::hash_key::{extract_hash_key_from_body, extract_hash_key_from_headers};
use super::RequestHeaders;

pub(crate) fn prompt_tokens(
    request_text: Option<&str>,
    headers: Option<&RequestHeaders>,
    tokens_per_byte: f64,
) -> usize {
    if let Some(tokens) = header_usize(headers, "x-prompt-tokens") {
        return tokens.max(1);
    }
    let bytes = request_text.map(prompt_bytes).unwrap_or(0);
    let scaled = (bytes as f64 * tokens_per_byte).ceil() as usize;
    scaled.max(1)
}

pub(crate) fn output_allowance(
    request_text: Option<&str>,
    default_output: usize,
    output_cap: usize,
) -> usize {
    let requested = request_text
        .and_then(json_output_tokens)
        .unwrap_or(default_output);
    requested.min(output_cap).max(1)
}

/// Session identity only. A missing session stays `None` so the caller can
/// fall through to load sampling instead of hashing the whole body.
pub(crate) fn session_key(
    request_text: Option<&str>,
    headers: Option<&RequestHeaders>,
) -> Option<String> {
    if let Some(headers) = headers {
        if let Some(key) = extract_hash_key_from_headers(headers) {
            return Some(key);
        }
    }
    extract_hash_key_from_body(request_text)
}

fn header_usize(headers: Option<&RequestHeaders>, name: &str) -> Option<usize> {
    headers
        .and_then(|headers| headers.get(name))
        .and_then(|value| value.parse().ok())
}

fn prompt_bytes(text: &str) -> usize {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return text.len();
    };
    if let Some(prompt) = value.get("prompt").and_then(|prompt| prompt.as_str()) {
        return prompt.len().max(1);
    }
    if let Some(messages) = value
        .get("messages")
        .and_then(|messages| messages.as_array())
    {
        let bytes: usize = messages
            .iter()
            .map(|message| {
                message
                    .get("content")
                    .and_then(|content| content.as_str())
                    .map(|content| content.len())
                    .unwrap_or(0)
            })
            .sum();
        if bytes > 0 {
            return bytes;
        }
    }
    text.len().max(1)
}

fn json_output_tokens(text: &str) -> Option<usize> {
    let value = serde_json::from_str::<serde_json::Value>(text).ok()?;
    for name in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
        if let Some(tokens) = value.get(name).and_then(|value| value.as_u64()) {
            return Some(tokens as usize);
        }
    }
    None
}
