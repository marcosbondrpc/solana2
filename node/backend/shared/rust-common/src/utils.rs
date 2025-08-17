use chrono::{DateTime, Utc};

pub fn current_timestamp() -> i64 {
    Utc::now().timestamp_millis()
}

pub fn format_lamports(lamports: u64) -> String {
    let sol = lamports as f64 / 1_000_000_000.0;
    format!("{:.9} SOL", sol)
}