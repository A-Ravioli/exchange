// #region agent log
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
// #endregion

fn log_build(hypothesis_id: &str, message: &str, data: &str) {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/Users/aravioli/Desktop/Coding/exchange/.cursor/debug.log")
    {
        let line = format!(
            r#"{{"sessionId":"debug-session","runId":"build","hypothesisId":"{}","location":"build.rs","message":"{}","data":{},"timestamp":{}}}"#,
            hypothesis_id, message, data, ts
        );
        let _ = writeln!(file, "{}", line);
    }
}

fn main() {
    let feature_python = std::env::var("CARGO_FEATURE_PYTHON").is_ok();
    let crate_type = std::env::var("CARGO_CRATE_TYPE").unwrap_or_else(|_| "unknown".to_string());

    // #region agent log
    log_build(
        "H1",
        "feature_flags",
        &format!(r#"{{"python_feature":{}}}"#, feature_python),
    );
    // #endregion

    // #region agent log
    log_build(
        "H2",
        "crate_type",
        &format!(r#"{{"crate_type":"{}"}}"#, crate_type),
    );
    // #endregion
}
