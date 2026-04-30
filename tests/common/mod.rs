//! Shared test helpers — spawn a server on a random port for integration tests.

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub struct TestServer {
    pub url: String,
    pub data_dir: String,
    child: Option<Child>,
}

impl TestServer {
    /// Spawn a fresh `./target/release/microresolve-studio` on a random port with a clean
    /// tmp data dir. Blocks until the server is ready (max 10s).
    pub fn spawn() -> Self {
        let port = pick_port();
        let unique = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let data_dir = format!("/tmp/microresolve_test_{}_{}_{}", pid, unique, port);
        let _ = std::fs::remove_dir_all(&data_dir);

        // Locate the server binary. Prefer release build (faster startup).
        let bin = if std::path::Path::new("./target/release/microresolve-studio").exists() {
            "./target/release/microresolve-studio"
        } else {
            "./target/debug/microresolve-studio"
        };

        let child = Command::new(bin)
            .args([
                "--port",
                &port.to_string(),
                "--no-open",
                "--data",
                &data_dir,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect(
                "failed to spawn server (build with `cargo build --release --features server`)",
            );

        let url = format!("http://localhost:{}", port);
        wait_for_ready(&url, Duration::from_secs(10));

        Self {
            url,
            data_dir,
            child: Some(child),
        }
    }

    /// Reqwest client for hitting endpoints.
    pub fn client(&self) -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = std::fs::remove_dir_all(&self.data_dir);
    }
}

fn pick_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("failed to find free port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn wait_for_ready(url: &str, timeout: Duration) {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(500))
        .build()
        .unwrap();
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(format!("{}/api/health", url)).send() {
            if resp.status().is_success() {
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!(
        "server at {} did not become ready within {:?}",
        url, timeout
    );
}

/// Helper: POST JSON, return status + body text.
pub fn post_json<T: serde::Serialize>(
    client: &reqwest::blocking::Client,
    url: &str,
    headers: &[(&str, &str)],
    body: &T,
) -> (u16, String) {
    let mut req = client.post(url).json(body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().expect("request failed");
    (resp.status().as_u16(), resp.text().unwrap_or_default())
}

/// Helper: GET, return status + body text.
pub fn get(
    client: &reqwest::blocking::Client,
    url: &str,
    headers: &[(&str, &str)],
) -> (u16, String) {
    let mut req = client.get(url);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().expect("request failed");
    (resp.status().as_u16(), resp.text().unwrap_or_default())
}

/// Helper: DELETE with JSON body.
pub fn delete_json<T: serde::Serialize>(
    client: &reqwest::blocking::Client,
    url: &str,
    headers: &[(&str, &str)],
    body: &T,
) -> u16 {
    let mut req = client.delete(url).json(body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().expect("request failed");
    resp.status().as_u16()
}

/// Helper: PATCH with JSON body.
pub fn patch_json<T: serde::Serialize>(
    client: &reqwest::blocking::Client,
    url: &str,
    headers: &[(&str, &str)],
    body: &T,
) -> u16 {
    let mut req = client.patch(url).json(body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().expect("request failed");
    resp.status().as_u16()
}

// Keep the import alive for tests that don't use it.
#[allow(dead_code)]
fn _unused() {
    let _ = BufReader::new(std::io::empty()).lines();
}
