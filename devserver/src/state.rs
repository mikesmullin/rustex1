use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::hub::{Hub, WsRoute};
use crate::{ApiHandler, RawHandler};

/// What a build operated on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// app sources → cargo build --target wasm32 + wasm-bindgen → web/pkg
    Wasm,
    /// server sources → cargo build → re-exec self
    Server,
    /// static assets → no build, just tell the client which file changed
    Web,
}

impl Target {
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Wasm => "wasm",
            Target::Server => "server",
            Target::Web => "web",
        }
    }
}

#[derive(Clone)]
pub struct BuildRecord {
    pub seq: u64,
    pub target: String,
    pub at_ms: u128,
    pub duration_ms: u128,
    pub ok: bool,
    pub trigger: Vec<String>,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct Activity {
    pub target: String,
    pub trigger: Vec<String>,
    pub started_ms: u128,
}

pub struct State {
    pub config: Config,
    /// Changes every time the server process starts. The HMR client compares it
    /// to decide between a hot swap and a full reload after a backend restart.
    pub boot_id: String,
    pub started: Instant,
    pub requests: AtomicU64,
    pub hub: Hub,
    pub builds: Mutex<Vec<BuildRecord>>,
    pub seq: AtomicU64,
    pub activity: Mutex<Option<Activity>>,
    pub last_error: Mutex<Option<String>>,
    /// App-registered JSON routes (`GET` only), installed once at startup.
    pub api: Arc<HashMap<String, ApiHandler>>,
    /// Websocket channels by path. `/__m_hmr` is in here too — the HMR socket is
    /// one route among them, not a special case in the request path.
    ///
    /// A `Mutex<HashMap>` rather than part of `State::new`'s arguments so an app
    /// can register channels without changing how every existing caller builds a
    /// `State`.
    pub ws: Mutex<HashMap<String, Arc<WsRoute>>>,
    /// Raw, prefix-matched routes: handlers that own the socket and may stream.
    ///
    /// Kept separate from `api` rather than folded into it. An `ApiHandler` is
    /// `-> Response`, i.e. a whole JSON document, and every existing handler
    /// depends on that. A streaming proxy has a different shape and a different
    /// failure mode, and pretending the two are one type buys a `Response` that
    /// cannot represent a stream.
    ///
    /// Sorted longest-prefix-first on insert so dispatch does not depend on the
    /// order the app happened to register them in.
    pub raw: Mutex<Vec<(String, Arc<RawHandler>)>>,
}

impl State {
    pub fn new(config: Config, api: HashMap<String, ApiHandler>) -> State {
        let boot_id = format!(
            "{:x}-{:x}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        State {
            config,
            boot_id,
            started: Instant::now(),
            requests: AtomicU64::new(0),
            hub: Hub::default(),
            builds: Mutex::new(Vec::new()),
            seq: AtomicU64::new(0),
            activity: Mutex::new(None),
            last_error: Mutex::new(None),
            api: Arc::new(api),
            ws: Mutex::new(HashMap::new()),
            raw: Mutex::new(Vec::new()),
        }
    }

    /// Register (or replace) a raw route under `prefix`.
    pub fn add_raw(&self, prefix: &str, handler: RawHandler) {
        let mut raw = self.raw.lock().unwrap();
        raw.retain(|(p, _)| p != prefix);
        raw.push((prefix.to_string(), Arc::new(handler)));
        raw.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
    }

    /// The raw route that serves `path`, if any. Longest matching prefix wins.
    pub fn raw_handler(&self, path: &str) -> Option<Arc<RawHandler>> {
        self.raw
            .lock()
            .unwrap()
            .iter()
            .find(|(prefix, _)| path.starts_with(prefix))
            .map(|(_, h)| Arc::clone(h))
    }

    /// Register (or replace) a websocket channel. Apps should go through
    /// [`Server::ws`](crate::Server::ws) instead; this exists so `run()` can
    /// install the dev server's own HMR route the same way.
    pub fn add_ws(&self, path: &str, route: Arc<WsRoute>) {
        self.ws
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(path.to_string(), route);
    }

    pub fn ws_route(&self, path: &str) -> Option<Arc<WsRoute>> {
        self.ws
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(path)
            .cloned()
    }

    /// Push one JSON message to every client of `path`, returning how many there
    /// were. Non-blocking: a stalled client queues its own messages and never
    /// holds up the caller — which matters for a 1 Hz feed with a laptop tab on a
    /// bad connection.
    pub fn ws_broadcast(&self, path: &str, msg: &serde_json::Value) -> usize {
        match self.ws_route(path) {
            Some(r) => {
                r.hub.broadcast(&msg.to_string());
                r.hub.count()
            }
            None => 0,
        }
    }

    pub fn ws_clients(&self, path: &str) -> usize {
        self.ws_route(path).map(|r| r.hub.count()).unwrap_or(0)
    }

    /// Paths with a live websocket channel, for the boot line and `/api/status`.
    pub fn ws_paths(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .ws
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .keys()
            .cloned()
            .collect();
        v.sort();
        v
    }

    pub fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub fn uptime_s(&self) -> u64 {
        self.started.elapsed().as_secs()
    }

    pub fn now_ms() -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    /// Send one JSON message to every connected `/__m_hmr` client.
    pub fn broadcast(&self, msg: serde_json::Value) {
        self.hub.broadcast(&msg.to_string());
    }

    pub fn record_build(&self, rec: BuildRecord) {
        let mut builds = self.builds.lock().unwrap();
        builds.push(rec);
        if builds.len() > 60 {
            let cut = builds.len() - 60;
            builds.drain(0..cut);
        }
    }

    pub fn set_activity(&self, a: Option<Activity>) {
        *self.activity.lock().unwrap() = a;
    }

    pub fn set_error(&self, e: Option<String>) {
        *self.last_error.lock().unwrap() = e.clone();
    }
}

static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Timestamped stdout line shared by the watcher, builder and http layers.
pub fn log(level: &str, msg: &str) {
    let secs = START.get_or_init(Instant::now).elapsed().as_secs_f32();
    println!("[{:>7.1}s] {:<7} {}", secs, level, msg);
}
