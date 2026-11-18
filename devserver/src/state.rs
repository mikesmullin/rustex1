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
    /// Path patterns that must answer **JSON** even when nothing serves them.
    ///
    /// A request that matches no `api` route, no built-in and no `raw` prefix falls
    /// through to the static handler, which serves `index.html` for any extensionless
    /// path so the m-js Router can run in path mode. That is right for a browser
    /// navigating to a client-side route and wrong for everything else: a typo'd
    /// machine path comes back `200 OK` + HTML, which reads as healthy to a load
    /// balancer and as a JSON parse error to an SDK.
    ///
    /// The `/api/` prefix is seeded by [`State::new`] — the case this mechanism was
    /// built for, and the reason it lives here rather than as another `starts_with`
    /// in the request path. Apps add their own with
    /// [`Server::json_only`](crate::Server::json_only) when they own a namespace the
    /// framework cannot enumerate: a gateway mounted at `/{preset}/v1/*` registers a
    /// route per configured profile, so an unknown profile matches nothing and would
    /// otherwise be served the SPA shell.
    ///
    /// A `Mutex<Vec<_>>` rather than a `State::new` argument, so registering these
    /// does not change how every existing caller builds a `State`.
    pub json_only: Mutex<Vec<String>>,
}

/// Match `path` against a [`State::json_only`] pattern, segment by segment.
///
/// `*` matches exactly one segment; a **trailing** `*` matches one or more, which is
/// what makes `/api/*` cover both `/api/x` and `/api/x/y`. Deliberately this small:
/// the only questions ever asked of these patterns are "is this the API namespace"
/// and "is this the gateway namespace", and a general glob engine would be a bigger
/// surface to get wrong than the handful of patterns that need matching.
///
/// Empty segments collapse, so `//api//x` matches `/api/*`. Harmless either way — no
/// route is registered with an empty segment — but stated because collapsing is the
/// kind of silence a matcher should not keep.
pub fn pattern_match(pattern: &str, path: &str) -> bool {
    // A nested fn rather than a closure: the closure's elided return lifetime cannot
    // relate `Vec<&str>` back to its `&str` argument, and an annotation there would
    // say less than this does.
    fn seg(s: &str) -> Vec<&str> {
        s.split('/').filter(|x| !x.is_empty()).collect()
    }
    let (p, x) = (seg(pattern), seg(path));
    for (i, pat) in p.iter().enumerate() {
        if *pat == "*" {
            // Trailing: consume everything left, but not nothing. `/api` is not under
            // `/api/`, and falling through to the static handler there is correct — a
            // bare `/api` is not an API route either.
            if i + 1 == p.len() {
                return x.len() > i;
            }
            if i >= x.len() {
                return false;
            }
            continue;
        }
        if i >= x.len() || x[i] != *pat {
            return false;
        }
    }
    // No trailing wildcard: the segment counts must agree exactly, so `/*/health`
    // claims `/qwen3-8/health` and not `/qwen3-8/sub/health`.
    p.len() == x.len()
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
            // The one rule the request path used to hardcode, now expressed through
            // the same matcher everything else uses. `/api/*` covers any path under
            // `/api/` with at least one segment after it — the exact set
            // `path.starts_with("/api/")` selected, including NOT matching a bare
            // `/api`, which the static handler serves as before.
            json_only: Mutex::new(vec!["/api/*".to_string()]),
        }
    }

    /// Register (or replace) a no-SPA-fallback pattern.
    pub fn add_json_only(&self, pattern: &str) {
        let mut v = self.json_only.lock().unwrap_or_else(|p| p.into_inner());
        v.retain(|p| p != pattern);
        v.push(pattern.to_string());
    }

    /// Does `path` match a registered pattern, such that a miss must be a JSON 404
    /// rather than the SPA shell?
    pub fn json_only_match(&self, path: &str) -> bool {
        let v = self.json_only.lock().unwrap_or_else(|p| p.into_inner());
        v.iter().any(|p| pattern_match(p, path))
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
