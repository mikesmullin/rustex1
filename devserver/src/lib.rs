//! # rustex1-devserver
//!
//! The reusable backend half of a rustex1 app: static files over several roots,
//! a small JSON API, the `/__m_hmr` websocket that drives m-js hot reload, the
//! filesystem watcher that rebuilds the wasm frontend, and self-restart when the
//! backend itself changes.
//!
//! An app's `src/main.rs` is meant to be this short:
//!
//! ```ignore
//! use rustex1_devserver::{Config, Server};
//! use serde_json::json;
//!
//! fn main() {
//!     Server::new(Config::discover(&std::env::current_dir().unwrap()))
//!         .api("/api/hello", |state, _req| json!({
//!             "message": "hello from rust",
//!             "requests": state.requests.load(Ordering::Relaxed),
//!         }))
//!         .run();
//! }
//! ```
//!
//! The only built-in routes are `/api/status`, `/api/builds`, `/api/config` and
//! the `/__m_hmr` socket; anything else you add comes from [`Server::api`] or,
//! for a push channel, [`Server::ws`].

mod build;
mod config;
mod http;
mod hub;
mod state;
mod watcher;

pub use build::{wasm_urls, Outcome};
pub use config::Config;
pub use http::{Request, Response, Stream, MAX_BODY};
pub use hub::{Hub, OnMessage, WsRoute};
pub use state::{Activity, BuildRecord, State, Target};

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use state::log;

pub type Json = serde_json::Value;

/// A handler gets the shared state plus the parsed request (method, path, query,
/// and — for write routes — the JSON body), and returns a [`Response`]. A bare
/// `serde_json::Value` is accepted too and becomes a 200.
pub type ApiHandler = Box<dyn Fn(&Arc<State>, &Request) -> Response + Send + Sync + 'static>;

/// A handler for a **prefix**, owning the response socket.
///
/// Registered with [`Server::raw`]. Unlike an [`ApiHandler`] it is handed a
/// [`Stream`], so it can send the status line, then frames, then the terminator —
/// which is the only way to proxy something that arrives over minutes. It is
/// prefix-matched rather than exact because the thing being proxied is a family
/// of paths (`/chat/completions`, `/completions`, `/models`, whatever the client
/// invents next), and an exact-match table is how you end up 405-ing a path you
/// never thought to enumerate.
pub type RawHandler = Box<dyn Fn(&Arc<State>, &Request, &mut Stream<'_>) + Send + Sync + 'static>;

/// How to serve an app's own websocket channel.
///
/// Deliberately push-first: a feed the browser only listens to needs nothing
/// more than a path, and the two optional hooks are here because some channels
/// do need to answer (a `ping`/`pong` keepalive the app defines itself) or to
/// greet a client with something other than nothing.
pub struct Ws {
    route: WsRoute,
}

impl Ws {
    /// A push-only channel: `state.ws_broadcast(path, &json!({…}))` reaches every
    /// client, and inbound frames are ignored.
    pub fn new(label: impl Into<String>) -> Ws {
        Ws {
            route: WsRoute::new(label),
        }
    }

    /// Bound per-client backlog and discard frames older than `max_age`. A slow
    /// live-feed subscriber must not turn into a history buffer. Writes also get
    /// this deadline so a wedged peer is disconnected rather than parked forever.
    pub fn live(mut self, max_pending: usize, max_age: Duration) -> Self {
        self.route.max_pending = max_pending.max(1);
        self.route.max_age = Some(max_age.max(Duration::from_millis(1)));
        self
    }

    /// Send `f()`'s value to a client the moment its handshake completes.
    pub fn hello<F>(mut self, f: F) -> Self
    where
        F: Fn(&Hub) -> serde_json::Value + Send + Sync + 'static,
    {
        self.route.hello = Some(std::sync::Arc::new(f));
        self
    }

    /// Handle an inbound text frame. Panics inside the handler are caught: one
    /// bad message must not kill the socket, and must never kill the server.
    pub fn on_message<F>(mut self, f: F) -> Self
    where
        F: Fn(&Arc<State>, &str) + Send + Sync + 'static,
    {
        self.route.on_message = Some(std::sync::Arc::new(f));
        self
    }
}

pub struct Server {
    config: Config,
    api: HashMap<String, ApiHandler>,
    raw: HashMap<String, RawHandler>,
    ws: HashMap<String, Ws>,
    on_start: Option<Box<dyn FnOnce(Arc<State>) + Send + 'static>>,
    force_rebuild: bool,
}

impl Server {
    pub fn new(config: Config) -> Server {
        Server {
            config,
            api: HashMap::new(),
            raw: HashMap::new(),
            ws: HashMap::new(),
            on_start: None,
            force_rebuild: false,
        }
    }

    /// Register an app route for any method (`GET`, `POST`, `PUT`, `PATCH`,
    /// `DELETE`). Handlers return a [`Response`], or a `serde_json::Value` for a
    /// plain 200:
    ///
    /// ```ignore
    /// .api("/api/tasks", |state, req| match req.method.as_str() {
    ///     "POST" => match req.json() {
    ///         Some(body) => tasks::create(state, &body),          // 201 / 400
    ///         None => Response::bad_request("expected a JSON body"),
    ///     },
    ///     _ => tasks::list(state, req),                            // 200
    /// })
    /// ```
    pub fn api<F, R>(mut self, path: &str, handler: F) -> Self
    where
        F: Fn(&Arc<State>, &Request) -> R + Send + Sync + 'static,
        R: Into<Response> + Send + 'static,
    {
        self.api
            .insert(path.to_string(), Box::new(move |s, r| handler(s, r).into()));
        self
    }

    /// Register a raw handler under `prefix`.
    ///
    /// Any request whose path starts with `prefix` is handed to the handler,
    /// which writes the response itself through a [`Stream`]. Longest prefix
    /// wins, so a catch-all and a specific override can coexist.
    ///
    /// ```ignore
    /// .raw("/qwen3-8-flash-next-vllm-850k/v1", |_state, req, out| {
    ///     gateway::handle(state, req, out);
    /// })
    /// ```
    ///
    /// Raw routes are matched **after** the exact `api` routes and the dev
    /// server's built-ins, so a prefix can never shadow `/api/status` or an app
    /// route it happens to overlap.
    pub fn raw<F>(mut self, prefix: &str, handler: F) -> Self
    where
        F: Fn(&Arc<State>, &Request, &mut Stream<'_>) + Send + Sync + 'static,
    {
        self.raw.insert(
            prefix.to_string(),
            Box::new(move |s, r, o| handler(s, r, o)),
        );
        self
    }

    /// Register an app-owned websocket channel at `path`.
    ///
    /// ```ignore
    /// .ws("/ws", Ws::new("feed").hello(|_| json!({ "type": "connected" })))
    /// ```
    ///
    /// Push to it from anywhere that holds the [`State`]:
    /// `state.ws_broadcast("/ws", &json!({ "type": "telem", … }))`.
    pub fn ws(mut self, path: &str, channel: Ws) -> Self {
        self.ws.insert(path.to_string(), channel);
        self
    }

    /// Run `f` with the shared [`State`] once the server is assembled — after
    /// routes are registered, before the listener is accepted.
    ///
    /// This exists because `State` is built inside [`Server::run`], and an app
    /// that owns a background thread (a feed bridge, a poller, a cache warmer)
    /// needs a handle to broadcast through. Without a hook the only alternatives
    /// were leaking the state into a global or not threading at all.
    ///
    /// It runs before the first connection is accepted, so anything the thread
    /// publishes is already in place when a browser asks for it.
    pub fn on_start<F>(mut self, f: F) -> Self
    where
        F: FnOnce(Arc<State>) + Send + 'static,
    {
        self.on_start = Some(Box::new(f));
        self
    }

    /// Rebuild the frontend on start even if nothing looks stale.
    pub fn rebuild_on_start(mut self) -> Self {
        self.force_rebuild = true;
        self
    }

    /// Build, watch, serve. Does not return.
    pub fn run(self) -> ! {
        let has_shell = self
            .config
            .statics
            .iter()
            .any(|root| root.join("index.html").is_file());
        if !has_shell {
            eprintln!(
                "rustex1: no index.html in any static root under {} — is this the app root?",
                self.config.root.display()
            );
            std::process::exit(2);
        }
        for route in self.api.keys() {
            log("api", &format!("GET {route}"));
        }
        for prefix in self.raw.keys() {
            log("raw", &format!("{prefix}* · streaming"));
        }
        // One move out of `self`: taking `ws` earlier and destructuring the rest
        // afterwards would be a partial move the borrow checker rightly rejects.
        let Server {
            config,
            api,
            raw,
            ws,
            on_start,
            force_rebuild,
        } = self;
        let app_ws: HashMap<String, Arc<WsRoute>> = ws
            .into_iter()
            .map(|(path, w)| {
                log("ws", &format!("{path} · {}", w.route.label));
                (path, Arc::new(w.route))
            })
            .collect();

        let state = Arc::new(State::new(config, api));
        // The HMR socket registers itself as a channel rather than being a
        // special case in the request path — one mechanism, one code path.
        state.add_ws(
            "/__m_hmr",
            Arc::new(WsRoute {
                label: "hmr".to_string(),
                hub: state.hub.clone(),
                hello: Some(crate::hub::hmr_hello(&state.boot_id)),
                on_message: None,
                max_pending: 64,
                max_age: None,
            }),
        );
        for (path, route) in app_ws {
            state.add_ws(&path, route);
        }
        for (prefix, handler) in raw {
            state.add_raw(&prefix, handler);
        }
        log(
            "boot",
            &format!(
                "{} v{} pid={} boot={} root={} wasm={}",
                state.config.name,
                env!("CARGO_PKG_VERSION"),
                std::process::id(),
                state.boot_id,
                state.config.root.display(),
                state.config.app.package,
            ),
        );
        if std::env::var("RUSTEX1_RESTARTED").is_ok() {
            log("boot", "restarted after a backend source change");
        }

        // 1 · the frontend must exist before the browser asks for it.
        if force_rebuild || build::wasm_is_stale(&state) {
            build::rebuild_wasm(&state, &["startup".to_string()]);
        } else {
            log("build", "reusing web/pkg/* (pass --rebuild to force)");
        }

        // 2 · sources, and whatever background work the app needs on the state.
        watcher::spawn(Arc::clone(&state));
        if let Some(f) = on_start {
            f(Arc::clone(&state));
        }

        // 3 · serve.
        let (js, wasm) = state.config.pkg_urls();
        let listener = bind(state.config.host, state.config.port);
        let bound = listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_else(|_| format!("{}:{}", state.config.host, state.config.port));
        log(
            "http",
            &format!(
                // Every live channel, not a hardcoded `/__m_hmr`: an app that
                // registers a websocket and cannot see it in the boot line has no
                // way to tell a typo'd path from a bridge that never started.
                "http://{bound}/  ·  ws {}  ·  {js} + {wasm}  ·  {} static root(s)",
                state.ws_paths().join(" + "),
                state.config.statics.len()
            ),
        );

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let state = Arc::clone(&state);
                    std::thread::spawn(move || {
                        // A panicking connection must not take the server down.
                        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            http::serve(stream, state);
                        }));
                    });
                }
                Err(e) => log("warn", &format!("accept failed: {e}")),
            }
        }

        std::process::exit(0);
    }
}

/// Hand the process over to the freshly built backend binary, keeping the port.
pub fn restart(state: &Arc<State>) {
    build::restart(state)
}

/// After a self-restart the old process may still hold the port for a moment.
fn bind(host: std::net::Ipv4Addr, port: u16) -> TcpListener {
    let addr = std::net::SocketAddr::from((host, port));
    if !addr.ip().is_loopback() {
        // Binding beyond loopback is a deliberate act, so the boot log says so and
        // names the half of the decision that lives outside this process. A dev
        // server is the whole backend for these apps; an app that has not thought
        // about authorization should not discover at 11pm that it has been
        // reachable from the network since it started.
        eprintln!(
            "rustex1: WARNING — bound to {addr}, reachable beyond loopback. This is \
             only safe when the port is published to loopback as well \
             (`-p 127.0.0.1:{port}:{port}` for docker/podman), leaving a reverse \
             proxy on this host as the only way in."
        );
    }
    let mut last = None;
    for attempt in 0..50 {
        match TcpListener::bind(addr) {
            Ok(l) => return l,
            Err(e) => {
                last = Some(e);
                std::thread::sleep(Duration::from_millis(100));
                if attempt == 4 {
                    log(
                        "boot",
                        &format!("port {port} busy, waiting for the old server to exit…"),
                    );
                }
            }
        }
    }
    eprintln!("rustex1: cannot bind {addr}: {}", last.unwrap());
    std::process::exit(1);
}

/// CLI convenience: `--port N`, `--rebuild`, or a bare port number.
pub fn args_to_config(args: &[String], mut config: Config) -> (Config, bool) {
    let mut rebuild = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                if let Some(v) = args.get(i + 1) {
                    if let Ok(p) = v.parse() {
                        config.port = p;
                    }
                    i += 1;
                }
            }
            "--rebuild" => rebuild = true,
            "--host" => {
                if let Some(v) = args.get(i + 1) {
                    // Ignored unless it parses: the default is the safe address, and
                    // a typo should not silently start a server on 0.0.0.0.
                    if let Ok(h) = v.parse::<std::net::Ipv4Addr>() {
                        config.host = h;
                    }
                    i += 1;
                }
            }
            "--root" => {
                if let Some(v) = args.get(i + 1) {
                    config.root = PathBuf::from(v);
                    i += 1;
                }
            }
            other => {
                if let Ok(p) = other.parse::<u16>() {
                    config.port = p;
                }
            }
        }
        i += 1;
    }
    (config, rebuild)
}
