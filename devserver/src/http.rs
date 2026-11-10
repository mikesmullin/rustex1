//! A deliberately small HTTP/1.1 server: static files over several roots, a
//! handful of built-in JSON routes, whatever the app registered with
//! [`Server::api`](crate::Server::api), and the websocket channels registered with
//! [`Server::ws`](crate::Server::ws) — of which `/__m_hmr` is one.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::build;
use crate::hub;
use crate::state::{log, State};

/// A parsed request head, plus its body.
///
/// `Default` exists so a handler can be tested against a request without a
/// socket. Every field is empty by default, which is what a request with no
/// method, no path and no body would mean anyway.
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub query: String,
    /// Raw request body, read up to `MAX_BODY` bytes when `Content-Length` is
    /// present. Must be consumed even if unused: closing a socket with unread
    /// input sends a TCP RST and the client loses the response.
    pub body: Vec<u8>,
    pub content_type: String,
    /// `Content-Length` as parsed from the request head.
    pub content_length: usize,
    /// The request head, `(lowercased name, value)`, in wire order.
    ///
    /// The JSON routes never needed this: their whole contract is a path and a
    /// JSON body. A proxy does — an OpenAI-compatible endpoint authenticates on
    /// `Authorization` and forwards `Content-Type`, and a client's identity is a
    /// header, not a query parameter. Names are lowercased because HTTP field
    /// names are case-insensitive, and a handler that compares them literally is
    /// a bug waiting for the next client to send `Content-type`.
    pub headers: Vec<(String, String)>,
}

/// Default ceiling for a request body handed to an `api` route (1 MiB).
///
/// Kept as a public constant for compatibility, but it is now only the *default*:
/// the effective limit is `Config::max_body_bytes`, set per app from
/// `rustex1.json`. An app that proxies uploads (an AI gateway carrying base64
/// images is ~10 MB before it is interesting) raises it deliberately instead of
/// discovering it as a truncated document.
pub const DEFAULT_MAX_BODY: usize = 1 << 20;

/// Back-compat alias for the old constant name.
pub const MAX_BODY: usize = DEFAULT_MAX_BODY;

/// An app route's reply: a status plus a JSON body. Plain `serde_json::Value`
/// also works (it converts with status 200), so read-only handlers stay short.
pub struct Response {
    pub status: u16,
    pub body: serde_json::Value,
}

impl Response {
    pub fn new(status: u16, body: serde_json::Value) -> Response {
        Response { status, body }
    }
    pub fn ok(body: serde_json::Value) -> Response {
        Response { status: 200, body }
    }
    pub fn created(body: serde_json::Value) -> Response {
        Response { status: 201, body }
    }
    /// `{"ok": false, "error": msg}` with status 400.
    pub fn bad_request(msg: impl Into<String>) -> Response {
        Response {
            status: 400,
            body: serde_json::json!({ "ok": false, "error": msg.into() }),
        }
    }
    pub fn not_found(msg: impl Into<String>) -> Response {
        Response {
            status: 404,
            body: serde_json::json!({ "ok": false, "error": msg.into() }),
        }
    }
    pub fn server_error(msg: impl Into<String>) -> Response {
        Response {
            status: 500,
            body: serde_json::json!({ "ok": false, "error": msg.into() }),
        }
    }
}

impl From<serde_json::Value> for Response {
    fn from(body: serde_json::Value) -> Response {
        Response { status: 200, body }
    }
}

/// A response the handler writes incrementally.
///
/// [`Response`] can only express a JSON endpoint, because a JSON endpoint *is*
/// a whole document. A proxy cannot be one: an inference generation arrives over
/// minutes as SSE frames, and the caller has to see each frame as it lands —
/// buffering it to parse the last one would turn a live stream into a timeout.
/// So a raw handler owns the socket through this:
///
/// ```ignore
/// .raw("/qwen/v1", |_state, req, out| {
///     let _ = out.start(200, &[("content-type", "text/event-stream")]);
///     for frame in upstream(req) { let _ = out.chunk(&frame); }
///     let _ = out.finish();
/// })
/// ```
///
/// Framing is chunked on purpose. For a stream the byte count is not knowable
/// up front, and `Connection: close` framing would make a generation that died
/// halfway indistinguishable from one that finished — the truncation would look
/// like the model's fault. The terminating `0\r\n\r\n` is what says *complete*.
///
/// A handler that returns without ever calling `start` gets a 500 rather than a
/// silently open socket.
pub struct Stream<'a> {
    stream: &'a mut TcpStream,
    started: bool,
    finished: bool,
    status: u16,
    head_only: bool,
}

impl<'a> Stream<'a> {
    /// Wrap a socket the caller owns.
    ///
    /// Public so an app can exercise its own raw handler against a socket it
    /// controls instead of standing up a listener to get one indirectly. The dev
    /// server itself only ever constructs one from a connection it accepted.
    pub fn new(stream: &'a mut TcpStream) -> Stream<'a> {
        Stream {
            stream,
            started: false,
            finished: false,
            status: 0,
            head_only: false,
        }
    }

    /// Write the status line and headers. Call once, before any chunk.
    ///
    /// `Transfer-Encoding: chunked` and `Connection: close` are added; every
    /// other header is exactly what the handler supplied, because a proxy that
    /// rewrites `content-type` is how an SSE stream arrives as a blob.
    pub fn start(&mut self, status: u16, headers: &[(&str, &str)]) -> std::io::Result<()> {
        if self.started {
            return Ok(());
        }
        let mut head = format!("HTTP/1.1 {status} {}\r\n", reason(status));
        for (name, value) in headers {
            if name.contains(['\r', '\n', ':']) || value.contains(['\r', '\n']) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "invalid response header",
                ));
            }
            if ["content-length", "transfer-encoding", "connection"]
                .iter()
                .any(|n| name.eq_ignore_ascii_case(n))
            {
                continue;
            }
            head.push_str(&format!("{name}: {value}\r\n"));
        }
        if status == 204 || status == 304 {
            self.head_only = true;
        }
        if !self.head_only {
            head.push_str("Transfer-Encoding: chunked\r\n");
        }
        head.push_str("Connection: close\r\n\r\n");
        self.started = true;
        self.status = status;
        self.stream.write_all(head.as_bytes())
    }

    /// One chunked frame. Auto-starts as `200 text/plain` if `start` was never
    /// called, so a handler that only knows bytes still produces a valid
    /// response instead of a headerless socket.
    pub fn chunk(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if !self.started {
            self.start(200, &[("content-type", "text/plain; charset=utf-8")])?;
        }
        if self.head_only {
            return Ok(());
        }
        self.stream
            .write_all(format!("{:x}\r\n", bytes.len()).as_bytes())?;
        self.stream.write_all(bytes)?;
        self.stream.write_all(b"\r\n")
    }

    /// A complete response with no body — a 204, or a status-only refusal.
    pub fn empty(&mut self, status: u16) -> std::io::Result<()> {
        self.start(status, &[])?;
        self.finish()
    }

    /// `application/json` in one frame. For the refusals a raw route makes
    /// before it ever reaches the upstream.
    pub fn json(&mut self, status: u16, body: &serde_json::Value) -> std::io::Result<()> {
        let text = body.to_string();
        self.start(status, &[("content-type", "application/json")])?;
        self.chunk(text.as_bytes())?;
        self.finish()
    }

    /// Close the framing. Safe to call twice; the second call does nothing.
    pub fn finish(&mut self) -> std::io::Result<()> {
        if self.finished {
            return Ok(());
        }
        if !self.started {
            return self.json(
                500,
                &serde_json::json!({"ok": false, "error": "raw handler returned without writing a response"}),
            );
        }
        self.finished = true;
        if self.head_only {
            return Ok(());
        }
        self.stream.write_all(b"0\r\n\r\n")
    }

    /// Abort a partial response without writing the successful chunk terminator.
    /// Dispatch calls `finish` after a raw handler returns; this suppresses it.
    pub fn abort(&mut self) {
        self.finished = true;
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }

    /// Detect abandonment while queued or waiting for the next upstream frame.
    /// No bytes are consumed. The short timeout is restored before returning.
    pub fn disconnected(&mut self) -> bool {
        let old = self.stream.read_timeout().ok().flatten();
        let _ = self.stream.set_read_timeout(Some(Duration::from_millis(1)));
        let result = self.stream.peek(&mut [0u8; 1]);
        let _ = self.stream.set_read_timeout(old);
        match result {
            Ok(0) => true,
            Err(e) => !matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::TimedOut
                    | std::io::ErrorKind::Interrupted
            ),
            _ => false,
        }
    }

    /// Has a status line gone out yet?
    /// change its answer, and pretending otherwise is how a truncated generation
    /// gets reported as a 200.
    pub fn started(&self) -> bool {
        self.started
    }

    pub fn status(&self) -> u16 {
        self.status
    }
}

impl Request {
    /// Parse the body as JSON, `None` when absent or malformed.
    pub fn json(&self) -> Option<serde_json::Value> {
        if self.body.is_empty() {
            return None;
        }
        serde_json::from_slice(&self.body).ok()
    }

    /// A query parameter, decoded.
    pub fn param(&self, key: &str) -> Option<String> {
        query_param(&self.query, key)
    }

    /// A request header value, by case-insensitive name.
    ///
    /// First match wins, which is what a repeated header means for the fields
    /// this server cares about (`Authorization`, `Content-Type`): the client sent
    /// one, and a second is a proxy artifact.
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k == &name)
            .map(|(_, v)| v.as_str())
    }
}

pub fn serve(mut stream: TcpStream, state: Arc<State>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_nodelay(true);
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));

    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    // `peek` never advances the socket, so every call reports the *same* prefix of
    // what has arrived. Appending each peek therefore duplicates the head: a
    // request head longer than the peek buffer came back as the first 2048 bytes
    // repeated until it hit the 64 KiB ceiling, and `find_head_end` then reported
    // an offset inside the duplication, so the drain that follows ate body bytes.
    // Small curl requests never showed it — they fit in one peek.
    //
    // So: replace, and grow. A peek that fills the buffer means the head is
    // longer than the buffer; a peek that fills less means the head is split
    // across segments. A short peek is immediately readable again (it does NOT
    // block waiting for more), so back off and enforce an absolute deadline.
    let head_deadline = std::time::Instant::now() + Duration::from_secs(30);
    let mut peek_len = 2048usize;
    let head_end = loop {
        let mut chunk = vec![0u8; peek_len];
        match stream.peek(&mut chunk) {
            Ok(0) => return,
            Ok(n) => {
                chunk.truncate(n);
                buf.clear();
                buf.extend_from_slice(&chunk);
                if let Some(pos) = find_head_end(&buf) {
                    break pos;
                }
                if buf.len() >= 64 * 1024 {
                    let _ = respond(&mut stream, 431, "text/plain; charset=utf-8", b"", false);
                    return;
                }
                if n == peek_len {
                    peek_len = (peek_len * 2).min(64 * 1024);
                } else {
                    std::thread::sleep(Duration::from_millis(2));
                }
                if std::time::Instant::now() >= head_deadline {
                    let _ = respond(
                        &mut stream,
                        408,
                        "text/plain",
                        b"request head timeout",
                        false,
                    );
                    return;
                }
            }
            Err(_) => return,
        }
    };

    let text = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let Some(mut req) = parse_request(&text) else {
        let _ = respond(
            &mut stream,
            400,
            "text/plain; charset=utf-8",
            b"bad request",
            false,
        );
        return;
    };

    state
        .requests
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let looks_like_upgrade = wants_websocket_upgrade(&text);

    // Websocket upgrade: hand the connection to the channel registered for this
    // path. The request bytes stay unread so tungstenite can perform its own
    // handshake parse. `/__m_hmr` arrives here like any other channel; an
    // unknown `Upgrade` gets a plain 426 rather than a dropped socket, so a
    // typo'd path shows up as an error in the browser instead of a hang.
    if looks_like_upgrade {
        match state.ws_route(&req.path) {
            Some(route) => {
                hub::session(stream, Arc::clone(&state), route);
                return;
            }
            None => {
                let _ = stream.write_all(
                    b"HTTP/1.1 426 Upgrade Required\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                return;
            }
        }
    }

    let head_only = req.method == "HEAD";

    // We only *peeked* the request. Consume the header bytes now (see the RST
    // note on Request::body) and then the body, so the socket is always drained
    // before it closes.
    if head_end > 0 {
        let mut drained = vec![0u8; head_end];
        let _ = stream.read_exact(&mut drained);
    }
    if req.header("transfer-encoding").is_some() {
        // Chunked uploads are not implemented. Never hand an empty/truncated
        // body to a JSON handler as if it were a legitimate request.
        let _ = respond(
            &mut stream,
            501,
            "application/json",
            b"{\"error\":\"chunked request bodies are not supported; send Content-Length\"}",
            head_only,
        );
        return;
    }
    if req.content_length > 0 {
        if req
            .header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
            && req.content_length <= state.config.max_body_bytes
        {
            let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
        }
        let limit = state.config.max_body_bytes;
        if req.content_length > limit {
            // Refuse, do not truncate. Handing a handler the first `limit` bytes
            // of a larger body is indistinguishable from a small, malformed
            // request: the JSON fails to parse — or worse, parses as something
            // shorter than what was sent — and every symptom points at the
            // handler while the real cause is a transport ceiling. A 413 that
            // names the limit is debuggable in one read.
            //
            // The remaining bytes are deliberately not drained. Reading a body we
            // have already rejected costs the full transfer for nothing, and the
            // response below closes the connection, so the backlog dies with the
            // socket. (See the RST note on `Request::body` for why draining is
            // normally required and why it is not here.)
            let msg = format!(
                "request body is {} bytes; this server accepts at most {} \
                 (raise maxBodyBytes in rustex1.json)",
                req.content_length, limit
            );
            log(
                "warn",
                &format!("{} {} → 413 {}", req.method, req.path, msg),
            );
            let _ = respond(
                &mut stream,
                413,
                "application/json",
                format!("{{\"ok\":false,\"error\":\"{msg}\"}}").as_bytes(),
                head_only,
            );
            return;
        }
        req.body = vec![0u8; req.content_length];
        if stream.read_exact(&mut req.body).is_err() {
            return;
        }
    }

    let result = route(&mut stream, &req, &state, head_only);
    if let Err(e) = result {
        log("warn", &format!("{} {} → {}", req.method, req.path, e));
    }
}

fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

/// Does this request head ask for a websocket upgrade?
///
/// Detected from the raw head rather than by matching the path, so a plain `GET`
/// to a websocket path is served normally instead of being handed to a handshake
/// parser that cannot complete it. The old code keyed on
/// `req.path == "/__m_hmr"` alone, which meant `curl localhost:port/__m_hmr`
/// produced a handshake warning rather than a 404.
///
/// `Connection` is deliberately not consulted: proxies rewrite it, and
/// `keep-alive, Upgrade` is legal. If `Upgrade` names websocket, upgrade.
fn wants_websocket_upgrade(head: &str) -> bool {
    head.lines().skip(1).any(|line| match line.split_once(':') {
        Some((k, v)) => {
            k.trim().eq_ignore_ascii_case("upgrade")
                && v.trim()
                    .split(',')
                    .any(|p| p.trim().eq_ignore_ascii_case("websocket"))
        }
        None => false,
    })
}

fn parse_request(text: &str) -> Option<Request> {
    let mut lines = text.lines();
    let request_line = lines.next()?.trim();
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_uppercase();
    let target = parts.next()?;

    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.to_string(), String::new()),
    };

    let mut content_length = 0usize;
    let mut content_type = String::new();
    let mut headers: Vec<(String, String)> = Vec::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim().to_string();
        match name.as_str() {
            "content-length" => {
                // Reject ambiguous framing instead of guessing which length a
                // downstream proxy meant. Duplicate framing/auth fields fail closed.
                if headers.iter().any(|(k, _)| k == "content-length") {
                    return None;
                }
                content_length = value.parse().ok()?;
                headers.push((name, value));
            }
            "content-type" => content_type = value.clone(),
            // A folded/continuation line (`\tvalue`) has no colon and is skipped,
            // same as the old parser did. Obssolete, and worse to guess at.
            _ => {
                if name == "authorization" && headers.iter().any(|(k, _)| k == "authorization") {
                    return None;
                }
                headers.push((name, value));
            }
        }
    }
    // `Content-Type` is kept in both places: `content_type` because every
    // existing handler reads that field, and `headers` because a raw handler
    // forwarding the request must be able to see the head it was given.
    if !content_type.is_empty() {
        headers.push(("content-type".to_string(), content_type.clone()));
    }

    Some(Request {
        method,
        path: percent_decode(&path),
        query,
        body: Vec::new(),
        content_type,
        content_length,
        headers,
    })
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("00");
                out.push(u8::from_str_radix(hex, 16).unwrap_or(b'%'));
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn query_param(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (k == key).then(|| percent_decode(v))
    })
}

fn route(
    stream: &mut TcpStream,
    req: &Request,
    state: &Arc<State>,
    head_only: bool,
) -> std::io::Result<()> {
    if req.method == "OPTIONS" {
        return respond_extra(
            stream,
            204,
            "text/plain",
            b"",
            head_only,
            &[
                ("Access-Control-Allow-Origin", "*"),
                ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "content-type"),
            ],
        );
    }

    // App-registered routes win over the built-ins of the same name. HEAD is
    // treated as a GET so a header-only probe does not fall through to the SPA
    // shell and come back as text/html.
    if let Some(handler) = state.api.get(&req.path) {
        if matches!(
            req.method.as_str(),
            "GET" | "HEAD" | "POST" | "PUT" | "PATCH" | "DELETE"
        ) {
            let resp = handler(state, req);
            return json_response(stream, resp.status, &resp.body, head_only);
        }
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET" | "HEAD", "/api/status") => {
            let activity = state.activity.lock().unwrap().clone();
            let last_error = state.last_error.lock().unwrap().clone();
            let (js, wasm) = state.config.pkg_urls();
            let json = serde_json::json!({
                "ok": true,
                "app": state.config.name,
                "server": "rustex1-devserver",
                "version": env!("CARGO_PKG_VERSION"),
                "port": state.config.port,
                "bootId": state.boot_id,
                "uptimeS": state.uptime_s(),
                "requests": state.requests.load(std::sync::atomic::Ordering::Relaxed),
                "clients": state.hub.count(),
                "root": state.config.root.display().to_string(),
                "wasmPackage": state.config.app.package,
                "wasmProfile": state.config.app.profile,
                "watching": {
                    "wasm": state.config.app.watch,
                    "server": state.config.server.watch,
                    "static": state.config.static_watch.iter().map(|p| rel(state, p)).collect::<Vec<_>>(),
                },
                "building": activity.map(|a| serde_json::json!({
                    "target": a.target, "trigger": a.trigger,
                    "elapsedMs": State::now_ms() - a.started_ms,
                })),
                "lastError": last_error,
                "artifacts": [js, wasm],
            });
            return json_response(stream, 200, &json, head_only);
        }
        ("GET" | "HEAD", "/api/config") => {
            // What web/app.mjs needs to know before it imports anything.
            let (js, wasm) = state.config.pkg_urls();
            return json_response(
                stream,
                200,
                &serde_json::json!({
                    "app": state.config.name,
                    "pkgJs": js,
                    "pkgWasm": wasm,
                    "hmrPath": "/__m_hmr",
                    "bootId": state.boot_id,
                }),
                head_only,
            );
        }
        ("GET" | "HEAD", "/api/builds") => {
            let limit = query_param(&req.query, "limit")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(20)
                .min(60);
            let builds = state.builds.lock().unwrap();
            let items: Vec<serde_json::Value> = builds
                .iter()
                .rev()
                .take(limit)
                .map(|b| {
                    serde_json::json!({
                        "seq": b.seq, "target": b.target, "atMs": b.at_ms, "ms": b.duration_ms,
                        "ok": b.ok, "trigger": b.trigger, "error": b.error,
                    })
                })
                .collect();
            return json_response(
                stream,
                200,
                &serde_json::json!({ "builds": items }),
                head_only,
            );
        }
        ("POST", "/api/rebuild") => {
            let out = build::rebuild_wasm(state, &["POST /api/rebuild".to_string()]);
            return json_response(
                stream,
                if out.ok { 200 } else { 500 },
                &serde_json::json!({
                    "ok": out.ok, "ms": out.ms, "changed": out.changed, "error": out.error,
                }),
                head_only,
            );
        }
        ("POST", "/api/restart") => {
            let state = Arc::clone(state);
            let _ = json_response(
                stream,
                200,
                &serde_json::json!({ "ok": true, "restarting": true }),
                head_only,
            );
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(250));
                build::restart(&state);
            });
            return Ok(());
        }
        _ => {}
    }

    // Raw, prefix-matched routes — before the method gate, because a proxy is
    // POST-heavy by nature and the gate below exists to stop a POST from falling
    // through into the SPA shell. That fallthrough is what an unimplemented
    // gateway used to look like: `POST /:preset/v1/chat/completions` answered
    // 405 "method not allowed" from the framework, which reads as a broken
    // client rather than a feature that does not exist yet.
    //
    // Longest prefix wins, so an app can register a catch-all and a specific
    // override without the registration order deciding which one answers.
    if let Some(handler) = state.raw_handler(&req.path) {
        let mut out = Stream::new(stream);
        out.head_only = head_only;
        handler(state, req, &mut out);
        return out.finish();
    }

    if req.method != "GET" && req.method != "HEAD" {
        return respond(
            stream,
            405,
            "text/plain; charset=utf-8",
            b"method not allowed",
            head_only,
        );
    }

    // Never SPA-fallback an API path: a typo'd route otherwise comes back as
    // 200 + index.html, which surfaces as a JSON parse error instead of a 404.
    if req.path.starts_with("/api/") {
        return json_response(
            stream,
            404,
            &serde_json::json!({
                "ok": false,
                "error": format!("no route for {} {} — register it with Server::api()", req.method, req.path),
            }),
            head_only,
        );
    }

    match static_file(state, &req.path) {
        Some((path, mime)) => {
            let body = std::fs::read(&path)?;
            respond_extra(
                stream,
                200,
                &mime,
                &body,
                head_only,
                &[("Cache-Control", "no-store")],
            )
        }
        None => respond(
            stream,
            404,
            "text/plain; charset=utf-8",
            format!("404 {}\n", req.path).as_bytes(),
            head_only,
        ),
    }
}

fn rel(state: &State, p: &PathBuf) -> String {
    p.strip_prefix(&state.config.root)
        .map(|r| r.to_string_lossy().to_string())
        .unwrap_or_else(|_| p.display().to_string())
}

/// Resolve a URL path against the static roots in priority order — the app's
/// own `web/` first, so it can shadow any kit asset (`app.mjs`, `hot.mjs`,
/// `kit.css`, even `vendor/phosphor/*`) without copying it. Extension-less
/// paths fall back to `index.html` so the m-js Router can run in path mode.
fn static_file(state: &State, url_path: &str) -> Option<(PathBuf, String)> {
    let rel = url_path.trim_start_matches('/');
    if rel.split('/').any(|seg| seg == "..") {
        return None;
    }

    let mut fallback = None;
    for root in &state.config.statics {
        let mut candidate = root.join(rel);
        if candidate.is_dir() {
            candidate = candidate.join("index.html");
        }
        if candidate.is_file() {
            let mime = mime_type(&candidate).to_string();
            return Some((candidate, mime));
        }
        if fallback.is_none() && !PathBuf::new().join(rel).extension().is_some() {
            let index = root.join("index.html");
            if index.is_file() {
                fallback = Some(index);
            }
        }
    }

    fallback.map(|p| {
        let mime = mime_type(&p).to_string();
        (p, mime)
    })
}

fn mime_type(path: &PathBuf) -> &'static str {
    // Getting these right matters: a .mjs or .css served as application/json
    // silently kills module imports and <link rel=stylesheet>.
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "wasm" => "application/wasm",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}

fn json_response(
    stream: &mut TcpStream,
    status: u16,
    value: &serde_json::Value,
    head: bool,
) -> std::io::Result<()> {
    let body = value.to_string();
    respond_extra(
        stream,
        status,
        "application/json; charset=utf-8",
        body.as_bytes(),
        head,
        &[
            ("Cache-Control", "no-store"),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
}

fn respond(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    head: bool,
) -> std::io::Result<()> {
    respond_extra(
        stream,
        status,
        content_type,
        body,
        head,
        &[("Cache-Control", "no-store")],
    )
}

fn respond_extra(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
    head: bool,
    extra: &[(&str, &str)],
) -> std::io::Result<()> {
    let mut out = Vec::with_capacity(body.len() + 256);
    out.extend_from_slice(format!("HTTP/1.1 {} {}\r\n", status, reason(status)).as_bytes());
    out.extend_from_slice(format!("Content-Type: {content_type}\r\n").as_bytes());
    out.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    for (k, v) in extra {
        out.extend_from_slice(format!("{k}: {v}\r\n").as_bytes());
    }
    out.extend_from_slice(b"Connection: close\r\n\r\n");
    if !head {
        out.extend_from_slice(body);
    }
    stream.write_all(&out)?;
    stream.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Content Too Large",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

// The request path is the part of a dev server that a unit test cannot fake:
// `peek` semantics, head framing and chunked output only show their bugs against
// a real socket. These tests drive `serve` over loopback for exactly that reason.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::collections::HashMap;
    use std::net::TcpListener;

    fn state() -> Arc<State> {
        let config = Config::discover(&std::env::current_dir().unwrap());
        Arc::new(State::new(config, HashMap::new()))
    }

    /// Serve on a free port; one thread per connection, like the real thing.
    fn serve_one(state: Arc<State>) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(s) => serve(s, Arc::clone(&state)),
                    Err(_) => break,
                }
            }
        });
        port
    }

    /// Every response these tests read is `Connection: close`, so EOF is the end
    /// of the body and `read_to_end` is the whole exchange.
    fn round_trip(port: u16, bytes: &[u8]) -> String {
        let mut sock = TcpStream::connect(("127.0.0.1", port)).unwrap();
        sock.write_all(bytes).unwrap();
        sock.flush().unwrap();
        let mut out = Vec::new();
        sock.read_to_end(&mut out).unwrap();
        String::from_utf8_lossy(&out).into_owned()
    }

    fn echo_body_len(st: &Arc<State>, prefix: &str) {
        st.add_raw(
            prefix,
            Box::new(|_s, req, out| {
                let _ = out.json(
                    200,
                    &serde_json::json!({
                        "path": req.path,
                        "body": req.body.len(),
                        "filler": req.header("x-filler").map(str::len),
                        "auth": req.header("authorization"),
                    }),
                );
            }),
        );
    }

    #[test]
    fn a_raw_handler_sees_the_headers_a_proxy_needs() {
        let st = state();
        st.add_raw(
            "/echo",
            Box::new(|_s, req, out| {
                let _ = out.json(
                    200,
                    &serde_json::json!({
                        "auth": req.header("Authorization"),
                        "ct": req.header("content-type"),
                        "session": req.header("x-mycloud-session"),
                    }),
                );
            }),
        );
        let port = serve_one(Arc::clone(&st));
        let res = round_trip(
            port,
            b"POST /echo/v1/chat HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer mc-token\r\nContent-Type: application/json\r\nX-Mycloud-Session: 2026-10-20\r\nContent-Length: 2\r\n\r\n{}",
        );
        // Case-insensitive on both sides: the client wrote `Authorization` and
        // `X-Mycloud-Session`, the handler asked for lowercase.
        assert!(res.contains("mc-token"), "{res}");
        assert!(res.contains("application/json"), "{res}");
        assert!(res.contains("2026-10-20"), "{res}");
    }

    #[test]
    fn a_head_bigger_than_the_peek_buffer_is_parsed_once() {
        // 6 KiB of header: past the 2048-byte peek, past the doubling. The old
        // loop appended each peek to the buffer, and since `peek` never advances
        // the socket it appended the *same* bytes again and again — the head grew
        // until the 64 KiB ceiling answered 431, or `find_head_end` reported an
        // offset inside the duplication and the drain ate the body.
        let st = state();
        echo_body_len(&st, "/echo");
        let port = serve_one(Arc::clone(&st));

        let filler = "x".repeat(6000);
        let body = "{\"a\":1}";
        let raw = format!(
            "POST /echo HTTP/1.1\r\nHost: x\r\nX-Filler: {filler}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let res = round_trip(port, raw.as_bytes());
        assert!(res.contains("\"body\":7"), "body was eaten: {res}");
        assert!(res.contains("\"filler\":6000"), "head was mangled: {res}");
        assert!(!res.contains("431"), "rejected a legal head: {res}");
    }

    #[test]
    fn a_head_split_across_segments_is_not_duplicated() {
        let st = state();
        echo_body_len(&st, "/echo");
        let port = serve_one(Arc::clone(&st));

        let whole =
            "POST /echo HTTP/1.1\r\nHost: x\r\nX-Filler: yyy\r\nContent-Length: 2\r\n\r\n{}";
        let (head, tail) = whole.split_at(24); // mid-header: no terminator yet
        let mut sock = TcpStream::connect(("127.0.0.1", port)).unwrap();
        sock.write_all(head.as_bytes()).unwrap();
        sock.flush().unwrap();
        std::thread::sleep(Duration::from_millis(80));
        sock.write_all(tail.as_bytes()).unwrap();
        sock.flush().unwrap();
        let mut out = Vec::new();
        sock.read_to_end(&mut out).unwrap();
        let res = String::from_utf8_lossy(&out).into_owned();
        assert!(
            res.contains("\"filler\":3"),
            "split head parsed wrong: {res}"
        );
        assert!(res.contains("\"body\":2"), "split head ate the body: {res}");
    }

    #[test]
    fn a_post_to_a_raw_prefix_is_not_405() {
        // The shape Mike hit: with no gateway registered, `POST /:preset/v1/...`
        // fell through to the framework's method gate and answered
        // `405 method not allowed` — a framework artifact dressed up as a
        // client bug. Registering the prefix must take the POST.
        let st = state();
        echo_body_len(&st, "/preset/v1");
        let port = serve_one(Arc::clone(&st));
        let res = round_trip(
            port,
            b"POST /preset/v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 2\r\n\r\n{}",
        );
        assert!(res.starts_with("HTTP/1.1 200"), "{res}");
        assert!(!res.contains("405"), "{res}");
    }

    #[test]
    fn a_raw_handler_streams_frames_and_terminates_them() {
        let st = state();
        st.add_raw(
            "/stream",
            Box::new(|_s, _r, out| {
                let _ = out.start(200, &[("content-type", "text/event-stream")]);
                for frame in ["data: one\n\n", "data: two\n\n"] {
                    let _ = out.chunk(frame.as_bytes());
                }
                let _ = out.finish();
            }),
        );
        let port = serve_one(Arc::clone(&st));
        let res = round_trip(port, b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(res.contains("HTTP/1.1 200"), "{res}");
        assert!(res.contains("Transfer-Encoding: chunked"), "{res}");
        assert!(res.contains("content-type: text/event-stream"), "{res}");
        // Both frames, in order, and the zero-length terminator that is the only
        // thing distinguishing "the stream ended" from "the stream died".
        let one = res.find("data: one").unwrap();
        let two = res.find("data: two").unwrap();
        assert!(one < two, "frames reordered: {res}");
        assert!(res.ends_with("0\r\n\r\n"), "unterminated stream: {res}");
    }

    #[test]
    fn the_longest_matching_prefix_wins_regardless_of_registration_order() {
        let st = state();
        st.add_raw(
            "/a",
            Box::new(|_s, _r, out| {
                let _ = out.json(200, &serde_json::json!({"hit": "short"}));
            }),
        );
        st.add_raw(
            "/a/b",
            Box::new(|_s, _r, out| {
                let _ = out.json(200, &serde_json::json!({"hit": "long"}));
            }),
        );
        let port = serve_one(Arc::clone(&st));
        assert!(round_trip(port, b"GET /a/b/c HTTP/1.1\r\nHost: x\r\n\r\n").contains("long"));
        assert!(round_trip(port, b"GET /a/z HTTP/1.1\r\nHost: x\r\n\r\n").contains("short"));
    }

    #[test]
    fn an_exact_api_route_beats_a_raw_prefix_it_overlaps() {
        // A gateway registered under a broad prefix must not be able to swallow
        // the dev server's own endpoints or an app's JSON route.
        let mut api: HashMap<String, crate::ApiHandler> = HashMap::new();
        api.insert(
            "/api/special".to_string(),
            Box::new(|_s, _r| Response::ok(serde_json::json!({"won": "api"}))),
        );
        let st = Arc::new(State::new(
            Config::discover(&std::env::current_dir().unwrap()),
            api,
        ));
        st.add_raw(
            "/api/",
            Box::new(|_s, _r, out| {
                let _ = out.json(200, &serde_json::json!({"won": "raw"}));
            }),
        );
        let port = serve_one(Arc::clone(&st));
        let res = round_trip(port, b"GET /api/special HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(res.contains("\"won\":\"api\""), "{res}");
        // …and the prefix still serves everything else under it.
        let other = round_trip(port, b"GET /api/other HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(other.contains("\"won\":\"raw\""), "{other}");
    }

    #[test]
    fn a_handler_that_writes_nothing_is_a_500_not_an_open_socket() {
        let st = state();
        st.add_raw("/silent", Box::new(|_s, _r, _out| {}));
        let port = serve_one(Arc::clone(&st));
        let res = round_trip(port, b"GET /silent HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(res.contains("HTTP/1.1 500"), "{res}");
        assert!(res.contains("without writing a response"), "{res}");
    }

    #[test]
    fn raw_abort_never_emits_the_successful_terminator() {
        let st = state();
        st.add_raw(
            "/cut",
            Box::new(|_, _, out| {
                out.chunk(b"partial").unwrap();
                out.abort();
            }),
        );
        let res = round_trip(serve_one(st), b"GET /cut HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(res.contains("partial"));
        assert!(!res.ends_with("0\r\n\r\n"));
    }

    #[test]
    fn raw_head_has_no_body_and_chunked_uploads_are_honestly_refused() {
        let st = state();
        echo_body_len(&st, "/echo");
        let port = serve_one(st);
        let head = round_trip(port, b"HEAD /echo HTTP/1.1\r\nHost: x\r\n\r\n");
        assert!(head.ends_with("\r\n\r\n"));
        assert!(!head.contains("\"body\""));
        let chunked = round_trip(
            port,
            b"POST /echo HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n",
        );
        assert!(chunked.starts_with("HTTP/1.1 501"));
    }

    #[test]
    fn ambiguous_lengths_and_duplicate_authorization_fail_closed() {
        assert!(parse_request("POST /x HTTP/1.1\r\nContent-Length: nope\r\n\r\n").is_none());
        assert!(parse_request(
            "POST /x HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\n"
        )
        .is_none());
        assert!(parse_request(
            "GET /x HTTP/1.1\r\nAuthorization: Bearer a\r\nAuthorization: Bearer b\r\n\r\n"
        )
        .is_none());
    }

    #[test]
    fn oversized_bodies_are_still_refused_before_the_handler_runs() {
        // The 413 gate predates this change and must survive it: a raw handler
        // that is handed a truncated body cannot tell it from a short one.
        let st = state();
        echo_body_len(&st, "/echo");
        let port = serve_one(Arc::clone(&st));
        let n = st.config.max_body_bytes + 10;
        let raw = format!("POST /echo HTTP/1.1\r\nHost: x\r\nContent-Length: {n}\r\n\r\n");
        let res = round_trip(port, raw.as_bytes());
        assert!(res.contains("413"), "{res}");
        assert!(res.contains("maxBodyBytes"), "{res}");
    }
}
