//! Websocket hubs — `/__m_hmr` and any channel an app registers with
//! [`Server::ws`](crate::Server::ws).
//!
//! One implementation serves both: the HMR socket is just the route the dev
//! server registers for itself. Before that it was a hardcoded path inside
//! `http::serve`, which left an app with no way to push anything to the browser
//! except by polling — and a polling "feed" is how a 1 Hz telemetry page ends up
//! either lagging or hammering the server.
//!
//! Wire protocol is the one m-js speaks (see `docs/index.md` → "Adding HMR to a
//! server you already have"):
//!
//! ```jsonc
//! { "type": "connected", "bootId": "…" }        // server -> client, on open
//! { "type": "change",  "path": "/styles.css" }  // server -> client, per file
//! ```
//!
//! plus three messages this prototype adds (a stock m-js client ignores them):
//!
//! ```jsonc
//! { "type": "build",   "phase": "start" | "ok" | "error", … }
//! { "type": "restart", "target": "server" }
//! ```

use std::collections::HashMap;
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tungstenite::{Error, Message};

use crate::state::log;

const POLL: Duration = Duration::from_millis(20);
const HEARTBEAT: Duration = Duration::from_secs(30);

#[derive(Default, Clone)]
pub struct Hub(Arc<Mutex<HashMap<u64, SyncSender<Queued>>>>);

#[derive(Clone)]
struct Queued { text: String, at: Instant }
impl Queued {
    fn fresh(&self, max_age: Option<Duration>) -> bool {
        max_age.is_none_or(|age|self.at.elapsed()<=age)
    }
}

static SEQ: AtomicU64 = AtomicU64::new(1);

impl Hub {
    fn register(&self, tx: SyncSender<Queued>) -> u64 {
        let id = SEQ.fetch_add(1, Ordering::SeqCst);
        self.0.lock().unwrap().insert(id, tx);
        id
    }

    fn remove(&self, id: u64) {
        self.0.lock().unwrap().remove(&id);
    }

    pub fn count(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    /// Non-blocking: a stalled client never blocks the watcher thread. Messages
    /// are queued per connection and drained by that connection's poll loop.
    pub fn broadcast(&self, text: &str) {
        let senders: Vec<SyncSender<Queued>> = self.0.lock().unwrap().values().cloned().collect();
        for tx in senders {
            // Slow viewers cannot back-pressure the producer or grow an
            // unbounded replay backlog. Live feeds can opt into a short TTL too.
            let _ = tx.try_send(Queued {text:text.to_string(),at:Instant::now()});
        }
    }
}

/// Inbound text frame handler. `None` means the channel is push-only, which is
/// what most app feeds are: the browser never has to say anything.
pub type OnMessage = Arc<dyn Fn(&Arc<crate::state::State>, &str) + Send + Sync + 'static>;

/// Builds the message handed to a client immediately after the handshake.
/// Takes the hub so a hello can carry the live client count, which is the whole
/// reason this is a closure and not a value: a baked-in number would tell every
/// new connection it was the only one.
pub type Hello = Arc<dyn Fn(&Hub) -> serde_json::Value + Send + Sync + 'static>;

/// One registered websocket channel.
pub struct WsRoute {
    /// Log label — `"hmr"`, or an app's own name. Every line this file writes is
    /// labelled with it, because "client connected" without a channel is
    /// ambiguous the moment a server has two.
    pub label: String,
    pub hub: Hub,
    pub hello: Option<Hello>,
    pub on_message: Option<OnMessage>,
    pub max_pending: usize,
    pub max_age: Option<Duration>,
}

impl WsRoute {
    pub fn new(label: impl Into<String>) -> WsRoute {
        WsRoute {
            label: label.into(),
            hub: Hub::default(),
            hello: None,
            on_message: None,
            max_pending: 64,
            max_age: None,
        }
    }
}

/// Serve one websocket connection. Blocks until the peer goes away.
pub fn session(stream: TcpStream, app: Arc<crate::state::State>, route: Arc<WsRoute>) {
    let hub = route.hub.clone();
    let _ = stream.set_nodelay(true);
    // The opening handshake must block normally; the 20 ms poll clock only
    // starts once the connection is upgraded.
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));

    let mut ws = match tungstenite::accept(stream) {
        Ok(ws) => ws,
        Err(err) => {
            log("warn", &format!("hmr handshake failed: {err}"));
            return;
        }
    };
    let _ = ws.get_ref().set_read_timeout(Some(POLL));

    if let Some(age)=route.max_age { let _=ws.get_ref().set_write_timeout(Some(age)); }
    let (tx, rx) = mpsc::sync_channel::<Queued>(route.max_pending.max(1));
    let id = hub.register(tx);

    // A channel with no hello opens silently. That is the right default for an
    // app feed: the client knows what it asked for, and an unsolicited frame
    // before the first data message is just something for the app to ignore.
    if let Some(hello) = route.hello.as_ref() {
        if push(&mut ws, &hello(&hub).to_string()).is_err() {
            hub.remove(id);
            return;
        }
    }
    log(&route.label, &format!("client connected ({})", hub.count()));

    let mut last_beat = Instant::now();
    loop {
        // 1 · anything the watcher queued for this socket
        loop {
            match rx.try_recv() {
                Ok(frame) => {
                    if !frame.fresh(route.max_age) {continue;}
                    if push(&mut ws, &frame.text).is_err() {
                        hub.remove(id);
                        return;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    hub.remove(id);
                    return;
                }
            }
        }

        // 2 · unidirectional heartbeat; browsers answer pings for us
        if last_beat.elapsed() >= HEARTBEAT {
            last_beat = Instant::now();
            let _ = ws.write(Message::Ping(Vec::new().into()));
        }
        let _ = ws.flush();

        // 3 · drive the protocol (close handshake, queued pongs, incoming frames)
        match ws.read() {
            Ok(Message::Close(_)) => break,
            Ok(Message::Text(t)) => {
                // Inbound text is the app's business. A handler that panics must
                // not take the connection down with it, and must not be able to
                // stall this socket's poll loop, so it runs inline but guarded.
                if let Some(f) = route.on_message.as_ref() {
                    let text = t.to_string();
                    let st = Arc::clone(&app);
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        f(&st, &text)
                    }));
                }
            }
            Ok(_) => {}
            Err(Error::Io(e))
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(Error::WriteBufferFull(_)) => {}
            Err(_) => break,
        }
    }

    hub.remove(id);
    log(&route.label, &format!("client left ({})", hub.count()));
}

/// The hello m-js expects on the HMR socket.
pub fn hmr_hello(boot_id: &str) -> Hello {
    let boot_id = boot_id.to_string();
    Arc::new(move |hub: &Hub| {
        serde_json::json!({
            "type": "connected",
            "bootId": boot_id,
            "server": "rustex1",
            "clients": hub.count(),
        })
    })
}

fn push<W: std::io::Read + std::io::Write>(
    ws: &mut tungstenite::WebSocket<W>,
    text: &str,
) -> Result<(), Error> {
    ws.write(Message::text(text.to_string()))?;
    ws.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_viewer_never_stalls_producer_or_accumulates_history() {
        let hub=Hub::default();let(tx,rx)=mpsc::sync_channel(2);let id=hub.register(tx);
        for _ in 0..10000 {hub.broadcast("frame");}
        assert_eq!(rx.try_iter().count(),2);hub.remove(id);
    }
    #[test]
    fn old_live_frames_are_not_replayed() {
        let old=Queued{text:"old".into(),at:Instant::now()-Duration::from_secs(2)};
        assert!(!old.fresh(Some(Duration::from_secs(1))));
        assert!(old.fresh(None));
    }
}
