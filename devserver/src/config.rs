//! Everything an app can vary, in one file: `rustex1.json` at the project root.
//!
//! ```jsonc
//! {
//!   "name": "tasks1",                 // also the default for the crate names
//!   "port": 3010,
//!   "app":    { "package": "tasks1-app", "lib": "tasks1_app",
//!               "profile": "wasm", "watch": ["app"] },
//!   "server": { "package": "tasks1-server", "watch": ["server"] },
//!   "static": ["web"],                // served first; wins over the kit's assets
//!   "kitWeb": "vendor/rustex1/kit/web" // optional: where app.mjs / hot.mjs / kit.css live
//! }
//! ```
//!
//! Every field has a default derived from `name`, so a two-line file is enough.

use std::path::{Path, PathBuf};

/// Default place the kit keeps the assets an app is allowed to shadow.
pub const KIT_WEB_CANDIDATES: &[&str] = &[
    "kit/web",                // rustex1 itself
    "vendor/rustex1/kit/web", // consumed as a git submodule
    "../rustex1/kit/web",     // sibling checkout
    "../kit/web",
];

#[derive(Clone, Debug)]
pub struct AppSide {
    /// cargo package to build for wasm32, e.g. `tasks1-app`
    pub package: String,
    /// lib name inside it — decides `/pkg/<lib>.js` and `/pkg/<lib>_bg.wasm`
    pub lib: String,
    /// cargo profile used for the wasm build
    pub profile: String,
    /// directories whose `.rs`/`Cargo.toml` changes trigger a wasm rebuild
    pub watch: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ServerSide {
    pub package: String,
    pub watch: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub name: String,
    pub port: u16,
    /// Interface to bind. **Defaults to loopback**, which is a security property,
    /// not a convenience: the dev server is the whole backend for these apps, and
    /// an app that has not thought about authorization should not be reachable
    /// from the network by default.
    ///
    /// The one legitimate reason to widen it is the container case. Inside a
    /// container the app must answer the *docker bridge*, so `127.0.0.1` will not
    /// do — but that is only safe when the published mapping is itself loopback
    /// (`-p 127.0.0.1:PORT:PORT`), so the host-facing socket stays on the host's
    /// loopback and a reverse proxy on that host remains the only way in. Binding
    /// here without narrowing the publish exports an unauthenticated control plane
    /// to whatever can route to the host. The two settings are one decision.
    pub host: std::net::Ipv4Addr,
    /// project root — the directory holding `rustex1.json`
    pub root: PathBuf,
    pub app: AppSide,
    pub server: ServerSide,
    /// static roots in priority order (app first, kit last)
    pub statics: Vec<PathBuf>,
    /// directories watched for *static* changes (no rebuild, just a change msg)
    pub static_watch: Vec<PathBuf>,
    /// Largest request body an `api` route will be handed, in bytes.
    ///
    /// Default is 1 MiB — right for a JSON API, far too small for anything that
    /// accepts uploads. An app that proxies image or audio payloads sets
    /// `"maxBodyBytes"` in `rustex1.json` to its own ceiling.
    ///
    /// Over the limit the server answers **413** rather than handing the handler
    /// a body that has been quietly cut off mid-document. Truncating is the worse
    /// failure: the JSON no longer parses, or worse, parses as something shorter
    /// than what was sent, and every symptom downstream points at the handler
    /// instead of at a transport limit.
    pub max_body_bytes: usize,
}

impl Config {
    /// Look for `rustex1.json` at/above `start`; fall back to name-based defaults.
    pub fn discover(start: &Path) -> Config {
        let mut dir = Some(start.to_path_buf());
        while let Some(d) = dir {
            let file = d.join("rustex1.json");
            if file.is_file() {
                match Config::load(&file) {
                    Ok(c) => return c,
                    Err(e) => {
                        eprintln!("rustex1: {}: {e}", file.display());
                        std::process::exit(2);
                    }
                }
            }
            dir = d.parent().map(PathBuf::from);
        }
        // No manifest: infer a root from the layout and take the defaults.
        let root = guess_root(start);
        Config::from_json(&root, &serde_json::json!({}))
    }

    pub fn load(file: &Path) -> Result<Config, String> {
        let text = std::fs::read_to_string(file)
            .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("invalid JSON in {}: {e}", file.display()))?;
        Ok(Config::from_json(
            &file.parent().unwrap().to_path_buf(),
            &json,
        ))
    }

    fn from_json(root: &Path, j: &serde_json::Value) -> Config {
        let s = |path: &[&str], default: &str| -> String {
            lookup(j, path)
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| default.to_string())
        };
        let n = |path: &[&str], default: u16| -> u16 {
            lookup(j, path)
                .and_then(|v| v.as_u64())
                .map(|v| v as u16)
                .unwrap_or(default)
        };
        let list = |path: &[&str], default: &[&str]| -> Vec<String> {
            match lookup(j, path).and_then(|v| v.as_array()) {
                Some(arr) => arr
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect(),
                None => default.iter().map(|s| s.to_string()).collect(),
            }
        };

        // A bad `host` is ignored rather than fatal, mirroring how `port` is read:
        // a typo in a config file should not stop a dev server from starting, and
        // the fallback is the safe value.
        let host = lookup(j, &["host"])
            .and_then(|v| v.as_str())
            .and_then(|v| v.parse().ok())
            .unwrap_or(std::net::Ipv4Addr::LOCALHOST);

        let name = s(&["name"], "rustex1");
        let app_package = s(&["app", "package"], &format!("{name}-app"));
        let app_lib = s(&["app", "lib"], &app_package.replace('-', "_"));
        let server_package = s(&["server", "package"], &format!("{name}-server"));

        let statics_rel = list(&["static"], &["web"]);
        let mut statics: Vec<PathBuf> = statics_rel.iter().map(|p| root.join(p)).collect();

        // The kit's own assets (app.mjs, hot.mjs, kit.css, vendor/) come last so
        // an app can shadow any of them by putting a file of the same name in its
        // own web/ — no copy step, and a `git submodule update` refreshes them.
        let kit_web = lookup(j, &["kitWeb"])
            .and_then(|v| v.as_str())
            .map(|p| root.join(p))
            .or_else(|| find_kit_web(root));
        if let Some(kit) = &kit_web {
            if kit.is_dir() && !statics.contains(kit) {
                statics.push(kit.clone());
            }
        }

        let static_watch: Vec<PathBuf> = statics.iter().filter(|p| p.is_dir()).cloned().collect();

        Config {
            name,
            port: n(&["port"], 3000),
            host,
            root: root.to_path_buf(),
            app: AppSide {
                package: app_package,
                lib: app_lib,
                profile: s(&["app", "profile"], "wasm"),
                watch: list(&["app", "watch"], &["app"]),
            },
            server: ServerSide {
                package: server_package,
                watch: list(&["server", "watch"], &["server"]),
            },
            statics,
            static_watch,
            // Bytes, not the u16 helper the other numeric knobs use — a body
            // ceiling is comfortably larger than 65535 and `as_u64` would have
            // silently truncated anything above a u16's range.
            max_body_bytes: lookup(j, &["maxBodyBytes"])
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .filter(|v| *v > 0)
                .unwrap_or(crate::http::DEFAULT_MAX_BODY),
        }
    }

    /// `/pkg/<lib>.js` + `/pkg/<lib>_bg.wasm` — the URLs the browser loads.
    pub fn pkg_urls(&self) -> (String, String) {
        (
            format!("/pkg/{}.js", self.app.lib),
            format!("/pkg/{}_bg.wasm", self.app.lib),
        )
    }

    /// Where wasm-bindgen writes the glue + binary. Always the *app's* first
    /// static root, never the kit's — generated files belong to the app.
    pub fn pkg_dir(&self) -> PathBuf {
        self.statics
            .first()
            .map(|p| p.join("pkg"))
            .unwrap_or_else(|| self.root.join("web/pkg"))
    }

    /// Absolute path of the wasm cargo emits before wasm-bindgen runs.
    pub fn wasm_artifact(&self) -> PathBuf {
        self.root.join(format!(
            "target/wasm32-unknown-unknown/{}/{}.wasm",
            self.app.profile, self.app.lib
        ))
    }

    /// Directories whose sources belong to the wasm half.
    pub fn app_watch_abs(&self) -> Vec<PathBuf> {
        self.app.watch.iter().map(|p| self.root.join(p)).collect()
    }

    /// Directories whose sources belong to the backend half.
    pub fn server_watch_abs(&self) -> Vec<PathBuf> {
        self.server
            .watch
            .iter()
            .map(|p| self.root.join(p))
            .collect()
    }

    pub fn has_wasm_sources(&self) -> bool {
        self.app_watch_abs().iter().any(|p| p.is_dir())
    }
}

fn lookup<'a>(j: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut cur = j;
    for key in path {
        cur = cur.get(*key)?;
    }
    Some(cur)
}

fn find_kit_web(root: &Path) -> Option<PathBuf> {
    KIT_WEB_CANDIDATES
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.is_dir())
}

/// Used when there is no `rustex1.json`: accept a directory as the root only if
/// it looks like a rustex1 app.
fn guess_root(start: &Path) -> PathBuf {
    let mut dir = Some(start.to_path_buf());
    while let Some(d) = dir {
        if d.join("rustex1.json").is_file()
            || (d.join("web/index.html").is_file() && d.join("Cargo.toml").is_file())
        {
            return d;
        }
        dir = d.parent().map(PathBuf::from);
    }
    start.to_path_buf()
}
