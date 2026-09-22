use std::{fmt::Write as _, path::Path, process::Child, sync::Mutex};

use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};

mod kernel;

pub use kernel::{Launcher, Ready, launcher, parse_ready};
use kernel::{shutdown_kernel, spawn_kernel};

pub fn window_url(outcome: Result<&Ready, &str>) -> String {
    match outcome {
        Ok(ready) => format!(
            "index.html?ws=ws://127.0.0.1:{}/ws&token={}",
            ready.port,
            url_encode(&ready.token)
        ),
        Err(message) => format!("index.html?error={}", url_encode(message)),
    }
}

fn url_encode(value: &str) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~".contains(&byte) {
            encoded.push(byte as char);
        } else {
            let _ = write!(encoded, "%{byte:02X}");
        }
    }
    encoded
}

struct Kernel(Mutex<Option<Child>>);

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .setup(|app| {
            let project = Path::new(env!("CARGO_MANIFEST_DIR")).join("../core");
            let launcher = launcher(tauri::is_dev(), &project, &app.path().resource_dir()?);
            let (child, url) = match spawn_kernel(launcher) {
                Ok((child, ready)) => (Some(child), window_url(Ok(&ready))),
                Err(error) => (None, window_url(Err(&error))),
            };
            app.manage(Kernel(Mutex::new(child)));
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App(url.into()))
                .disable_drag_drop_handler()
                .title("Huddol")
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                let kernel = app.state::<Kernel>();
                if let Some(child) = kernel.0.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                    shutdown_kernel(child);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn development_launcher_runs_the_core_project_with_uv() {
        let launcher = launcher(true, Path::new("repo/core"), Path::new("resources"));
        assert_eq!(launcher.program, Path::new("uv"));
        assert_eq!(
            launcher.args[..7],
            [
                "run",
                "--project",
                "repo/core",
                "python",
                "-m",
                "huddol",
                "--webui-dir"
            ]
        );
        assert_eq!(
            Path::new(&launcher.args[7]),
            Path::new("repo").join("webui/dist")
        );
    }

    #[test]
    fn bundled_launcher_runs_the_packaged_kernel() {
        let launcher = launcher(false, Path::new("repo/core"), Path::new("resources"));
        let expected = if cfg!(windows) {
            "core/huddol.exe"
        } else {
            "core/huddol"
        };
        assert_eq!(launcher.program, Path::new("resources").join(expected));
        assert!(launcher.args.is_empty());
    }

    #[test]
    fn every_launcher_lets_the_system_pick_the_port() {
        for development in [true, false] {
            let launcher = launcher(development, Path::new("repo/core"), Path::new("resources"));
            assert_eq!(launcher.env, [("HUDDOL_PORT".to_string(), "0".to_string())]);
        }
    }

    #[test]
    fn parses_a_valid_ready_line() {
        let ready = parse_ready(
            r#"{"type":"ready","port":4567,"token":"abc-_","url":"http://127.0.0.1:4567/?token=abc-_"}"#,
        )
        .unwrap();
        assert_eq!(
            ready,
            Ready {
                port: 4567,
                token: "abc-_".to_string()
            }
        );
    }

    #[test]
    fn rejects_ready_lines_with_missing_fields() {
        assert_eq!(
            parse_ready(r#"{"type":"ready","token":"abc"}"#).unwrap_err(),
            "Invalid ready line: missing port"
        );
        assert_eq!(
            parse_ready(r#"{"type":"ready","port":1}"#).unwrap_err(),
            "Invalid ready line: missing token"
        );
        assert_eq!(
            parse_ready(r#"{"type":"log","port":1,"token":"abc"}"#).unwrap_err(),
            "Invalid ready line: type is not ready"
        );
    }

    #[test]
    fn rejects_non_json_ready_lines() {
        assert!(
            parse_ready("Traceback (most recent call last)")
                .unwrap_err()
                .starts_with("Invalid ready line:")
        );
    }

    #[test]
    fn builds_the_window_url_from_the_ready_line() {
        let ready = Ready {
            port: 4567,
            token: "abc-_".to_string(),
        };
        assert_eq!(
            window_url(Ok(&ready)),
            "index.html?ws=ws://127.0.0.1:4567/ws&token=abc-_"
        );
    }

    #[test]
    fn builds_an_encoded_error_url_on_failure() {
        assert_eq!(
            window_url(Err(
                "Failed to start Huddol: No such file (os error 2) &x=1"
            )),
            "index.html?error=Failed%20to%20start%20Huddol%3A%20No%20such%20file%20%28os%20error%202%29%20%26x%3D1"
        );
    }

    #[test]
    fn app_url_with_query_resolves_under_both_base_urls() {
        let path = window_url(Err("boom"));
        let dev = tauri::Url::parse("http://localhost:1420")
            .unwrap()
            .join(&path)
            .unwrap();
        let bundled = tauri::Url::parse("tauri://localhost")
            .unwrap()
            .join(&path)
            .unwrap();
        assert_eq!(dev.as_str(), "http://localhost:1420/index.html?error=boom");
        assert_eq!(bundled.as_str(), "tauri://localhost/index.html?error=boom");
    }
}
