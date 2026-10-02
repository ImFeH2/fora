use std::{
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use serde_json::Value;

const READY_TIMEOUT: Duration = Duration::from_secs(30);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub struct Launcher {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

pub fn launcher(development: bool, project: &Path, resources: &Path) -> Launcher {
    let env = vec![("FORA_PORT".to_string(), "0".to_string())];
    if development {
        let parent = project.parent().expect("the core project has a parent");
        Launcher {
            program: "uv".into(),
            args: vec![
                "run".into(),
                "--project".into(),
                project.to_string_lossy().into_owned(),
                "python".into(),
                "-m".into(),
                "fora".into(),
                "--webui-dir".into(),
                parent.join("webui/dist").to_string_lossy().into_owned(),
            ],
            env,
        }
    } else {
        Launcher {
            program: resources.join(if cfg!(windows) {
                "core/fora.exe"
            } else {
                "core/fora"
            }),
            args: Vec::new(),
            env,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Ready {
    pub port: u16,
    pub token: String,
}

pub fn parse_ready(line: &str) -> Result<Ready, String> {
    let message: Value =
        serde_json::from_str(line).map_err(|error| format!("Invalid ready line: {error}"))?;
    if message.get("type").and_then(Value::as_str) != Some("ready") {
        return Err("Invalid ready line: type is not ready".to_string());
    }
    let port = message
        .get("port")
        .and_then(Value::as_u64)
        .and_then(|port| u16::try_from(port).ok())
        .ok_or("Invalid ready line: missing port")?;
    let token = message
        .get("token")
        .and_then(Value::as_str)
        .ok_or("Invalid ready line: missing token")?
        .to_string();
    Ok(Ready { port, token })
}

pub(crate) fn spawn_kernel(launcher: Launcher) -> Result<(Child, Ready), String> {
    let mut command = Command::new(&launcher.program);
    command
        .args(&launcher.args)
        .envs(launcher.env.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("Failed to start Fora: {error}"))?;
    let stdout = child.stdout.take().expect("stdout is piped");
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        let _ = sender.send(lines.next().and_then(Result::ok));
        for _ in lines {}
    });
    let ready = match receiver.recv_timeout(READY_TIMEOUT) {
        Ok(Some(line)) => parse_ready(&line),
        Ok(None) => Err("Fora exited before it was ready".to_string()),
        Err(_) => Err("Fora did not start within 30 seconds".to_string()),
    };
    match ready {
        Ok(ready) => Ok((child, ready)),
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            Err(error)
        }
    }
}

pub(crate) fn shutdown_kernel(child: &mut Child) {
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(b"{\"method\":\"system.shutdown\"}\n");
    }
    let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
    while Instant::now() < deadline {
        if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    let _ = child.kill();
    let _ = child.wait();
}
