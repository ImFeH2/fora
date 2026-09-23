import { useEffect, useId, useRef, useState } from "react";
import { Button, Field, Input } from "@/components/ui/index";
import { backend } from "@/lib/backend";
import { VoiceRecording, type VoiceState } from "@/lib/voice";

type Values = {
  mode: "local" | "remote";
  address: string;
  model: string;
  api_key_set: boolean;
};

export function VoicePanel() {
  const id = useId();
  const [values, setValues] = useState<Values | null>(null);
  const [key, setKey] = useState("");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [present, setPresent] = useState(false);
  const [download, setDownload] = useState<number | null>(null);
  const [connected, setConnected] = useState(false);
  const [text, setText] = useState("");
  const [state, setState] = useState<VoiceState>("closed");
  const [level, setLevel] = useState(0);
  const socket = useRef<WebSocket | null>(null);
  const recording = useRef<VoiceRecording | null>(null);
  const active = state !== "closed";

  useEffect(() => {
    let mounted = true;
    const fail = (failure: unknown) => {
      if (mounted)
        setError(failure instanceof Error ? failure.message : String(failure));
    };
    void backend.settings("voice").then((value) => {
      if (mounted) setValues(value as Values);
    }, fail);
    void backend.voiceUrl().then((url) => {
      if (!mounted) return;
      const connection = new WebSocket(url);
      socket.current = connection;
      connection.onopen = () => {
        if (!mounted) return;
        setConnected(true);
        connection.send(JSON.stringify({ type: "model.status" }));
      };
      connection.onmessage = (event) => {
        if (!mounted) return;
        try {
          const message = JSON.parse(event.data);
          switch (message.type) {
            case "model.status":
              setPresent(message.present === true);
              break;
            case "model.progress":
              setDownload(message.received / message.total);
              break;
            case "model.downloaded":
              setPresent(true);
              setDownload(null);
              break;
            case "model.cancelled":
              setDownload(null);
              break;
            case "error":
              setDownload(null);
              setError(message.message);
              break;
            default:
              throw new Error("Unexpected model download event.");
          }
        } catch (failure) {
          fail(failure);
          connection.close();
        }
      };
      connection.onerror = () =>
        fail(new Error("Model download connection failed."));
      connection.onclose = () => {
        if (!mounted) return;
        setConnected(false);
        setDownload(null);
        fail(
          new Error(
            "Model download connection closed. Reopen Voice settings to reconnect.",
          ),
        );
      };
    }, fail);
    const off = backend.onEvent((event) => {
      if (event.type === "connection.closed") socket.current?.close();
    });
    return () => {
      mounted = false;
      off();
      const current = recording.current;
      recording.current = null;
      current?.cancel();
      socket.current?.close();
      socket.current = null;
    };
  }, []);

  const change = (next: Values) => {
    setValues(next);
    setDirty(true);
    setText("");
    setError("");
  };
  const save = async () => {
    if (!values || busy || active) return;
    setBusy(true);
    setError("");
    setText("");
    try {
      const update: Record<string, unknown> = {
        mode: values.mode,
        address: values.address,
        model: values.model,
      };
      if (key) update.api_key = key;
      setValues((await backend.updateSettings("voice", update)) as Values);
      setKey("");
      setDirty(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
  };
  const command = (type: string) => {
    const connection = socket.current;
    if (!connection || connection.readyState !== WebSocket.OPEN) {
      setError(
        "Model download connection is unavailable. Reopen Voice settings.",
      );
      return;
    }
    setError("");
    if (type === "model.download") setDownload(0);
    connection.send(JSON.stringify({ type }));
  };
  const test = () => {
    if (active || dirty || busy || !values) return;
    setText("");
    setError("");
    const current = new VoiceRecording((event) => {
      if (recording.current !== current) return;
      if (event.type === "state") {
        setState(event.state);
        if (event.state === "closed") {
          recording.current = null;
          setLevel(0);
        }
      } else if (event.type === "transcript") setText(event.text);
      else if (event.type === "level") setLevel(event.level);
      else setError(event.message);
    });
    recording.current = current;
    void current.start();
  };

  return (
    <div className="flex max-w-[560px] flex-col gap-4">
      {error ? (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      ) : null}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void save();
        }}
      >
        <fieldset
          disabled={!values || busy || active}
          className="m-0 flex min-w-0 flex-col gap-4 border-0 p-0"
        >
          <Field label="Transcription" htmlFor={`${id}-mode`}>
            <select
              id={`${id}-mode`}
              className="rounded border border-line bg-surface p-2"
              value={values?.mode ?? "local"}
              onChange={(event) => {
                if (!values) throw new Error("Voice settings are unavailable");
                change({
                  ...values,
                  mode: event.target.value as Values["mode"],
                });
              }}
            >
              <option value="local">Local · Whisper base</option>
              <option value="remote">Remote · OpenAI Realtime</option>
            </select>
          </Field>
          {values?.mode === "remote" ? (
            <>
              <Field label="Service address" htmlFor={`${id}-address`}>
                <Input
                  id={`${id}-address`}
                  type="url"
                  required
                  value={values.address}
                  onChange={(event) =>
                    change({ ...values, address: event.target.value })
                  }
                />
              </Field>
              <Field label="Model" htmlFor={`${id}-model`}>
                <Input
                  id={`${id}-model`}
                  required
                  value={values.model}
                  onChange={(event) =>
                    change({ ...values, model: event.target.value })
                  }
                />
              </Field>
              <Field
                label="API key"
                htmlFor={`${id}-key`}
                hint={
                  values.api_key_set
                    ? "Leave blank to keep the stored key."
                    : undefined
                }
              >
                <Input
                  id={`${id}-key`}
                  type="password"
                  autoComplete="off"
                  required={!values.api_key_set}
                  value={key}
                  onChange={(event) => {
                    setKey(event.target.value);
                    change(values);
                  }}
                />
              </Field>
              <p className="text-sm text-fg-muted">
                Audio is sent through Huddol to {values.address} for
                transcription.
              </p>
            </>
          ) : (
            <p className="text-sm text-fg-muted">
              Whisper runs on the computer hosting Huddol. Download the
              multilingual base model (141 MiB) to use local transcription.
            </p>
          )}
          <Button type="submit" variant="primary" disabled={!dirty}>
            Save
          </Button>
        </fieldset>
      </form>
      {values?.mode === "local" ? (
        <div className="flex flex-col gap-2">
          <p>{present ? "Whisper base downloaded" : "Whisper base required"}</p>
          {download !== null ? (
            <>
              <progress aria-label="Model download" max={1} value={download} />
              <Button type="button" onClick={() => command("model.cancel")}>
                Cancel download
              </Button>
            </>
          ) : (
            <Button
              type="button"
              disabled={!connected || active}
              onClick={() => command("model.download")}
            >
              {present ? "Download again" : "Download model"}
            </Button>
          )}
        </div>
      ) : null}
      <div className="flex flex-col gap-3 border-t border-line pt-4">
        <p className="text-sm">
          Test saved settings ·{" "}
          {values?.mode === "remote" ? values.address : "Local Whisper"}
        </p>
        {dirty ? (
          <p className="text-sm text-fg-muted">
            Save the configuration to test it.
          </p>
        ) : null}
        <div className="flex items-center gap-2">
          {active ? (
            <>
              <Button
                type="button"
                disabled={state === "finishing"}
                onClick={() => recording.current?.stop()}
              >
                Stop
              </Button>
              <Button type="button" onClick={() => recording.current?.cancel()}>
                Cancel
              </Button>
              <span role="status">
                {state === "starting"
                  ? "Starting…"
                  : state === "finishing"
                    ? "Transcribing…"
                    : "Listening…"}
              </span>
              <meter
                aria-label="Microphone volume"
                min={0}
                max={1}
                value={level}
              />
            </>
          ) : (
            <Button
              type="button"
              disabled={
                !values ||
                busy ||
                dirty ||
                download !== null ||
                (values.mode === "local" && !present)
              }
              onClick={test}
            >
              Start microphone test
            </Button>
          )}
        </div>
        <p
          className="min-h-16 whitespace-pre-wrap rounded border border-line p-3"
          aria-live="polite"
        >
          {text || "Transcribed text appears here."}
        </p>
      </div>
    </div>
  );
}
