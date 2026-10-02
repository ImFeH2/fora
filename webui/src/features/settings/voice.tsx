import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Button, Field, Input, Spinner } from "@/components/ui/index";
import { backend } from "@/lib/backend";
import { VoiceRecording, type VoiceState } from "@/lib/voice";

type Values = {
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
  const [loadFailed, setLoadFailed] = useState(false);
  const [text, setText] = useState("");
  const [state, setState] = useState<VoiceState>("closed");
  const [level, setLevel] = useState(0);
  const recording = useRef<VoiceRecording | null>(null);
  const dirtyRef = useRef(false);
  const active = state !== "closed";

  const load = useCallback(async () => {
    if (dirtyRef.current) return;
    try {
      const next = (await backend.settings("voice")) as Values;
      if (dirtyRef.current) return;
      setValues(next);
      setDirty(false);
      setLoadFailed(false);
      setError("");
    } catch (failure) {
      setLoadFailed(true);
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }, []);

  useEffect(() => {
    void load();
    const off = backend.onEvent((event) => {
      if (event.type === "connection.restored") void load();
    });
    return () => {
      off();
      const current = recording.current;
      recording.current = null;
      current?.cancel();
    };
  }, [load]);

  const change = (next: Values) => {
    dirtyRef.current = true;
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
        address: values.address,
        model: values.model,
      };
      if (key) update.api_key = key;
      const next = (await backend.updateSettings("voice", update)) as Values;
      setValues(next);
      setKey("");
      dirtyRef.current = false;
      setDirty(false);
      setLoadFailed(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setBusy(false);
    }
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
        <div
          role="alert"
          className="flex items-center gap-2 text-sm text-danger"
        >
          <span className="min-w-0 flex-1">{error}</span>
          {loadFailed ? (
            <Button size="sm" onClick={() => void load()}>
              Retry
            </Button>
          ) : null}
        </div>
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
          <Field label="Service address" htmlFor={`${id}-address`}>
            <Input
              id={`${id}-address`}
              type="url"
              required
              value={values?.address ?? ""}
              onChange={(event) => {
                if (!values) throw new Error("Voice settings are unavailable");
                change({ ...values, address: event.target.value });
              }}
            />
          </Field>
          <Field label="Model" htmlFor={`${id}-model`}>
            <Input
              id={`${id}-model`}
              required
              value={values?.model ?? ""}
              onChange={(event) => {
                if (!values) throw new Error("Voice settings are unavailable");
                change({ ...values, model: event.target.value });
              }}
            />
          </Field>
          <Field
            label="API key"
            htmlFor={`${id}-key`}
            hint={
              values?.api_key_set
                ? "Leave blank to keep the stored key."
                : undefined
            }
          >
            <Input
              id={`${id}-key`}
              type="password"
              autoComplete="off"
              required={!values?.api_key_set}
              value={key}
              onChange={(event) => {
                setKey(event.target.value);
                if (values) change(values);
              }}
            />
          </Field>
          <p className="text-sm text-fg-muted">
            Audio is sent through Fora to{" "}
            {values?.address ?? "the configured service"} for transcription.
          </p>
          <Button type="submit" variant="primary" disabled={!dirty}>
            {busy ? <Spinner label="Saving voice settings" /> : null}
            Save
          </Button>
        </fieldset>
      </form>
      <div className="flex flex-col gap-3 border-t border-line pt-4">
        <p className="text-sm">
          Test saved OpenAI Realtime settings · {values?.address ?? ""}
        </p>
        <div className="flex items-center gap-2">
          {active ? (
            <>
              <Button
                type="button"
                disabled={state === "finishing"}
                onClick={() => recording.current?.stop()}
              >
                {state === "starting" || state === "finishing" ? (
                  <Spinner
                    label={
                      state === "finishing"
                        ? "Transcribing"
                        : "Starting microphone"
                    }
                  />
                ) : null}
                Stop
              </Button>
              <Button type="button" onClick={() => recording.current?.cancel()}>
                Cancel
              </Button>
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
                !values || busy || dirty || (!values.api_key_set && !key)
              }
              onClick={test}
            >
              {dirty ? "Save to enable test" : "Start microphone test"}
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
