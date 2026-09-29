import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  Button,
  Chip,
  Field,
  Spinner,
  Textarea,
  toast,
} from "@/components/ui/index";
import {
  reportLoadFailure,
  useReportSettingsSave,
} from "@/features/settings/saver";
import { BackendError, backend } from "@/lib/backend";

export type ExecutionSettings = {
  write_directories: string[];
  unusable_write_directories: { path: string; reason: string }[];
  error: string | null;
};

export function directoryDraft(settings: ExecutionSettings): string {
  return settings.write_directories.join("\n");
}

export function executionChanged(
  saved: ExecutionSettings,
  draft: string,
): boolean {
  return draft !== directoryDraft(saved) || saved.error !== null;
}

export function executionUpdate(directories: string) {
  return {
    write_directories: [
      ...new Set(
        directories
          .split("\n")
          .map((path) => path.trim())
          .filter(Boolean),
      ),
    ],
  };
}

export function ExecutionForm({
  initial,
  onSave,
}: {
  initial: ExecutionSettings;
  onSave: (
    values: ReturnType<typeof executionUpdate>,
  ) => Promise<ExecutionSettings>;
}) {
  const id = useId();
  const input = useRef<HTMLTextAreaElement>(null);
  const [info, setInfo] = useState(initial);
  const [draft, setDraft] = useState(() => directoryDraft(initial));
  const [busy, setBusy] = useState(false);
  useReportSettingsSave("execution", busy);
  const previous = useRef(initial);
  useEffect(() => {
    const old = previous.current;
    previous.current = initial;
    setDraft((current) =>
      current === directoryDraft(old) ? directoryDraft(initial) : current,
    );
    setInfo(initial);
  }, [initial]);
  const changed = executionChanged(info, draft);
  const save = async () => {
    if (busy || !changed) return;
    setBusy(true);
    try {
      const result = await onSave(executionUpdate(draft));
      previous.current = result;
      setInfo(result);
      setDraft(directoryDraft(result));
    } catch (failure) {
      if (failure instanceof BackendError && failure.transport)
        backend.reportFailure(failure);
      else
        toast({
          tone: "danger",
          title: "Could not save",
          description:
            failure instanceof Error ? failure.message : String(failure),
        });
    } finally {
      setBusy(false);
      requestAnimationFrame(() => {
        if (document.activeElement === document.body) input.current?.focus();
      });
    }
  };
  return (
    <form
      className="m-0 flex min-w-0 max-w-[560px] flex-col gap-4 border-0 p-0"
      aria-label="Execution settings"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      {info.error || info.unusable_write_directories.length ? (
        <ul className="m-0 flex list-none flex-col gap-2 p-0">
          {info.error ? (
            <li className="flex min-w-0 items-center gap-2">
              <Chip tone="danger">Unavailable</Chip>
              <span className="flex-1 min-w-0 text-fg-muted text-sm wrap-anywhere">
                {info.error}
              </span>
            </li>
          ) : null}
          {info.unusable_write_directories.map((item) => (
            <li key={item.path} className="flex min-w-0 items-center gap-2">
              <Chip tone="warning">{item.reason}</Chip>
              <span className="flex-1 min-w-0 truncate font-mono text-sm text-fg-muted wrap-anywhere">
                {item.path}
              </span>
            </li>
          ))}
        </ul>
      ) : null}
      <fieldset
        className="m-0 flex min-w-0 max-w-[560px] flex-col gap-4 border-0 p-0"
        disabled={busy}
      >
        <Field label="Writable directories" htmlFor={id}>
          <Textarea
            ref={input}
            id={id}
            rows={6}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
          />
        </Field>
        <div className="flex items-center gap-3 pt-1">
          <Button variant="primary" type="submit" disabled={!changed || busy}>
            {busy ? <Spinner label="Saving execution settings" /> : null}
            Save
          </Button>
        </div>
      </fieldset>
    </form>
  );
}

export function ExecutionPanel() {
  const [info, setInfo] = useState<ExecutionSettings | null>(null);
  const load = useCallback(async () => {
    try {
      setInfo((await backend.settings("execution")) as ExecutionSettings);
    } catch (failure) {
      reportLoadFailure("settings-execution", failure, () => void load());
    }
  }, []);
  useEffect(() => {
    void load();
    return backend.onEvent((event) => {
      if (event.type === "connection.restored") void load();
    });
  }, [load]);
  if (!info) return null;
  return (
    <ExecutionForm
      initial={info}
      onSave={async (values) =>
        (await backend.updateSettings("execution", values)) as ExecutionSettings
      }
    />
  );
}
