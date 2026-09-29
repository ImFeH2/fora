import {
  createContext,
  createElement,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { toast } from "@/components/ui/index";
import { BackendError, backend } from "@/lib/backend";

export function isSettingsSaving(sections: Record<string, boolean>): boolean {
  return Object.values(sections).some(Boolean);
}

const SettingsSaveContext = createContext<{
  saving: boolean;
  report: (section: string, saving: boolean) => void;
} | null>(null);

export function SettingsSaveProvider({ children }: { children: ReactNode }) {
  const [sections, setSections] = useState<Record<string, boolean>>({});
  const report = useCallback((section: string, saving: boolean) => {
    setSections((current) =>
      current[section] === saving ? current : { ...current, [section]: saving },
    );
  }, []);
  const saving = isSettingsSaving(sections);
  const value = useMemo(() => ({ saving, report }), [saving, report]);
  return createElement(SettingsSaveContext.Provider, { value }, children);
}

export function useSettingsSaving() {
  return useContext(SettingsSaveContext)?.saving ?? false;
}

export function useReportSettingsSave(section: string, saving: boolean) {
  const report = useContext(SettingsSaveContext)?.report;
  useEffect(() => {
    report?.(section, saving);
    return () => report?.(section, false);
  }, [report, section, saving]);
}

export function useSaver(
  load: () => Promise<void>,
  field: RefObject<HTMLElement | null>,
) {
  const [saving, setSaving] = useState(false);

  const save = async (section: string, values: Record<string, unknown>) => {
    setSaving(true);
    try {
      await backend.updateSettings(section, values);
      await load();
      return true;
    } catch (error) {
      if (error instanceof BackendError && error.transport)
        backend.reportFailure(error);
      else
        toast({
          tone: "danger",
          title: "Could not save",
          description: error instanceof Error ? error.message : String(error),
        });
      return false;
    } finally {
      setSaving(false);
      requestAnimationFrame(() => {
        if (document.activeElement === document.body) field.current?.focus();
      });
    }
  };

  return { saving, save };
}

export function reportLoadFailure(
  id: string,
  failure: unknown,
  retry: () => void,
): void {
  if (failure instanceof BackendError && failure.transport) {
    backend.reportFailure(failure);
    return;
  }
  toast({
    id,
    tone: "danger",
    title: "Could not load",
    description: failure instanceof Error ? failure.message : String(failure),
    duration: null,
    action: { label: "Retry", onClick: retry },
  });
}
