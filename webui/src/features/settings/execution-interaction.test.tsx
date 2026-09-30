import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";
import type { ExecutionSettings } from "@/features/settings/execution";
import type { BackendEvent } from "@/lib/backend";

let act: typeof import("@testing-library/react").act;
let cleanup: typeof import("@testing-library/react").cleanup;
let fireEvent: typeof import("@testing-library/react").fireEvent;
let render: typeof import("@testing-library/react").render;
let screen: typeof import("@testing-library/react").screen;
let waitFor: typeof import("@testing-library/react").waitFor;
let ExecutionForm: typeof import("@/features/settings/execution").ExecutionForm;
let ExecutionPanel: typeof import("@/features/settings/execution").ExecutionPanel;
let SettingsSaveProvider: typeof import("@/features/settings/saver").SettingsSaveProvider;
let useSettingsSaving: typeof import("@/features/settings/saver").useSettingsSaving;
let backend: typeof import("@/lib/backend").backend;
let BackendError: typeof import("@/lib/backend").BackendError;
let clearToasts: typeof import("@/components/ui/toast").clearToasts;
let readToasts: typeof import("@/components/ui/toast").readToasts;
let closeDom: () => void;
let event: (value: BackendEvent) => void;
const off = vi.fn();

const saved: ExecutionSettings = {
  write_directories: ["/work", "/other"],
  unusable_write_directories: [],
  error: null,
};
const updated: ExecutionSettings = {
  ...saved,
  write_directories: ["/remote"],
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

beforeAll(async () => {
  const packageName = "jsdom";
  const { JSDOM } = await import(packageName);
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://localhost",
  });
  closeDom = () => dom.window.close();
  vi.stubGlobal("window", dom.window);
  vi.stubGlobal("document", dom.window.document);
  vi.stubGlobal("navigator", dom.window.navigator);
  vi.stubGlobal("HTMLElement", dom.window.HTMLElement);
  vi.stubGlobal("Node", dom.window.Node);
  vi.stubGlobal("MutationObserver", dom.window.MutationObserver);
  vi.stubGlobal("Event", dom.window.Event);
  vi.stubGlobal("EventTarget", dom.window.EventTarget);
  vi.stubGlobal("CustomEvent", dom.window.CustomEvent);
  vi.stubGlobal("MouseEvent", dom.window.MouseEvent);
  vi.stubGlobal("KeyboardEvent", dom.window.KeyboardEvent);
  vi.stubGlobal("AbortController", dom.window.AbortController);
  vi.stubGlobal("AbortSignal", dom.window.AbortSignal);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("requestAnimationFrame", vi.fn());
  ({ act, cleanup, fireEvent, render, screen, waitFor } = await import(
    "@testing-library/react"
  ));
  ({ ExecutionForm, ExecutionPanel } = await import(
    "@/features/settings/execution"
  ));
  ({ SettingsSaveProvider, useSettingsSaving } = await import(
    "@/features/settings/saver"
  ));
  ({ backend, BackendError } = await import("@/lib/backend"));
  ({ clearToasts, readToasts } = await import("@/components/ui/toast"));
}, 30000);

beforeEach(() => {
  off.mockClear();
  clearToasts();
  vi.spyOn(backend, "settings").mockResolvedValue(saved);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    event = listener;
    return off;
  });
  vi.spyOn(backend, "updateSettings").mockImplementation(async (_, values) => ({
    ...saved,
    write_directories: values.write_directories,
  }));
});

afterEach(() => {
  cleanup();
  clearToasts();
  vi.restoreAllMocks();
});

afterAll(() => {
  closeDom();
  vi.unstubAllGlobals();
});

function button(name: string) {
  return screen.getByRole("button", { name }) as HTMLButtonElement;
}

function input(name = "Edit directory path") {
  return screen.getByRole("textbox", { name }) as HTMLInputElement;
}

function change(name: string, value: string) {
  fireEvent.change(input(name), { target: { value } });
}

function edit(path = "/work", value = "/edited") {
  fireEvent.click(button(`Edit ${path}`));
  change("Edit directory path", value);
}

function add(value = "/new") {
  fireEvent.click(button("Add directory"));
  change("New directory path", value);
}

async function openPanel() {
  const view = render(<ExecutionPanel />);
  await waitFor(() => expect(button("Edit /work")).toBeTruthy());
  return view;
}

async function save(expected: string[]) {
  fireEvent.click(button("Save"));
  await waitFor(() => expect(button("Save").disabled).toBe(true));
  expect(backend.updateSettings).toHaveBeenLastCalledWith("execution", {
    write_directories: expected,
  });
  await waitFor(() =>
    expect(
      screen.queryByRole("status", { name: "Saving execution settings" }),
    ).toBeNull(),
  );
}

it("adds, completes, edits, cancels and deletes with valid focus", async () => {
  await openPanel();
  expect(button("Save").disabled).toBe(true);
  add(" /new ");
  expect(document.activeElement).toBe(input("New directory path"));
  fireEvent.keyDown(input("New directory path"), { key: "Enter" });
  expect(document.activeElement).toBe(button("Add directory"));
  expect(button("Edit /new")).toBeTruthy();
  edit("/new", "/cancelled");
  fireEvent.keyDown(input(), { key: "Escape" });
  expect(document.activeElement).toBe(button("Edit /new"));
  edit("/new", "/done");
  fireEvent.click(button("Done"));
  expect(document.activeElement).toBe(button("Edit /done"));
  expect(backend.updateSettings).not.toHaveBeenCalled();
  fireEvent.click(button("Delete /done"));
  expect(document.activeElement).toBe(button("Edit /other"));
  fireEvent.click(button("Delete /other"));
  fireEvent.click(button("Delete /work"));
  expect(document.activeElement).toBe(button("Add directory"));
  await save([]);
});

it.each(["edit", "add", "both"])(
  "saves pending %s inputs directly",
  async (mode) => {
    await openPanel();
    if (mode !== "add") edit();
    if (mode !== "edit") add();
    expect(button("Save").disabled).toBe(false);
    await save([
      mode === "add" ? "/work" : "/edited",
      "/other",
      ...(mode === "edit" ? [] : ["/new"]),
    ]);
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(document.activeElement).toBe(button("Add directory"));
  },
);

it("preserves current input when switching rows, adding and deleting earlier rows", async () => {
  await openPanel();
  edit("/other", "/pending-other");
  add("/pending-add");
  expect(input().value).toBe("/pending-other");
  fireEvent.click(button("Delete /work"));
  expect(input().value).toBe("/pending-other");
  await save(["/pending-other", "/pending-add"]);
  edit("/pending-other", "/first-draft");
  fireEvent.click(button("Edit /pending-add"));
  expect(button("Edit /first-draft")).toBeTruthy();
  change("Edit directory path", "/second-draft");
  await save(["/first-draft", "/second-draft"]);
});

it("keeps editing text while completing an addition", async () => {
  await openPanel();
  edit("/other", "/work");
  add("/third");
  fireEvent.keyDown(input("New directory path"), { key: "Enter" });
  expect(input().value).toBe("/work");
  await save(["/work", "/third"]);
});

it("filters empty paths and duplicates and focuses the surviving item", async () => {
  await openPanel();
  edit("/other", " /work ");
  fireEvent.keyDown(input(), { key: "Enter" });
  expect(document.activeElement).toBe(button("Edit /work"));
  expect(screen.getAllByRole("listitem")).toHaveLength(1);
  add(" /work ");
  await save(["/work"]);
  edit("/work", "   ");
  fireEvent.click(button("Done"));
  expect(document.activeElement).toBe(button("Add directory"));
  await save([]);
});

it("cancels new input and leaves composing Enter and Escape in the input", async () => {
  await openPanel();
  add("/temporary");
  fireEvent.keyDown(input("New directory path"), { key: "Escape" });
  expect(button("Save").disabled).toBe(true);
  expect(document.activeElement).toBe(button("Add directory"));
  edit();
  fireEvent.keyDown(input(), { key: "Enter", isComposing: true });
  fireEvent.keyDown(input(), { key: "Escape", isComposing: true });
  fireEvent.keyDown(input(), { key: "Enter", keyCode: 229 });
  expect(input().value).toBe("/edited");
  expect(backend.updateSettings).not.toHaveBeenCalled();
  fireEvent.click(button("Cancel"));
  expect(button("Save").disabled).toBe(true);
});

it("preserves all inputs after a save failure and supports correction", async () => {
  await openPanel();
  vi.mocked(backend.updateSettings).mockRejectedValueOnce(
    new Error("Absolute paths required"),
  );
  edit("/work", "relative/path");
  add(" /new ");
  fireEvent.click(button("Save"));
  await waitFor(() => expect(readToasts()).toHaveLength(1));
  expect(input().value).toBe("relative/path");
  expect(input("New directory path").value).toBe(" /new ");
  expect(button("Save").disabled).toBe(false);
  expect(readToasts()[0].description).toBe("Absolute paths required");
  change("Edit directory path", "/corrected");
  await save(["/corrected", "/other", "/new"]);
});

it("reports a transport failure once and keeps the editable draft", async () => {
  await openPanel();
  const failure = new BackendError("offline", "Disconnected", true);
  const report = vi
    .spyOn(backend, "reportFailure")
    .mockImplementation(() => {});
  vi.mocked(backend.updateSettings).mockRejectedValueOnce(failure);
  edit();
  fireEvent.click(button("Save"));
  await waitFor(() => expect(report).toHaveBeenCalledExactlyOnceWith(failure));
  expect(readToasts()).toHaveLength(0);
  expect(input().value).toBe("/edited");
});

function SavingNavigation() {
  return (
    <button type="button" disabled={useSettingsSaving()}>
      Other section
    </button>
  );
}

it("disables the form and Settings navigation and prevents repeat submission", async () => {
  const pending = deferred<Record<string, unknown>>();
  vi.mocked(backend.updateSettings).mockReturnValueOnce(pending.promise);
  render(
    <SettingsSaveProvider>
      <SavingNavigation />
      <ExecutionPanel />
    </SettingsSaveProvider>,
  );
  await waitFor(() => expect(button("Edit /work")).toBeTruthy());
  edit();
  add();
  fireEvent.click(button("Save"));
  expect(button("Save").disabled).toBe(true);
  expect(button("Other section").disabled).toBe(true);
  expect(input().matches(":disabled")).toBe(true);
  expect(
    screen
      .getByRole("status", { name: "Saving execution settings" })
      .closest("button"),
  ).toBe(button("Save"));
  fireEvent.submit(screen.getByRole("form", { name: "Execution settings" }));
  fireEvent.change(input(), { target: { value: "/during-save" } });
  fireEvent.click(button("Delete /other"));
  expect(input().value).toBe("/edited");
  expect(backend.updateSettings).toHaveBeenCalledOnce();
  await act(async () =>
    pending.resolve({
      ...saved,
      write_directories: ["/edited", "/other", "/new"],
    }),
  );
  expect(button("Other section").disabled).toBe(false);
});

it("allows retrying unchanged directories when the environment is unavailable", async () => {
  vi.mocked(backend.settings).mockResolvedValueOnce({
    ...saved,
    error: "Unavailable execution",
  });
  await openPanel();
  expect(screen.getByText("Unavailable execution")).toBeTruthy();
  expect(button("Save").disabled).toBe(false);
  await save(saved.write_directories);
  expect(screen.queryByText("Unavailable execution")).toBeNull();
});

it("uses Backend normalization and associates diagnostics with the current path", async () => {
  vi.mocked(backend.settings).mockResolvedValueOnce({
    ...saved,
    unusable_write_directories: [
      { path: "/work", reason: "invalid_directory" },
    ],
  });
  await openPanel();
  expect(
    screen.getByText("invalid_directory").closest("li")?.textContent,
  ).toContain("/work");
  edit("/work", "/work/../normalized");
  expect(screen.queryByText("invalid_directory")).toBeNull();
  vi.mocked(backend.updateSettings).mockResolvedValueOnce({
    ...saved,
    write_directories: ["/normalized", "/other"],
  });
  await save(["/work/../normalized", "/other"]);
  expect(button("Edit /normalized")).toBeTruthy();
});

it.each(["clean", "done", "edit", "add"])(
  "refreshes with a %s draft after reconnect",
  async (mode) => {
    await openPanel();
    if (mode === "done" || mode === "edit") edit();
    if (mode === "done") fireEvent.click(button("Done"));
    if (mode === "add") add();
    vi.mocked(backend.settings).mockResolvedValueOnce(updated);
    await act(async () => event({ type: "connection.restored" }));
    expect(backend.settings).toHaveBeenCalledTimes(2);
    if (mode === "clean") expect(button("Edit /remote")).toBeTruthy();
    if (mode === "done") expect(button("Edit /edited")).toBeTruthy();
    if (mode === "edit") expect(input().value).toBe("/edited");
    if (mode === "add") expect(input("New directory path").value).toBe("/new");
  },
);

it("keeps inputs entered while a reconnect read is pending", async () => {
  await openPanel();
  const reading = deferred<Record<string, unknown>>();
  vi.mocked(backend.settings).mockReturnValueOnce(reading.promise);
  await act(async () => event({ type: "connection.restored" }));
  edit();
  add();
  await act(async () => reading.resolve(updated));
  expect(input().value).toBe("/edited");
  expect(input("New directory path").value).toBe("/new");
});

it.each(["success", "failure"])(
  "ignores old read %s after a successful save",
  async (outcome) => {
    await openPanel();
    const reading = deferred<Record<string, unknown>>();
    vi.mocked(backend.settings).mockReturnValueOnce(reading.promise);
    await act(async () => event({ type: "connection.restored" }));
    edit();
    await save(["/edited", "/other"]);
    await act(async () => {
      if (outcome === "success") reading.resolve(updated);
      else reading.reject(new Error("outdated read"));
    });
    expect(button("Edit /edited")).toBeTruthy();
    expect(readToasts()).toHaveLength(0);
  },
);

it("ignores reconnect reads during a save and refreshes on a later reconnect", async () => {
  await openPanel();
  const pending = deferred<Record<string, unknown>>();
  vi.mocked(backend.updateSettings).mockReturnValueOnce(pending.promise);
  edit();
  fireEvent.click(button("Save"));
  await act(async () => event({ type: "connection.restored" }));
  expect(backend.settings).toHaveBeenCalledOnce();
  await act(async () =>
    pending.resolve({ ...saved, write_directories: ["/edited", "/other"] }),
  );
  vi.mocked(backend.settings).mockResolvedValueOnce(updated);
  await act(async () => event({ type: "connection.restored" }));
  expect(button("Edit /remote")).toBeTruthy();
});

it("preserves a failed-save draft across reconnect and retries", async () => {
  await openPanel();
  vi.mocked(backend.updateSettings).mockRejectedValueOnce(new Error("offline"));
  edit();
  add();
  fireEvent.click(button("Save"));
  await waitFor(() => expect(readToasts()).toHaveLength(1));
  await act(async () => event({ type: "connection.restored" }));
  expect(input().value).toBe("/edited");
  expect(input("New directory path").value).toBe("/new");
  await save(["/edited", "/other", "/new"]);
});

it("takes the latest read and releases the listener on unmount", async () => {
  const view = await openPanel();
  const reading = deferred<Record<string, unknown>>();
  vi.mocked(backend.settings).mockReturnValueOnce(reading.promise);
  await act(async () => event({ type: "connection.restored" }));
  vi.mocked(backend.settings).mockResolvedValueOnce(updated);
  await act(async () => event({ type: "connection.restored" }));
  await act(async () => reading.resolve(saved));
  expect(button("Edit /remote")).toBeTruthy();
  view.unmount();
  expect(off).toHaveBeenCalledOnce();
});

it("ignores a late read after unmount and offers Retry after a current read failure", async () => {
  const reading = deferred<Record<string, unknown>>();
  vi.mocked(backend.settings).mockReturnValueOnce(reading.promise);
  const view = render(<ExecutionPanel />);
  view.unmount();
  await act(async () => reading.reject(new Error("late failure")));
  expect(readToasts()).toHaveLength(0);
  vi.mocked(backend.settings).mockRejectedValueOnce(new Error("read failed"));
  render(<ExecutionPanel />);
  await waitFor(() => expect(readToasts()).toHaveLength(1));
  await act(async () => readToasts()[0].action?.onClick());
  expect(button("Edit /work")).toBeTruthy();
});

it("protects whitespace input and duplicate additions from external initial updates", () => {
  const onSave = vi.fn(async () => saved);
  const view = render(<ExecutionForm initial={saved} onSave={onSave} />);
  edit("/work", " /work ");
  add("/other");
  view.rerender(<ExecutionForm initial={updated} onSave={onSave} />);
  expect(input().value).toBe(" /work ");
  expect(input("New directory path").value).toBe("/other");
});
