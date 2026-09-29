import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";
import type { BackendEvent } from "@/lib/backend";

let act: typeof import("@testing-library/react").act;
let cleanup: typeof import("@testing-library/react").cleanup;
let fireEvent: typeof import("@testing-library/react").fireEvent;
let render: typeof import("@testing-library/react").render;
let screen: typeof import("@testing-library/react").screen;
let waitFor: typeof import("@testing-library/react").waitFor;
let VoicePanel: typeof import("@/features/settings/voice").VoicePanel;
let backend: typeof import("@/lib/backend").backend;
let closeDom: () => void;
let event: (value: BackendEvent) => void;

type Values = {
  address: string;
  model: string;
  api_key_set: boolean;
};

const saved: Values = {
  address: "wss://saved.example/voice",
  model: "saved-model",
  api_key_set: true,
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
  vi.stubGlobal(
    "getComputedStyle",
    dom.window.getComputedStyle.bind(dom.window),
  );
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("requestAnimationFrame", vi.fn());
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  ({ act, cleanup, fireEvent, render, screen, waitFor } = await import(
    "@testing-library/react"
  ));
  ({ VoicePanel } = await import("@/features/settings/voice"));
  ({ backend } = await import("@/lib/backend"));
}, 30000);

beforeEach(() => {
  vi.spyOn(backend, "settings").mockResolvedValue(saved);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    event = listener;
    return vi.fn();
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

afterAll(() => {
  closeDom();
  vi.unstubAllGlobals();
});

async function openPanel() {
  const view = render(<VoicePanel />);
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Service address") as HTMLInputElement).value,
    ).toBe(saved.address),
  );
  return view;
}

it("offers Retry after a voice settings read failure", async () => {
  vi.mocked(backend.settings).mockRejectedValueOnce(new Error("offline"));
  render(<VoicePanel />);
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain("offline"),
  );
  vi.mocked(backend.settings).mockResolvedValueOnce(saved);
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() =>
    expect(
      (screen.getByLabelText("Service address") as HTMLInputElement).value,
    ).toBe(saved.address),
  );
  expect(backend.settings).toHaveBeenCalledTimes(2);
});

it("does not start a reconnect read over unsaved voice edits", async () => {
  await openPanel();
  const address = screen.getByLabelText("Service address") as HTMLInputElement;
  fireEvent.change(address, {
    target: { value: "wss://edited.example/voice" },
  });

  event({ type: "connection.restored" });
  await Promise.resolve();

  expect(backend.settings).toHaveBeenCalledOnce();
  expect(address.value).toBe("wss://edited.example/voice");
});

it("keeps edits made during a reconnect read", async () => {
  await openPanel();
  const address = screen.getByLabelText("Service address") as HTMLInputElement;
  const remote = deferred<Values>();
  vi.mocked(backend.settings).mockReturnValueOnce(remote.promise);

  await act(async () => {
    event({ type: "connection.restored" });
    await waitFor(() => expect(backend.settings).toHaveBeenCalledTimes(2));
  });
  fireEvent.change(address, {
    target: { value: "wss://edited.example/voice" },
  });
  remote.resolve({
    address: "wss://reconnected.example/voice",
    model: "reconnected-model",
    api_key_set: true,
  });
  await waitFor(() => expect(address.value).toBe("wss://edited.example/voice"));
});
