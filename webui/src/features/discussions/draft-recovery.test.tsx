import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";
import { Composer } from "@/features/discussions/composer";
import { BackendError, type BackendEvent, backend } from "@/lib/backend";

const storage = vi.hoisted(() => ({
  get: vi.fn(),
  getAll: vi.fn(),
  close: vi.fn(),
}));
vi.mock("idb", () => ({
  openDB: vi.fn(async () => storage),
}));

let cleanup: typeof import("@testing-library/react").cleanup;
let render: typeof import("@testing-library/react").render;
let screen: typeof import("@testing-library/react").screen;
let waitFor: typeof import("@testing-library/react").waitFor;
let act: typeof import("@testing-library/react").act;
let closeDom: () => void;
let listeners: Set<(event: BackendEvent) => void>;

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
  vi.stubGlobal("sessionStorage", dom.window.sessionStorage);
  vi.stubGlobal("crypto", { randomUUID: () => "tab-id" });
  vi.stubGlobal("HTMLElement", dom.window.HTMLElement);
  vi.stubGlobal("Node", dom.window.Node);
  vi.stubGlobal("Event", dom.window.Event);
  vi.stubGlobal("EventTarget", dom.window.EventTarget);
  vi.stubGlobal("MutationObserver", dom.window.MutationObserver);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal(
    "requestAnimationFrame",
    vi.fn(() => 1),
  );
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal("getComputedStyle", () => ({
    borderBottomWidth: "0px",
    borderTopWidth: "0px",
    lineHeight: "20px",
    paddingBottom: "12px",
    paddingTop: "12px",
  }));
  Object.defineProperty(dom.window.document, "fonts", {
    configurable: true,
    value: { ready: Promise.resolve() },
  });
  Object.defineProperty(dom.window.navigator, "locks", {
    configurable: true,
    value: {
      request: async (
        _name: string,
        _options: unknown,
        callback: (lock: object) => Promise<unknown>,
      ) => callback({}),
    },
  });
  ({ cleanup, render, screen, waitFor, act } = await import(
    "@testing-library/react"
  ));
}, 30000);

beforeEach(() => {
  storage.get.mockReset().mockResolvedValue({
    key: "test-organization:1:1:tab-id",
    updatedAt: 0,
    body: "Recovered message",
    bodyRevision: 0,
    files: [],
    pending: null,
  });
  storage.getAll.mockReset().mockResolvedValue([]);
  storage.close.mockReset();
  listeners = new Set();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

afterAll(() => {
  closeDom();
  vi.unstubAllGlobals();
});

function organization() {
  return {
    uuid: "test-organization",
    human_id: 1,
  } as Awaited<ReturnType<typeof backend.organization>>;
}

function mountComposer(discussionId: number) {
  return render(
    <Composer
      discussionId={discussionId}
      members={[]}
      memberIds={new Set<number>()}
      busy={false}
      placeholder="Message"
      onSend={async () => true}
      onHeightChange={() => {}}
      onOpenVoiceSettings={() => {}}
    />,
  );
}

function input() {
  return screen.getByRole("combobox", {
    name: "Message",
  }) as HTMLTextAreaElement;
}

it("retries a failed draft initialization from the Composer", async () => {
  let failureObserved!: () => void;
  const failed = new Promise<void>((resolve) => {
    failureObserved = resolve;
  });
  vi.spyOn(backend, "organization")
    .mockImplementationOnce(async () => {
      failureObserved();
      throw new Error("Draft initialization failed");
    })
    .mockResolvedValueOnce(organization());
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });

  mountComposer(1);
  await failed;
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain(
      "Draft initialization failed",
    ),
  );
  expect(input().disabled).toBe(true);
  const retry = screen.getByRole("button", { name: "Retry" });

  act(() => retry.click());

  await waitFor(() => expect(input().disabled).toBe(false));
  expect(input().value).toBe("Recovered message");
  expect(screen.queryByRole("alert")).toBeNull();
  expect(backend.organization).toHaveBeenCalledTimes(2);
});

it("re-enables the Composer after connection recovery without duplicate feedback", async () => {
  let failureObserved!: () => void;
  const failed = new Promise<void>((resolve) => {
    failureObserved = resolve;
  });
  const connectionError = new BackendError(
    "disconnected",
    "Connection lost. Reconnect to continue.",
    true,
  );
  vi.spyOn(backend, "organization")
    .mockImplementationOnce(async () => {
      failureObserved();
      throw connectionError;
    })
    .mockResolvedValueOnce(organization());
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });

  mountComposer(2);
  await failed;
  await waitFor(() => expect(backend.organization).toHaveBeenCalledOnce());
  expect(input().disabled).toBe(true);
  expect(screen.queryByRole("alert")).toBeNull();

  act(() => {
    for (const listener of listeners) listener({ type: "connection.restored" });
  });

  await waitFor(() => expect(input().disabled).toBe(false));
  expect(input().value).toBe("Recovered message");
  expect(screen.queryByRole("alert")).toBeNull();
  expect(backend.organization).toHaveBeenCalledTimes(2);
});
