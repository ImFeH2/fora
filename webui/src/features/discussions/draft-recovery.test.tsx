import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";
import { useDraft } from "@/features/discussions/draft";
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

function DraftProbe() {
  const { controller, view, error } = useDraft(1);
  return (
    <>
      <textarea
        aria-label="Message"
        disabled={!controller}
        readOnly
        value={view?.draft.body ?? ""}
      />
      {error ? <div role="alert">{error}</div> : null}
    </>
  );
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
  vi.stubGlobal("sessionStorage", dom.window.sessionStorage);
  vi.stubGlobal("crypto", { randomUUID: () => "tab-id" });
  vi.stubGlobal("HTMLElement", dom.window.HTMLElement);
  vi.stubGlobal("Node", dom.window.Node);
  vi.stubGlobal("Event", dom.window.Event);
  vi.stubGlobal("EventTarget", dom.window.EventTarget);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
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

it("re-enables the message input and restores the draft after connection recovery", async () => {
  const organization = {
    uuid: "test-organization",
    human_id: 1,
  } as Awaited<ReturnType<typeof backend.organization>>;
  const connectionError = new BackendError(
    "disconnected",
    "Connection lost. Reconnect to continue.",
    true,
  );
  vi.spyOn(backend, "organization")
    .mockRejectedValueOnce(connectionError)
    .mockResolvedValueOnce(organization);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });

  render(<DraftProbe />);
  const input = () =>
    screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;

  await waitFor(() => expect(input().disabled).toBe(true));
  expect(screen.queryByRole("alert")).toBeNull();

  act(() => {
    for (const listener of listeners) listener({ type: "connection.restored" });
  });

  await waitFor(() => expect(input().disabled).toBe(false));
  expect(input().value).toBe("Recovered message");
  expect(screen.queryByRole("alert")).toBeNull();
  expect(backend.organization).toHaveBeenCalledTimes(2);
});
