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
  put: vi.fn(),
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
  storage.put.mockReset().mockResolvedValue(undefined);
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

function mountComposer(
  discussionId: number,
  onSend: () => Promise<boolean> = async () => true,
) {
  return render(
    <Composer
      discussionId={discussionId}
      members={[]}
      memberIds={new Set<number>()}
      busy={false}
      placeholder="Message"
      onSend={onSend}
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

it("completes upload cancellation after connection recovery", async () => {
  const file = new File(["content"], "recover.txt");
  const upload = {
    id: "recover-upload",
    state: "receiving" as const,
    expires_at: Date.now() / 1000 + 60,
  };
  storage.get.mockResolvedValue({
    key: "test-organization:1:3:tab-id",
    updatedAt: 0,
    body: "Edited message",
    bodyRevision: 1,
    files: [
      {
        id: "file",
        clientId: "client",
        file,
        upload,
      },
    ],
    pending: {
      id: "cancel-attempt",
      body: "Edited message",
      bodyRevision: 1,
      files: [
        {
          id: "file",
          clientId: "client",
          file,
          upload,
        },
      ],
      phase: "uploading",
      cancelRequested: true,
    },
  });
  const connectionError = new BackendError(
    "unconfirmed",
    "Upload cancellation result unknown",
    true,
    "upload.cancel",
  );
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "disconnected", "get").mockReturnValue(false);
  vi.spyOn(backend, "reconnecting", "get").mockReturnValue(false);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "cancelUploads")
    .mockRejectedValueOnce(connectionError)
    .mockResolvedValue({ cancelled: 1 });
  const send = vi.fn().mockResolvedValue(true);

  mountComposer(3, send);
  await waitFor(() => expect(backend.cancelUploads).toHaveBeenCalledOnce());
  expect(screen.getByText("Cancelling upload")).toBeTruthy();
  expect(screen.getByText(/Could not finish cancelling send/)).toBeTruthy();
  expect(screen.queryByRole("button", { name: "Retry send" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Cancel upload" })).toBeNull();

  act(() => {
    for (const listener of listeners)
      listener({ type: "connection.closed", error: connectionError });
  });

  await waitFor(() =>
    expect(screen.queryByText(/Could not finish cancelling send/)).toBeNull(),
  );
  expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();

  act(() => {
    for (const listener of listeners) listener({ type: "connection.restored" });
  });

  await waitFor(() =>
    expect(screen.queryByText("Cancelling upload")).toBeNull(),
  );
  expect(backend.cancelUploads).toHaveBeenCalledTimes(2);
  expect(send).not.toHaveBeenCalled();
  expect(input().value).toBe("Edited message");
  expect(screen.getByText("recover.txt")).toBeTruthy();
});

it("keeps connection feedback global when cancellation fails after closure", async () => {
  const file = new File(["content"], "closed-cancel.txt");
  const upload = {
    id: "closed-upload",
    state: "receiving" as const,
    expires_at: Date.now() / 1000 + 60,
  };
  storage.get.mockResolvedValue({
    key: "test-organization:1:5:tab-id",
    updatedAt: 0,
    body: "Edited during cancellation",
    bodyRevision: 1,
    files: [{ id: "file", clientId: "client", file, upload }],
    pending: {
      id: "closed-attempt",
      body: "Original message",
      bodyRevision: 0,
      files: [{ id: "file", clientId: "client", file, upload }],
      phase: "uploading",
      cancelRequested: true,
    },
  });
  let rejectCancellation!: (error: BackendError) => void;
  const cancellation = new Promise<
    Awaited<ReturnType<typeof backend.cancelUploads>>
  >((_resolve, reject) => {
    rejectCancellation = reject;
  });
  const error = new BackendError(
    "unconfirmed",
    "Upload cancellation result unknown",
    true,
    "upload.cancel",
  );
  let disconnected = false;
  vi.spyOn(backend, "disconnected", "get").mockImplementation(
    () => disconnected,
  );
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "cancelUploads")
    .mockReturnValueOnce(cancellation)
    .mockResolvedValue({ cancelled: 1 });
  const send = vi.fn().mockResolvedValue(true);

  mountComposer(5, send);
  await waitFor(() => expect(backend.cancelUploads).toHaveBeenCalledOnce());
  await act(async () => {
    disconnected = true;
    for (const listener of listeners)
      listener({ type: "connection.closed", error });
    rejectCancellation(error);
    await cancellation.catch(() => {});
  });

  expect(input().value).toBe("Edited during cancellation");
  expect(screen.getByText("Cancelling upload")).toBeTruthy();
  expect(screen.queryByText(/Could not finish cancelling send/)).toBeNull();
  expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Retry send" })).toBeNull();
  expect(
    screen.getByRole("list", { name: "Draft attachments" }).textContent,
  ).toContain("closed-cancel.txt");

  act(() => {
    disconnected = false;
    for (const listener of listeners) listener({ type: "connection.restored" });
  });
  await waitFor(() =>
    expect(screen.queryByText("Cancelling upload")).toBeNull(),
  );
  expect(backend.cancelUploads).toHaveBeenCalledTimes(2);
  expect(input().value).toBe("Edited during cancellation");
  expect(
    screen.getByRole("list", { name: "Draft attachments" }).textContent,
  ).toContain("closed-cancel.txt");
  expect(send).not.toHaveBeenCalled();
});

it("offers a retry for an upload cleanup timeout on an open connection", async () => {
  const file = new File(["content"], "retry-cancel.txt");
  const upload = {
    id: "retry-upload",
    state: "receiving" as const,
    expires_at: Date.now() / 1000 + 60,
  };
  storage.get.mockResolvedValue({
    key: "test-organization:1:4:tab-id",
    updatedAt: 0,
    body: "Edited message",
    bodyRevision: 1,
    files: [
      {
        id: "file",
        clientId: "client",
        file,
        upload,
      },
    ],
    pending: {
      id: "retry-attempt",
      body: "Edited message",
      bodyRevision: 1,
      files: [
        {
          id: "file",
          clientId: "client",
          file,
          upload,
        },
      ],
      phase: "uploading",
      cancelRequested: true,
    },
  });
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "cancelUploads")
    .mockRejectedValueOnce(
      new BackendError(
        "unconfirmed",
        "Upload cancellation result unknown",
        true,
        "upload.cancel",
      ),
    )
    .mockResolvedValue({ cancelled: 1 });
  const send = vi.fn().mockResolvedValue(true);

  mountComposer(4, send);
  await waitFor(() => expect(backend.cancelUploads).toHaveBeenCalledOnce());
  expect(screen.getByText("Cancelling upload")).toBeTruthy();
  const retry = screen.getByRole("button", { name: "Retry" });

  act(() => retry.click());

  await waitFor(() =>
    expect(screen.queryByText("Cancelling upload")).toBeNull(),
  );
  expect(backend.cancelUploads).toHaveBeenCalledTimes(2);
  expect(send).not.toHaveBeenCalled();
  expect(input().value).toBe("Edited message");
  expect(screen.queryByRole("button", { name: "Retry send" })).toBeNull();
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
