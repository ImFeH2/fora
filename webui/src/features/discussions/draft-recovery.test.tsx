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
let fireEvent: typeof import("@testing-library/react").fireEvent;
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
  ({ cleanup, render, screen, waitFor, act, fireEvent } = await import(
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

afterEach(async () => {
  await act(async () => cleanup());
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

async function mountComposer(
  discussionId: number,
  onSend: () => Promise<boolean> = async () => true,
) {
  await act(async () => {
    render(
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
  });
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

  await mountComposer(1);
  await failed;
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain(
      "Draft initialization failed",
    ),
  );
  const message = input();
  expect(message.disabled).toBe(true);
  const retry = screen.getByRole("button", { name: "Retry" });

  await act(async () => retry.click());

  expect(message.isConnected).toBe(true);
  await waitFor(() => expect(message.disabled).toBe(false));
  expect(message.value).toBe("Recovered message");
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

  await mountComposer(3, send);
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

it("clears a timed-out upload cancellation when the connection closes", async () => {
  const file = new File(["content"], "cancel-timeout.txt");
  storage.get.mockResolvedValue({
    key: "test-organization:1:8:tab-id",
    updatedAt: 0,
    body: "Edited message",
    bodyRevision: 1,
    files: [{ id: "file", clientId: "client", file }],
    pending: null,
  });
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "disconnected", "get").mockReturnValue(false);
  vi.spyOn(backend, "reconnecting", "get").mockReturnValue(false);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "createUpload").mockResolvedValue({
    id: "cancel-timeout-upload",
    state: "reserved",
    expires_at: Date.now() / 1000 + 3600,
  });
  let stopUpload!: () => void;
  vi.spyOn(backend, "uploadFile").mockImplementation(
    (_id, _file, signal) =>
      new Promise((_resolve, reject) => {
        stopUpload = () => reject(signal.reason);
      }),
  );
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

  await mountComposer(8, send);
  await waitFor(() => expect(input().value).toBe("Edited message"));
  act(() => screen.getByRole("button", { name: "Send · Enter" }).click());
  await waitFor(() => expect(stopUpload).toBeTypeOf("function"));
  await act(async () => {
    screen.getByRole("button", { name: "Cancel upload" }).click();
    stopUpload();
  });
  await waitFor(() =>
    expect(screen.getByText(/Could not finish cancelling send/)).toBeTruthy(),
  );
  expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();

  act(() => {
    for (const listener of listeners)
      listener({
        type: "connection.closed",
        error: new BackendError("disconnected", "Connection lost", true),
      });
  });

  expect(screen.queryByText(/Could not finish cancelling send/)).toBeNull();
  expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
  expect(input().value).toBe("Edited message");
  expect(
    screen.getByRole("list", { name: "Draft attachments" }).textContent,
  ).toContain("cancel-timeout.txt");

  act(() => {
    for (const listener of listeners) listener({ type: "connection.restored" });
  });
  await waitFor(() =>
    expect(screen.queryByText("Cancelling upload")).toBeNull(),
  );
  expect(backend.cancelUploads).toHaveBeenCalledTimes(2);
  expect(input().value).toBe("Edited message");
  expect(
    screen.getByRole("list", { name: "Draft attachments" }).textContent,
  ).toContain("cancel-timeout.txt");
  expect(send).not.toHaveBeenCalled();
});

it("shows one local save error when cancelling an in-progress upload", async () => {
  const file = new File(["content"], "local-upload.txt");
  storage.get.mockResolvedValue({
    key: "test-organization:1:9:tab-id",
    updatedAt: 0,
    body: "Edited during upload",
    bodyRevision: 1,
    files: [{ id: "file", clientId: "client", file }],
    pending: null,
  });
  let failSave = false;
  storage.put.mockImplementation(
    async (_store: string, draft: { pending: unknown }) => {
      if (failSave && !draft.pending) {
        failSave = false;
        throw new Error("Acceptance local save unavailable");
      }
    },
  );
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "createUpload").mockResolvedValue({
    id: "local-upload-id",
    state: "reserved",
    expires_at: Date.now() / 1000 + 3600,
  });
  let stopUpload!: () => void;
  vi.spyOn(backend, "uploadFile").mockImplementation(
    (_id, _file, signal) =>
      new Promise((_resolve, reject) => {
        stopUpload = () => reject(signal.reason);
      }),
  );
  vi.spyOn(backend, "cancelUploads").mockResolvedValue({ cancelled: 1 });
  const send = vi.fn().mockResolvedValue(true);

  await mountComposer(9, send);
  await waitFor(() => expect(input().value).toBe("Edited during upload"));
  act(() => screen.getByRole("button", { name: "Send · Enter" }).click());
  await waitFor(() => expect(stopUpload).toBeTypeOf("function"));
  fireEvent.change(input(), {
    target: { value: "Edited during upload again" },
  });
  await waitFor(() => expect(input().value).toBe("Edited during upload again"));
  failSave = true;
  await act(async () => {
    screen.getByRole("button", { name: "Cancel upload" }).click();
    stopUpload();
  });
  expect(failSave).toBe(false);
  expect(screen.getByText("Cancelling upload")).toBeTruthy();

  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toBe(
      "Could not finish cancelling send: Draft not saved: Acceptance local save unavailable",
    ),
  );
  expect(screen.getByText("Cancelling upload")).toBeTruthy();
  expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();

  act(() => screen.getByRole("button", { name: "Retry" }).click());
  await waitFor(() =>
    expect(screen.queryByText("Cancelling upload")).toBeNull(),
  );
  expect(input().value).toBe("Edited during upload again");
  expect(
    screen.getByRole("list", { name: "Draft attachments" }).textContent,
  ).toContain("local-upload.txt");
  expect(send).not.toHaveBeenCalled();
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

  await mountComposer(5, send);
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

it("keeps a local cancellation save failure after connection closure", async () => {
  const file = new File(["content"], "local-cancel.txt");
  const upload = {
    id: "local-upload",
    state: "receiving" as const,
    expires_at: Date.now() / 1000 + 60,
  };
  storage.get.mockResolvedValue({
    key: "test-organization:1:6:tab-id",
    updatedAt: 0,
    body: "Local draft",
    bodyRevision: 1,
    files: [{ id: "file", clientId: "client", file, upload }],
    pending: {
      id: "local-attempt",
      body: "Local draft",
      bodyRevision: 1,
      files: [{ id: "file", clientId: "client", file, upload }],
      phase: "uploading",
      cancelRequested: true,
    },
  });
  storage.put
    .mockResolvedValueOnce(undefined)
    .mockRejectedValueOnce(new Error("Local storage unavailable"));
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "cancelUploads").mockResolvedValue({ cancelled: 1 });
  const send = vi.fn().mockResolvedValue(true);

  await mountComposer(6, send);
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain(
      "Local storage unavailable",
    ),
  );
  expect(screen.getByRole("alert").textContent).toBe(
    "Could not finish cancelling send: Draft not saved: Local storage unavailable",
  );
  expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();

  act(() => {
    for (const listener of listeners)
      listener({
        type: "connection.closed",
        error: new BackendError("disconnected", "Connection lost", true),
      });
  });

  expect(screen.getByRole("alert").textContent).toBe(
    "Could not finish cancelling send: Draft not saved: Local storage unavailable",
  );
  expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  expect(input().value).toBe("Local draft");
  expect(send).not.toHaveBeenCalled();
});

it("keeps a file validation error after replacing cancellation feedback", async () => {
  const file = new File(["content"], "replace-cancel.txt");
  const upload = {
    id: "replace-upload",
    state: "receiving" as const,
    expires_at: Date.now() / 1000 + 60,
  };
  storage.get.mockResolvedValue({
    key: "test-organization:1:7:tab-id",
    updatedAt: 0,
    body: "Edited message",
    bodyRevision: 1,
    files: [{ id: "file", clientId: "client", file, upload }],
    pending: {
      id: "replace-attempt",
      body: "Original message",
      bodyRevision: 0,
      files: [{ id: "file", clientId: "client", file, upload }],
      phase: "uploading",
      cancelRequested: true,
    },
  });
  vi.spyOn(backend, "organization").mockResolvedValue(organization());
  vi.spyOn(backend, "disconnected", "get").mockReturnValue(false);
  vi.spyOn(backend, "reconnecting", "get").mockReturnValue(false);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  });
  vi.spyOn(backend, "cancelUploads").mockRejectedValue(
    new BackendError(
      "unconfirmed",
      "Upload cancellation result unknown",
      true,
      "upload.cancel",
    ),
  );

  await mountComposer(7, vi.fn());
  await waitFor(() =>
    expect(screen.getByRole("alert").textContent).toContain(
      "Upload cancellation result unknown",
    ),
  );
  const fileInput = screen.getByLabelText(
    "Choose attachments",
  ) as HTMLInputElement;
  Object.defineProperty(fileInput, "files", {
    configurable: true,
    value: [new File(["content"], "bad/name.txt")],
  });
  await act(async () => {
    fileInput.dispatchEvent(new Event("change", { bubbles: true }));
  });
  expect(screen.getByRole("alert").textContent).toContain("Choose a filename");

  act(() => {
    for (const listener of listeners)
      listener({
        type: "connection.closed",
        error: new BackendError("disconnected", "Connection lost", true),
      });
  });

  expect(screen.getByRole("alert").textContent).toContain("Choose a filename");
  expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  expect(input().value).toBe("Edited message");
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

  await mountComposer(4, send);
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

  await mountComposer(2);
  await failed;
  await waitFor(() => expect(backend.organization).toHaveBeenCalledOnce());
  const message = input();
  expect(message.disabled).toBe(true);
  expect(screen.queryByRole("alert")).toBeNull();

  await act(async () => {
    for (const listener of listeners) listener({ type: "connection.restored" });
  });

  expect(message.isConnected).toBe(true);
  await waitFor(() => expect(message.disabled).toBe(false));
  expect(message.value).toBe("Recovered message");
  expect(screen.queryByRole("alert")).toBeNull();
  expect(backend.organization).toHaveBeenCalledTimes(2);
});
