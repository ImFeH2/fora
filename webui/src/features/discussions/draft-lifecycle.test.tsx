import "fake-indexeddb/auto";
import { IDBFactory } from "fake-indexeddb";
import { openDB } from "idb";
import { StrictMode } from "react";
import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";
import {
  type Draft,
  type DraftController,
  useDraft,
} from "@/features/discussions/draft";
import { backend } from "@/lib/backend";

const storage = vi.hoisted(() => ({
  gate: null as Promise<void> | null,
  fail: false,
}));
vi.mock("idb", async (importOriginal) => {
  const actual = await importOriginal<typeof import("idb")>();
  return {
    ...actual,
    openDB: async (...args: Parameters<typeof actual.openDB>) => {
      if (storage.gate) await storage.gate;
      if (storage.fail) throw new Error("Controlled IndexedDB save failure");
      return actual.openDB(...args);
    },
  };
});

let renderHook: typeof import("@testing-library/react").renderHook;
let cleanup: typeof import("@testing-library/react").cleanup;
let waitFor: typeof import("@testing-library/react").waitFor;
let act: typeof import("@testing-library/react").act;
let closeDom: () => void;
let discussion = 1000;

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
  vi.stubGlobal("HTMLElement", dom.window.HTMLElement);
  vi.stubGlobal("Node", dom.window.Node);
  vi.stubGlobal("MutationObserver", dom.window.MutationObserver);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  sessionStorage.setItem("fora.draft-tab", "lifecycle-tab");
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: vi.fn(async (_name, _options, callback) => callback({})),
    },
  });
  ({ renderHook, cleanup, waitFor, act } = await import(
    "@testing-library/react"
  ));
}, 30000);

beforeEach(() => {
  vi.stubGlobal("indexedDB", new IDBFactory());
  storage.gate = null;
  storage.fail = false;
  vi.spyOn(backend, "organization").mockResolvedValue({
    uuid: "lifecycle-organization",
    human_id: 1,
  } as Awaited<ReturnType<typeof backend.organization>>);
  vi.spyOn(backend, "onEvent").mockReturnValue(() => {});
  vi.spyOn(backend, "sendStatus").mockResolvedValue({
    state: "unknown",
    message: null,
  });
  vi.spyOn(backend, "cancelUploads").mockResolvedValue({ cancelled: 1 });
});

afterEach(async () => {
  await act(async () => cleanup());
  vi.restoreAllMocks();
});

afterAll(() => {
  closeDom();
  vi.unstubAllGlobals();
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

function draft(id: number, body = "Saved message"): Draft {
  return {
    key: `lifecycle-organization:1:${id}:lifecycle-tab`,
    updatedAt: 0,
    body,
    bodyRevision: 0,
    files: [],
    pending: null,
  };
}

async function database() {
  return openDB("fora-drafts", 1, {
    upgrade(db) {
      db.createObjectStore("drafts", { keyPath: "key" });
    },
  });
}

async function seed(value: Draft) {
  const db = await database();
  try {
    await db.put("drafts", value);
  } finally {
    db.close();
  }
}

function requireController(controller: DraftController | null) {
  if (!controller) throw new Error("Draft controller did not load");
  return controller;
}

async function mount(id: number) {
  const hook = renderHook(() => useDraft(id));
  await waitFor(() => expect(hook.result.current.controller).not.toBeNull());
  return hook;
}

it("releases saved drafts and restores text and File bytes through IndexedDB", async () => {
  const id = discussion++;
  const value = draft(id);
  value.files = [
    {
      id: "file",
      clientId: "file-client",
      file: new File(["attachment bytes"], "saved.txt", { type: "text/plain" }),
    },
  ];
  await seed(value);
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  act(() => original.setBody("Updated saved message"));
  await waitFor(() => expect(first.result.current.view?.saving).toBe(false));
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).not.toBe(original);
  const restored = second.result.current.view?.draft;
  expect(restored?.key).toBe(value.key);
  expect(restored?.body).toBe("Updated saved message");
  expect(restored?.files[0].file).toBeInstanceOf(File);
  expect(restored?.files[0].file.name).toBe("saved.txt");
  expect(restored?.files[0].file.type).toBe("text/plain");
  expect(await restored?.files[0].file.text()).toBe("attachment bytes");
  expect(restored?.pending).toBeNull();
});

it("keeps one controller until every simultaneous user has left", async () => {
  const id = discussion++;
  const first = await mount(id);
  const second = await mount(id);
  const original = requireController(first.result.current.controller);
  expect(second.result.current.controller).toBe(original);
  first.unmount();
  const third = await mount(id);
  expect(third.result.current.controller).toBe(original);
  second.unmount();
  third.unmount();
  const fourth = await mount(id);
  expect(fourth.result.current.controller).not.toBe(original);
  expect(backend.organization).toHaveBeenCalledTimes(2);
});

it("preserves the loading identity during StrictMode and late completion", async () => {
  const id = discussion++;
  const organization =
    deferred<Awaited<ReturnType<typeof backend.organization>>>();
  vi.mocked(backend.organization).mockReturnValueOnce(organization.promise);
  const first = renderHook(() => useDraft(id), { wrapper: StrictMode });
  expect(first.result.current.controller).toBeNull();
  first.unmount();
  const second = renderHook(() => useDraft(id), { wrapper: StrictMode });
  await act(async () => {
    organization.resolve({
      uuid: "lifecycle-organization",
      human_id: 1,
    } as Awaited<ReturnType<typeof backend.organization>>);
  });
  await waitFor(() => expect(second.result.current.controller).not.toBeNull());
  expect(backend.organization).toHaveBeenCalledOnce();
  const original = requireController(second.result.current.controller);
  second.unmount();
  const third = await mount(id);
  expect(third.result.current.controller).not.toBe(original);
  expect(backend.organization).toHaveBeenCalledTimes(2);
});

it("keeps asynchronous saves through departure and releases after the last write", async () => {
  const id = discussion++;
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  const gate = deferred<void>();
  storage.gate = gate.promise;
  act(() => {
    original.setBody("First edit");
    original.setBody("Last edit");
  });
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  expect(second.result.current.view?.draft.body).toBe("Last edit");
  second.unmount();
  await act(async () => {
    storage.gate = null;
    gate.resolve();
  });
  await waitFor(() => expect(original.snapshot().saving).toBe(false));
  const third = await mount(id);
  expect(third.result.current.controller).not.toBe(original);
  expect(third.result.current.view?.draft.body).toBe("Last edit");
});

it("retains unsaved body, File and the retry error across departure", async () => {
  const id = discussion++;
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  storage.fail = true;
  const file = new File(["unsaved bytes"], "unsaved.txt");
  act(() => {
    original.setBody("Unsaved message");
    original.addFiles([file]);
  });
  await waitFor(() =>
    expect(first.result.current.view?.storageError).toContain("save failure"),
  );
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  expect(second.result.current.view?.draft.body).toBe("Unsaved message");
  expect(second.result.current.view?.draft.files[0].file).toBe(file);
  expect(second.result.current.view?.storageError).toContain("save failure");
  storage.fail = false;
  act(() => original.saveAgain());
  await waitFor(() =>
    expect(second.result.current.view?.storageError).toBeNull(),
  );
  second.unmount();
  const third = await mount(id);
  expect(third.result.current.controller).not.toBe(original);
  expect(third.result.current.view?.draft.body).toBe("Unsaved message");
  expect(await third.result.current.view?.draft.files[0].file.text()).toBe(
    "unsaved bytes",
  );
});

it("keeps an upload and send on the original controller while every user leaves", async () => {
  const id = discussion++;
  const value = draft(id);
  value.files = [
    {
      id: "file",
      clientId: "upload-client",
      file: new File(["bytes"], "upload.txt"),
    },
  ];
  await seed(value);
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  const uploading = deferred<Awaited<ReturnType<typeof backend.uploadFile>>>();
  const sending = deferred<boolean>();
  vi.spyOn(backend, "createUpload").mockResolvedValue({
    id: "upload",
    state: "reserved",
    expires_at: Date.now() / 1000 + 3600,
  });
  vi.spyOn(backend, "uploadFile").mockReturnValue(uploading.promise);
  const send = vi.fn(() => sending.promise);
  let operation!: Promise<void>;
  act(() => {
    operation = original.send(send);
  });
  await waitFor(() => expect(backend.uploadFile).toHaveBeenCalledOnce());
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  second.unmount();
  await act(async () => {
    uploading.resolve({
      id: "upload",
      name: "upload.txt",
      size: 5,
      media_type: "text/plain",
      sha256: "digest",
      width: null,
      height: null,
    });
  });
  await waitFor(() => expect(send).toHaveBeenCalledOnce());
  const third = await mount(id);
  expect(third.result.current.controller).toBe(original);
  third.unmount();
  await act(async () => {
    sending.resolve(true);
    await operation;
  });
  const fourth = await mount(id);
  expect(fourth.result.current.controller).not.toBe(original);
  expect(fourth.result.current.view?.draft.body).toBe("");
  expect(fourth.result.current.view?.draft.files).toEqual([]);
  expect(send).toHaveBeenCalledOnce();
});

it("retains a confirmed send after the completion save fails without resending", async () => {
  const id = discussion++;
  await seed(draft(id));
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  const send = vi.fn(async () => {
    storage.fail = true;
    return true;
  });
  await act(async () => {
    await original.send(send);
  });
  const pendingId = original.snapshot().draft.pending?.id;
  expect(original.snapshot().submissionResult).toEqual({
    id: pendingId,
    state: "sent",
  });
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  expect(second.result.current.view?.error).toBe("Could not update draft");
  storage.fail = false;
  act(() => original.saveAgain());
  await waitFor(() => expect(second.result.current.view?.saving).toBe(false));
  second.unmount();
  const third = await mount(id);
  expect(third.result.current.controller).toBe(original);
  expect(third.result.current.view?.submissionResult).toEqual({
    id: pendingId,
    state: "sent",
  });
  await act(async () => {
    await original.checkResult();
  });
  expect(backend.sendStatus).not.toHaveBeenCalled();
  expect(send).toHaveBeenCalledOnce();
  third.unmount();
  const fourth = await mount(id);
  expect(fourth.result.current.controller).not.toBe(original);
  expect(fourth.result.current.view?.draft.pending).toBeNull();
});

it("holds receipt lookup across departure and restores the same unknown send ID", async () => {
  const id = discussion++;
  const value = draft(id);
  value.pending = {
    id: "original-send",
    body: value.body,
    bodyRevision: 0,
    files: [],
    phase: "sending",
  };
  await seed(value);
  const receipt = deferred<Awaited<ReturnType<typeof backend.sendStatus>>>();
  vi.mocked(backend.sendStatus).mockReturnValueOnce(receipt.promise);
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  expect(backend.sendStatus).toHaveBeenCalledOnce();
  second.unmount();
  await act(async () => {
    receipt.resolve({ state: "unknown", message: null });
  });
  await waitFor(() => expect(original.snapshot().busy).toBe(false));
  const third = await mount(id);
  expect(third.result.current.controller).not.toBe(original);
  expect(third.result.current.view?.draft.pending?.id).toBe("original-send");
  await waitFor(() => expect(backend.sendStatus).toHaveBeenCalledTimes(2));
  expect(backend.sendStatus).toHaveBeenNthCalledWith(2, id, "original-send");
});

it("preserves confirmed cancellation and finishes recovery after leaving", async () => {
  const id = discussion++;
  const value = draft(id);
  value.pending = {
    id: "cancel-send",
    body: value.body,
    bodyRevision: 0,
    files: [],
    phase: "sending",
  };
  await seed(value);
  const first = await mount(id);
  await waitFor(() => expect(first.result.current.view?.busy).toBe(false));
  const original = requireController(first.result.current.controller);
  const cancelling = deferred<Awaited<ReturnType<typeof backend.cancelSend>>>();
  vi.spyOn(backend, "cancelSend").mockReturnValueOnce(cancelling.promise);
  let operation!: Promise<void>;
  act(() => {
    operation = original.discardAttempt();
  });
  first.unmount();
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  storage.fail = true;
  await act(async () => {
    cancelling.resolve({ state: "cancelled", message: null });
    await operation;
  });
  const result = original.snapshot().submissionResult;
  expect(result).toEqual({ id: "cancel-send", state: "cancelled" });
  second.unmount();
  const third = await mount(id);
  expect(third.result.current.controller).toBe(original);
  storage.fail = false;
  await act(async () => {
    await original.checkResult();
  });
  third.unmount();
  const fourth = await mount(id);
  expect(fourth.result.current.controller).not.toBe(original);
  expect(fourth.result.current.view?.draft.pending).toBeNull();
  expect(fourth.result.current.view?.draft.body).toBe("Saved message");
  expect(backend.cancelSend).toHaveBeenCalledOnce();
});

it("holds voice draft writes and subscriptions until both have ended", async () => {
  const id = discussion++;
  const first = await mount(id);
  const original = requireController(first.result.current.controller);
  const gate = deferred<void>();
  storage.gate = gate.promise;
  const unsubscribe = original.subscribe(() => {});
  act(() => original.setVoiceBody("Transcribed message", null));
  first.unmount();
  await act(async () => {
    storage.gate = null;
    gate.resolve();
  });
  await waitFor(() => expect(original.snapshot().saving).toBe(false));
  const second = await mount(id);
  expect(second.result.current.controller).toBe(original);
  second.unmount();
  unsubscribe();
  const third = await mount(id);
  expect(third.result.current.controller).not.toBe(original);
  expect(third.result.current.view?.draft.body).toBe("Transcribed message");
});

it("does not expose the previous discussion while the next draft is loading", async () => {
  const firstId = discussion++;
  const secondId = discussion++;
  await seed(draft(firstId, "First discussion"));
  await seed(draft(secondId, "Second discussion"));
  const hook = renderHook(({ id }) => useDraft(id), {
    initialProps: { id: firstId },
  });
  await waitFor(() => expect(hook.result.current.controller).not.toBeNull());
  const original = requireController(hook.result.current.controller);
  const gate = deferred<void>();
  storage.gate = gate.promise;
  hook.rerender({ id: secondId });
  expect(hook.result.current.controller).toBeNull();
  expect(hook.result.current.view).toBeNull();
  hook.rerender({ id: firstId });
  await act(async () => {
    storage.gate = null;
    gate.resolve();
  });
  await waitFor(() =>
    expect(hook.result.current.view?.draft.body).toBe("First discussion"),
  );
  expect(hook.result.current.controller).not.toBe(original);
  hook.unmount();
  const next = await mount(secondId);
  expect(next.result.current.view?.draft.body).toBe("Second discussion");
});

it("recovers an available older tab with the original body, File and send ID", async () => {
  const id = discussion++;
  const value = draft(id);
  value.key = `lifecycle-organization:1:${id}:older-tab`;
  value.files = [
    {
      id: "file",
      clientId: "client",
      file: new File(["older bytes"], "older.txt"),
    },
  ];
  value.pending = {
    id: "older-send",
    body: value.body,
    bodyRevision: 0,
    files: value.files,
    phase: "sending",
  };
  await seed(value);
  const first = await mount(id);
  expect(first.result.current.view?.draft.key).toBe(draft(id).key);
  expect(first.result.current.view?.draft.pending?.id).toBe("older-send");
  expect(await first.result.current.view?.draft.files[0].file.text()).toBe(
    "older bytes",
  );
  const db = await database();
  try {
    expect(await db.get("drafts", value.key)).toBeUndefined();
  } finally {
    db.close();
  }
});

it("preserves saved content across two warmup and five full rounds of ten discussions", async () => {
  const ids = Array.from({ length: 10 }, () => discussion++);
  for (const [index, id] of ids.entries()) {
    const value = draft(id, index % 2 === 0 ? `Discussion ${id}` : "");
    if (index % 3 === 0)
      value.files = [
        {
          id: `file-${id}`,
          clientId: `client-${id}`,
          file: new File([`attachment ${id}`], `file-${id}.txt`),
        },
      ];
    await seed(value);
  }
  const previous = new Map<number, ReturnType<typeof useDraft>["controller"]>();
  for (let round = 0; round < 7; round++) {
    for (const [index, id] of ids.entries()) {
      const hook = await mount(id);
      const controller = hook.result.current.controller;
      if (previous.has(id)) expect(controller).not.toBe(previous.get(id));
      expect(hook.result.current.view?.draft.body).toBe(
        index % 2 === 0 ? `Discussion ${id}` : "",
      );
      if (index % 3 === 0)
        expect(await hook.result.current.view?.draft.files[0].file.text()).toBe(
          `attachment ${id}`,
        );
      previous.set(id, controller);
      hook.unmount();
    }
  }
  expect(backend.organization).toHaveBeenCalledTimes(70);
}, 30000);
