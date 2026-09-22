import { beforeEach, describe, expect, it, vi } from "vitest";
import { type Draft, DraftController } from "@/features/discussions/draft";
import { backend } from "@/lib/backend";

const storage = vi.hoisted(() => ({
  put: vi.fn<(store: string, draft: Draft) => Promise<void>>(),
}));
vi.mock("idb", () => ({
  openDB: async () => ({ put: storage.put, close: () => {} }),
}));
vi.mock("@/lib/backend", () => ({
  backend: {
    sendStatus: vi.fn(),
    createUpload: vi.fn(),
    uploadStatus: vi.fn(),
    uploadFile: vi.fn(),
    cancelUploads: vi.fn(),
  },
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => {
    resolve = complete;
  });
  return { promise, resolve };
}

function draft(body = "First message"): Draft {
  return {
    key: "organization:1:1:tab",
    updatedAt: 0,
    body,
    bodyRevision: 0,
    files: [],
    pending: null,
  };
}

beforeEach(() => {
  vi.resetAllMocks();
  storage.put.mockResolvedValue();
});

describe("persistent send attempts", () => {
  it("preserves text edited while the previous message is sending", async () => {
    const controller = new DraftController(1, draft());
    const sent = deferred<boolean>();
    const onSend = vi.fn(() => sent.promise);
    const operation = controller.send(onSend);
    await vi.waitFor(() => expect(onSend).toHaveBeenCalledOnce());
    controller.setBody("Next message");
    sent.resolve(true);
    await operation;
    expect(controller.snapshot().draft.body).toBe("Next message");
    expect(controller.snapshot().draft.pending).toBeNull();
    expect(onSend.mock.calls[0]).toEqual([
      "First message",
      [],
      expect.any(String),
    ]);
  });

  it("checks the same send ID after losing a response", async () => {
    const controller = new DraftController(1, draft());
    const onSend = vi.fn().mockResolvedValue(false);
    await controller.send(onSend);
    const attempt = controller.snapshot().draft.pending;
    expect(attempt?.phase).toBe("sending");
    expect(controller.snapshot().draft.body).toBe("First message");
    vi.mocked(backend.sendStatus).mockResolvedValue({
      message: { id: 9 },
    } as Awaited<ReturnType<typeof backend.sendStatus>>);
    await controller.send(onSend);
    expect(backend.sendStatus).toHaveBeenCalledWith(1, attempt?.id);
    expect(onSend).toHaveBeenCalledOnce();
    expect(controller.snapshot().draft.pending).toBeNull();
    expect(controller.snapshot().draft.body).toBe("");
  });

  it("retains the receipt until the local completion can be saved", async () => {
    const controller = new DraftController(1, draft());
    const onSend = vi.fn(async () => {
      storage.put.mockRejectedValue(new Error("Storage quota exceeded"));
      return true;
    });
    await controller.send(onSend);
    expect(controller.snapshot().draft.pending?.phase).toBe("sending");
    expect(controller.snapshot().storageError).toContain(
      "Storage quota exceeded",
    );
    storage.put.mockResolvedValue();
    vi.mocked(backend.sendStatus).mockResolvedValue({
      message: { id: 9 },
    } as Awaited<ReturnType<typeof backend.sendStatus>>);
    await controller.checkResult();
    expect(controller.snapshot().draft.pending).toBeNull();
    expect(controller.snapshot().storageError).toBeNull();
    expect(onSend).toHaveBeenCalledOnce();
  });

  it("blocks a second send while recovery is checking a receipt", async () => {
    const initial = draft();
    initial.pending = {
      id: "attempt",
      body: initial.body,
      bodyRevision: 0,
      files: [],
      phase: "sending",
    };
    const controller = new DraftController(1, initial);
    const response = deferred<Awaited<ReturnType<typeof backend.sendStatus>>>();
    vi.mocked(backend.sendStatus).mockReturnValue(response.promise);
    const check = controller.checkResult();
    const send = vi.fn();
    await controller.send(send);
    expect(send).not.toHaveBeenCalled();
    response.resolve({ message: null });
    await check;
    expect(controller.snapshot().busy).toBe(false);
    expect(controller.snapshot().draft.pending?.id).toBe("attempt");
  });

  it("keeps newly added files separate from an active send", async () => {
    const controller = new DraftController(1, draft(""));
    const first = new File(["first"], "first.txt");
    const next = new File(["next"], "next.txt");
    controller.addFiles([first]);
    vi.mocked(backend.createUpload).mockResolvedValue({
      id: "upload-first",
      state: "reserved",
      expires_at: Date.now() / 1000 + 60,
    });
    const uploading =
      deferred<Awaited<ReturnType<typeof backend.uploadFile>>>();
    vi.mocked(backend.uploadFile).mockReturnValue(uploading.promise);
    const send = vi.fn().mockResolvedValue(true);
    const operation = controller.send(send);
    await vi.waitFor(() => expect(backend.uploadFile).toHaveBeenCalledOnce());
    controller.removeFile(controller.snapshot().draft.files[0].id);
    controller.addFiles([next]);
    controller.setBody("Next message");
    uploading.resolve({
      id: "upload-first",
      name: first.name,
      size: first.size,
      media_type: "text/plain",
      sha256: "digest",
      width: null,
      height: null,
    });
    await operation;
    expect(send).toHaveBeenCalledWith("", ["upload-first"], expect.any(String));
    expect(controller.snapshot().draft.body).toBe("Next message");
    expect(controller.snapshot().draft.files.map((item) => item.file)).toEqual([
      next,
    ]);
    expect(controller.snapshot().draft.pending).toBeNull();
  });

  it("restores an interrupted upload with the saved file and a new upload ID", async () => {
    const initial = draft("");
    const file = new File(["content"], "resume.txt");
    const item = {
      id: "file",
      clientId: "old-client",
      file,
      upload: { id: "old-upload", state: "receiving" as const, expires_at: 0 },
    };
    initial.files = [item];
    initial.pending = {
      id: "send-attempt",
      body: "",
      bodyRevision: 0,
      files: [item],
      phase: "uploading",
    };
    vi.mocked(backend.uploadStatus).mockResolvedValue([
      { id: "old-upload", state: "expired", expires_at: 0 },
    ]);
    vi.mocked(backend.createUpload).mockResolvedValue({
      id: "new-upload",
      state: "reserved",
      expires_at: Date.now() / 1000 + 60,
    });
    const controller = new DraftController(1, initial);
    const send = vi.fn().mockResolvedValue(true);
    await controller.send(send);
    expect(backend.cancelUploads).toHaveBeenCalledWith(["old-upload"]);
    expect(backend.createUpload).toHaveBeenCalledWith(
      1,
      expect.any(String),
      file,
    );
    expect(vi.mocked(backend.createUpload).mock.calls[0][1]).not.toBe(
      "old-client",
    );
    expect(backend.uploadFile).toHaveBeenCalledWith(
      "new-upload",
      file,
      expect.any(AbortSignal),
      expect.any(Function),
    );
    expect(send).toHaveBeenCalledWith("", ["new-upload"], "send-attempt");
    expect(controller.snapshot().draft.files).toEqual([]);
  });

  it("preserves the file and attempt after cancelling an upload", async () => {
    const controller = new DraftController(1, draft(""));
    const file = new File(["content"], "cancel.txt");
    controller.addFiles([file]);
    vi.mocked(backend.createUpload).mockResolvedValue({
      id: "upload",
      state: "reserved",
      expires_at: Date.now() / 1000 + 60,
    });
    vi.mocked(backend.uploadFile).mockImplementation(
      async (_id, _file, signal) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason));
        }),
    );
    const send = vi.fn();
    const operation = controller.send(send);
    await vi.waitFor(() => expect(backend.uploadFile).toHaveBeenCalledOnce());
    controller.cancel();
    await operation;
    expect(send).not.toHaveBeenCalled();
    expect(controller.snapshot().draft.files[0].file).toBe(file);
    expect(controller.snapshot().draft.pending?.files[0].file).toBe(file);
    expect(controller.snapshot().busy).toBe(false);
    expect(controller.snapshot().error).toBeTruthy();
  });

  it("does not transmit when saving the attempt fails", async () => {
    storage.put.mockRejectedValue(new Error("Storage unavailable"));
    const controller = new DraftController(1, draft());
    const send = vi.fn();
    await controller.send(send);
    expect(send).not.toHaveBeenCalled();
    expect(controller.snapshot().draft.body).toBe("First message");
    expect(controller.snapshot().storageError).toContain("Storage unavailable");
  });
});
