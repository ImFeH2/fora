// @vitest-environment jsdom
import { act, StrictMode, useSyncExternalStore } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { clearToasts, readToasts } from "@/components/ui/toast";
import { Composer } from "@/features/discussions/composer";
import {
  type Draft,
  DraftController,
  type SendDraft,
} from "@/features/discussions/draft";
import { BackendError, backend } from "@/lib/backend";
import type { VoiceEvent } from "@/lib/voice";

const harness = vi.hoisted(() => ({
  controller: null as DraftController | null,
  controllers: new Map<number, DraftController>(),
  callbacks: [] as ((event: VoiceEvent) => void)[],
  cancel: vi.fn(),
  put: vi.fn(),
}));
vi.mock("idb", () => ({
  openDB: async () => ({ put: harness.put, close: () => {} }),
}));
vi.mock("@/features/discussions/draft", async (original) => ({
  ...(await original<typeof import("@/features/discussions/draft")>()),
  useDraft: (discussionId: number) => {
    const controller =
      (discussionId === 1 ? harness.controller : null) ??
      harness.controllers.get(discussionId) ??
      null;
    const snapshot = controller?.snapshot ?? (() => null);
    const subscribe = controller?.subscribe ?? (() => () => {});
    const view = useSyncExternalStore(subscribe, snapshot, snapshot);
    return { controller, view, error: null };
  },
}));
vi.mock("@/lib/voice", () => ({
  VoiceRecording: class {
    #callback: (event: VoiceEvent) => void;
    constructor(callback: (event: VoiceEvent) => void) {
      this.#callback = callback;
      harness.callbacks.push(callback);
    }
    start = async () => {
      this.#callback({ type: "state", state: "recording" });
    };
    stop = vi.fn(() => {
      this.#callback({ type: "state", state: "closed" });
    });
    cancel = harness.cancel;
  },
}));

function draft(body = "Saved message"): Draft {
  return {
    key: "voice:1:1:tab",
    updatedAt: 0,
    body,
    bodyRevision: 0,
    files: [],
    pending: null,
  };
}
function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}
function mount(
  discussionId = 1,
  autoStart = true,
  onOpenVoiceSettings: () => void = () => {},
  strictMode = false,
  onSend: SendDraft = async () => true,
) {
  const container = document.createElement("div");
  document.body.append(container);
  const root: Root = createRoot(container);
  let currentDiscussion = discussionId;
  const render = (id: number) => {
    currentDiscussion = id;
    const composer = (
      <Composer
        discussionId={id}
        members={[]}
        memberIds={new Set()}
        busy={false}
        placeholder="Message"
        onSend={onSend}
        onHeightChange={() => {}}
        onOpenVoiceSettings={onOpenVoiceSettings}
      />
    );
    act(() =>
      root.render(strictMode ? <StrictMode>{composer}</StrictMode> : composer),
    );
  };
  render(discussionId);
  let callback: ((event: VoiceEvent) => void) | undefined;
  let ready = Promise.resolve();
  const start = () => {
    const button = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Start voice input"]',
    );
    if (!button) throw new Error("Voice input start button is missing");
    const callbackIndex = harness.callbacks.length;
    act(() => button.click());
    ready = vi.waitFor(() => {
      callback = harness.callbacks[callbackIndex];
      if (!callback) throw new Error("Voice recording callback is missing");
    });
    return ready;
  };
  if (autoStart) void start();
  return {
    container,
    render,
    start,
    ready: () => ready,
    emit: (event: VoiceEvent) =>
      act(async () => {
        await ready;
        callback?.(event);
        const controller =
          (currentDiscussion === 1 ? harness.controller : null) ??
          harness.controllers.get(currentDiscussion);
        if (controller)
          await vi.waitFor(() =>
            expect(controller.snapshot().saving).toBe(false),
          );
      }),
    close: () => {
      act(() => root.unmount());
      container.remove();
    },
  };
}
function body(voice: ReturnType<typeof mount>) {
  const input = voice.container.querySelector<HTMLTextAreaElement>(
    'textarea[aria-label="Message"]',
  );
  if (!input) throw new Error("Composer textarea is missing");
  return input.value;
}
function transcript(text: string): VoiceEvent {
  return { type: "transcript", text };
}

beforeEach(() => {
  vi.restoreAllMocks();
  harness.controllers.clear();
  harness.callbacks = [];
  harness.cancel.mockReset();
  harness.put.mockReset().mockResolvedValue(undefined);
  harness.controller = new DraftController(1, draft());
  vi.spyOn(backend, "settings").mockResolvedValue({
    address: "wss://example.test/transcription",
    model: "test-model",
    api_key_set: true,
  });
  clearToasts();
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("getComputedStyle", () => ({
    lineHeight: "20px",
    paddingTop: "0px",
    paddingBottom: "0px",
    borderTopWidth: "0px",
    borderBottomWidth: "0px",
  }));
  Object.defineProperty(document, "fonts", {
    configurable: true,
    value: { ready: Promise.resolve() },
  });
});
afterEach(() => {
  clearToasts();
  vi.unstubAllGlobals();
  document.body.replaceChildren();
});

it("checks saved voice settings before opening voice input", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const openVoiceSettings = vi.fn();
  vi.mocked(backend.settings).mockResolvedValueOnce({
    address: "wss://example.test/transcription",
    model: "test-model",
    api_key_set: false,
  });
  const voice = mount(1, false, openVoiceSettings);
  const button = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Start voice input"]',
  );
  if (!button) throw new Error("Voice input start button is missing");
  await act(async () => {
    button.click();
    await vi.waitFor(() =>
      expect(
        readToasts().find((item) => item.id === "voice-settings"),
      ).toMatchObject({
        title: "Set up voice transcription",
        action: { label: "Open Voice settings" },
      }),
    );
  });
  expect(backend.settings).toHaveBeenCalledWith("voice");
  expect(harness.callbacks).toHaveLength(0);
  expect(controller.snapshot().draft.body).toBe("Saved message");
  const notification = readToasts().find(
    (item) => item.id === "voice-settings",
  );
  notification?.action?.onClick();
  expect(openVoiceSettings).toHaveBeenCalledOnce();
  voice.close();
});

it("starts configured voice input after a StrictMode remount", async () => {
  const voice = mount(1, false, () => {}, true);
  await voice.start();
  expect(harness.callbacks).toHaveLength(1);
  voice.close();
});

it("shows the voice settings notification after a StrictMode remount", async () => {
  const openVoiceSettings = vi.fn();
  vi.mocked(backend.settings).mockResolvedValueOnce({
    address: "wss://example.test/transcription",
    model: "test-model",
    api_key_set: false,
  });
  const voice = mount(1, false, openVoiceSettings, true);
  const button = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Start voice input"]',
  );
  if (!button) throw new Error("Voice input start button is missing");
  await act(async () => {
    button.click();
    await vi.waitFor(() =>
      expect(
        readToasts().find((item) => item.id === "voice-settings"),
      ).toMatchObject({
        title: "Set up voice transcription",
        action: { label: "Open Voice settings" },
      }),
    );
  });
  expect(harness.callbacks).toHaveLength(0);
  voice.close();
});

it("ignores a voice settings response after StrictMode unmount", async () => {
  const settingsRead = deferred<Record<string, unknown>>();
  vi.mocked(backend.settings).mockImplementationOnce(
    () => settingsRead.promise,
  );
  const voice = mount(1, false, () => {}, true);
  const button = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Start voice input"]',
  );
  if (!button) throw new Error("Voice input start button is missing");
  act(() => button.click());
  await vi.waitFor(() => expect(backend.settings).toHaveBeenCalledTimes(1));
  voice.close();
  await act(async () => {
    settingsRead.resolve({
      address: "wss://example.test/transcription",
      model: "test-model",
      api_key_set: true,
    });
    await settingsRead.promise;
  });
  expect(harness.callbacks).toHaveLength(0);
  expect(readToasts()).toEqual([]);
});

it("shows the saved voice settings read error without changing the draft", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  vi.mocked(backend.settings).mockRejectedValueOnce(
    new Error("Voice settings unavailable"),
  );
  const voice = mount(1, false);
  const button = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Start voice input"]',
  );
  if (!button) throw new Error("Voice input start button is missing");
  await act(async () => {
    button.click();
    await vi.waitFor(() =>
      expect(backend.settings).toHaveBeenCalledWith("voice"),
    );
    expect(backend.settings).toHaveBeenCalledTimes(1);
  });
  await vi.waitFor(() =>
    expect(
      readToasts().find((item) => item.id === "voice-input"),
    ).toMatchObject({
      title: "Could not read voice settings",
      description: "Voice settings unavailable",
    }),
  );
  expect(harness.callbacks).toHaveLength(0);
  expect(controller.snapshot().draft.body).toBe("Saved message");
  voice.close();
});

it("routes backend voice configuration errors to voice settings", async () => {
  const openVoiceSettings = vi.fn();
  const voice = mount(1, false, openVoiceSettings);
  await voice.start();
  await voice.emit({
    type: "error",
    code: "voice_config",
    message: "Invalid transcription settings",
  });
  const notification = readToasts().find(
    (item) => item.id === "voice-settings",
  );
  expect(notification).toMatchObject({
    title: "Set up voice transcription",
    action: { label: "Open Voice settings" },
  });
  expect(voice.container.textContent).not.toContain(
    "Invalid transcription settings",
  );
  notification?.action?.onClick();
  expect(openVoiceSettings).toHaveBeenCalledOnce();
  voice.close();
});

it("keeps ordinary voice errors in composer feedback", async () => {
  const voice = mount(1, false);
  await voice.start();
  await voice.emit({
    type: "error",
    code: "voice_capacity",
    message: "Audio capacity reached",
  });
  await vi.waitFor(() =>
    expect(
      readToasts().find((item) => item.id === "voice-input"),
    ).toMatchObject({
      title: "Could not transcribe voice input",
      description: "Audio capacity reached",
    }),
  );
  expect(voice.container.textContent).not.toContain("Audio capacity reached");
  expect(
    readToasts().find((item) => item.id === "voice-settings"),
  ).toBeUndefined();
  voice.close();
});

it("keeps an unknown Discussion send in Composer recovery feedback", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const failure = new BackendError(
    "unconfirmed",
    "Your message may have been sent. Check the discussion before sending it again.",
    true,
    "discussion.send",
  );
  const onSend = vi.fn<SendDraft>(async () => {
    throw failure;
  });
  const voice = mount(1, false, () => {}, false, onSend);
  const send = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Send · Enter"]',
  );
  if (!send) throw new Error("Send button is missing");
  await act(async () => {
    send.click();
    await vi.waitFor(() => expect(controller.snapshot().busy).toBe(false));
  });
  expect(onSend).toHaveBeenCalledOnce();
  expect(controller.snapshot().draft.pending?.phase).toBe("sending");
  expect(controller.snapshot().error).toBe(failure.message);
  expect(voice.container.textContent).toContain("Check result");
  expect(readToasts()).toEqual([]);
  voice.close();
});

it("dismisses a voice error when Composer unmounts", async () => {
  const voice = mount(1, false);
  await voice.start();
  await voice.emit({
    type: "error",
    code: "voice_capacity",
    message: "Audio capacity reached",
  });
  await vi.waitFor(() =>
    expect(
      readToasts().find((item) => item.id === "voice-input" && item.open),
    ).toBeDefined(),
  );
  voice.close();
  expect(
    readToasts().find((item) => item.id === "voice-input" && item.open),
  ).toBeUndefined();
});

it("releases a stale voice settings read after switching discussions", async () => {
  const firstRead = deferred<Record<string, unknown>>();
  const secondRead = deferred<Record<string, unknown>>();
  const settings = {
    address: "wss://example.test/transcription",
    model: "test-model",
    api_key_set: true,
  };
  vi.mocked(backend.settings)
    .mockImplementationOnce(() => firstRead.promise)
    .mockImplementationOnce(() => secondRead.promise);
  const other = new DraftController(2, draft("Other Discussion"));
  harness.controllers.set(2, other);
  const voice = mount(1, false);
  const firstButton = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Start voice input"]',
  );
  if (!firstButton) throw new Error("Voice input start button is missing");
  act(() => firstButton.click());
  await vi.waitFor(() => expect(backend.settings).toHaveBeenCalledTimes(1));
  voice.render(2);
  const secondStart = voice.start();
  await vi.waitFor(() => expect(backend.settings).toHaveBeenCalledTimes(2));
  await act(async () => {
    firstRead.resolve(settings);
    await firstRead.promise;
  });
  expect(harness.callbacks).toHaveLength(0);
  await act(async () => {
    secondRead.resolve(settings);
    await secondStart;
  });
  expect(harness.callbacks).toHaveLength(1);
  voice.close();
});

it("suppresses a stale voice settings read error after switching discussions", async () => {
  const firstRead = deferred<Record<string, unknown>>();
  const secondRead = deferred<Record<string, unknown>>();
  const settings = {
    address: "wss://example.test/transcription",
    model: "test-model",
    api_key_set: true,
  };
  vi.mocked(backend.settings)
    .mockImplementationOnce(() => firstRead.promise)
    .mockImplementationOnce(() => secondRead.promise);
  const other = new DraftController(2, draft("Other Discussion"));
  harness.controllers.set(2, other);
  const voice = mount(1, false);
  const firstButton = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Start voice input"]',
  );
  if (!firstButton) throw new Error("Voice input start button is missing");
  act(() => firstButton.click());
  await vi.waitFor(() => expect(backend.settings).toHaveBeenCalledTimes(1));
  voice.render(2);
  const secondStart = voice.start();
  await vi.waitFor(() => expect(backend.settings).toHaveBeenCalledTimes(2));
  const failure = new Error("Old discussion settings failed");
  await act(async () => {
    firstRead.reject(failure);
    await expect(firstRead.promise).rejects.toBe(failure);
  });
  expect(voice.container.textContent).not.toContain(
    "Old discussion settings failed",
  );
  await act(async () => {
    secondRead.resolve(settings);
    await secondStart;
  });
  expect(harness.callbacks).toHaveLength(1);
  voice.close();
});

it("keeps saving feedback while draft persistence is pending", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: () => void;
  harness.put.mockImplementationOnce(
    () =>
      new Promise<void>((resolve) => {
        finish = resolve;
      }),
  );
  controller.setBody("Changed draft");
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const voice = mount(1, false);
  expect(voice.container.textContent).not.toContain("Saving draft");
  finish();
  await vi.waitFor(() => expect(controller.snapshot().saving).toBe(false));
  expect(voice.container.textContent).not.toContain("Draft saved");
  voice.close();
});

it("persists partial and final while keeping the active submission snapshot", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: (value: boolean) => void;
  const send = vi.fn(
    () =>
      new Promise<boolean>((resolve) => {
        finish = resolve;
      }),
  );
  const operation = controller.send(send);
  await vi.waitFor(() => expect(send).toHaveBeenCalledOnce());
  const pending = structuredClone(controller.snapshot().draft.pending);
  const voice = mount();
  await voice.ready();
  expect(voice.container.textContent).not.toContain("Listening…");
  expect(voice.container.textContent).toContain("Cancel");
  expect(voice.container.textContent).not.toContain("Saved send attempt");
  expect(voice.container.textContent).not.toContain("Draft saved");
  await voice.emit(transcript(" partial"));
  await voice.emit(transcript(" final"));
  expect(controller.snapshot().draft.pending).toEqual(pending);
  await act(async () => {
    finish(true);
    await operation;
  });
  expect(controller.snapshot().draft.body).toBe(" final");
  expect(controller.snapshot().draft.pending).toBeNull();
  expect(harness.put).toHaveBeenLastCalledWith(
    "drafts",
    expect.objectContaining({ body: " final" }),
  );
});

it("keeps only new voice text when sending finishes before the first transcript", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: (value: boolean) => void;
  const operation = controller.send(
    () =>
      new Promise<boolean>((resolve) => {
        finish = resolve;
      }),
  );
  await vi.waitFor(() =>
    expect(controller.snapshot().draft.pending).not.toBeNull(),
  );
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const pending = structuredClone(controller.snapshot().draft.pending);
  const voice = mount();
  await act(async () => {
    finish(true);
    await operation;
  });
  expect(controller.snapshot().draft.body).toBe("");
  await voice.emit(transcript("New words"));
  expect(controller.snapshot().draft.pending).toBeNull();
  expect(controller.snapshot().draft.body).toBe("New words");
  expect(pending?.body).toBe("Saved message");
});

it("keeps partial text across send completion and final replacement", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: (value: boolean) => void;
  const operation = controller.send(
    () =>
      new Promise<boolean>((resolve) => {
        finish = resolve;
      }),
  );
  await vi.waitFor(() =>
    expect(controller.snapshot().draft.pending).not.toBeNull(),
  );
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const voice = mount();
  await voice.emit(transcript("Partial"));
  await act(async () => {
    finish(true);
    await operation;
  });
  await voice.emit(transcript("Final"));
  expect(controller.snapshot().draft.body).toBe("Final");
  expect(controller.snapshot().draft.pending).toBeNull();
});

it("preserves earlier voice text across recordings during a pending send", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: (result: boolean) => void;
  const sending = controller.send(
    () => new Promise<boolean>((resolve) => (finish = resolve)),
  );
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const voice = mount();
  await voice.emit(transcript(" one"));
  const stop = voice.container.querySelector<HTMLButtonElement>(
    'button[aria-label="Stop recording"]',
  );
  if (!stop) throw new Error("Stop recording button is missing");
  act(() => stop.click());
  voice.start();
  await voice.emit(transcript(" two"));
  expect(body(voice)).toBe("Saved message one two");
  await act(async () => {
    finish(true);
    await sending;
  });
  expect(body(voice)).toBe(" one two");
  expect(controller.snapshot().draft.pending).toBeNull();
  voice.close();
});

it("removes the submitted prefix when sending finishes after Discussion changes", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const other = new DraftController(2, draft("Other Discussion"));
  harness.controllers.set(2, other);
  let finish!: (result: boolean) => void;
  const sending = controller.send(
    () => new Promise<boolean>((resolve) => (finish = resolve)),
  );
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const voice = mount();
  await voice.emit(transcript(" New words"));
  voice.render(2);
  expect(harness.cancel).toHaveBeenCalledOnce();
  await act(async () => {
    finish(true);
    await sending;
  });
  expect(controller.snapshot().draft.body).toBe(" New words");
  expect(other.snapshot().draft.body).toBe("Other Discussion");
  expect(body(voice)).toBe("Other Discussion");
  voice.close();
});

it("retains the submitted body when a concurrent send is cancelled", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: (result: "cancelled") => void;
  const operation = controller.send(
    () => new Promise<"cancelled">((resolve) => (finish = resolve)),
  );
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const voice = mount();
  const pendingId = controller.snapshot().draft.pending?.id;
  await voice.emit(transcript(" partial"));
  expect(body(voice)).toBe("Saved message partial");
  await act(async () => {
    finish("cancelled");
    await operation;
  });
  expect(controller.snapshot().draft.pending).toBeNull();
  expect(controller.snapshot().submissionResult).toEqual({
    id: pendingId,
    state: "cancelled",
  });
  expect(body(voice)).toBe("Saved message partial");
  await voice.emit(transcript(" final"));
  expect(body(voice)).toBe("Saved message final");
  expect(controller.snapshot().submissionResult).toEqual({
    id: pendingId,
    state: "cancelled",
  });
  voice.close();
});

it("preserves a user edit during voice input after the send succeeds", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  let finish!: (result: boolean) => void;
  const sending = controller.send(
    () => new Promise<boolean>((resolve) => (finish = resolve)),
  );
  await vi.waitFor(() => expect(finish).toBeTypeOf("function"));
  const voice = mount();
  await voice.emit(transcript(" partial"));
  const input = voice.container.querySelector<HTMLTextAreaElement>(
    'textarea[aria-label="Message"]',
  );
  if (!input) throw new Error("Composer textarea is missing");
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )?.set;
    setter?.call(input, "Hand edited text");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await vi.waitFor(() => expect(controller.snapshot().saving).toBe(false));
  });
  await act(async () => {
    finish(true);
    await sending;
  });
  expect(body(voice)).toBe("Hand edited text");
  expect(controller.snapshot().draft.voiceSubmission).toBeNull();
  voice.close();
});

it("preserves voice text when upload cancellation interrupts a pending send", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const file = new File(["content"], "notes.txt", { type: "text/plain" });
  act(() => controller.addFiles([file]));
  await vi.waitFor(() => expect(controller.snapshot().saving).toBe(false));
  vi.spyOn(backend, "createUpload").mockResolvedValue({
    id: "upload-1",
    state: "reserved",
    expires_at: Date.now() / 1000 + 3600,
  });
  let uploadStarted!: () => void;
  vi.spyOn(backend, "uploadFile").mockImplementation(
    (_id, _file, signal) =>
      new Promise((_resolve, reject) => {
        uploadStarted = () => reject(signal.reason);
      }),
  );
  const sending = controller.send(async () => true);
  await vi.waitFor(() => expect(uploadStarted).toBeTypeOf("function"));
  const voice = mount();
  await voice.emit(transcript(" partial"));
  const cancelUpload = Array.from(
    voice.container.querySelectorAll<HTMLButtonElement>("button"),
  ).find((button) => button.textContent === "Cancel upload");
  if (!cancelUpload) throw new Error("Cancel upload button is missing");
  act(() => {
    cancelUpload.click();
    uploadStarted();
  });
  await act(async () => sending);
  expect(controller.snapshot().draft.pending?.phase).toBe("uploading");
  expect(body(voice)).toBe("Saved message partial");
  voice.close();
});

it("preserves voice text when an upload fails", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const file = new File(["content"], "notes.txt", { type: "text/plain" });
  act(() => controller.addFiles([file]));
  await vi.waitFor(() => expect(controller.snapshot().saving).toBe(false));
  vi.spyOn(backend, "createUpload").mockResolvedValue({
    id: "upload-1",
    state: "reserved",
    expires_at: Date.now() / 1000 + 3600,
  });
  let failUpload!: () => void;
  vi.spyOn(backend, "uploadFile").mockImplementation(
    () =>
      new Promise((_resolve, reject) => {
        failUpload = () => reject(new Error("Upload failed"));
      }),
  );
  const sending = controller.send(async () => true);
  await vi.waitFor(() => expect(failUpload).toBeTypeOf("function"));
  const voice = mount();
  await voice.emit(transcript(" partial"));
  await act(async () => {
    failUpload();
    await sending;
  });
  expect(controller.snapshot().draft.pending?.phase).toBe("uploading");
  expect(body(voice)).toBe("Saved message partial");
  expect(voice.container.textContent).toContain("Upload failed");
  voice.close();
});

it("reconciles a recording started before unknown-send confirmation", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  await controller.send(async () => false);
  const pendingId = controller.snapshot().draft.pending?.id;
  const voice = mount();
  await voice.emit(transcript("Partial"));
  expect(body(voice)).toBe("Saved messagePartial");
  let resolveStatus!: (result: { state: "sent"; message: null }) => void;
  vi.spyOn(backend, "sendStatus").mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveStatus = resolve;
      }),
  );
  const check = Array.from(
    voice.container.querySelectorAll<HTMLButtonElement>("button"),
  ).find((button) => button.textContent === "Check result");
  if (!check) throw new Error("Check result button is missing");
  act(() => check.click());
  await vi.waitFor(() => expect(resolveStatus).toBeTypeOf("function"));
  await act(async () => {
    resolveStatus({ state: "sent", message: null });
    await vi.waitFor(() => {
      expect(controller.snapshot().draft.pending).toBeNull();
      expect(controller.snapshot().busy).toBe(false);
    });
  });
  expect(body(voice)).toBe("Partial");
  expect(controller.snapshot().submissionResult).toEqual({
    id: pendingId,
    state: "sent",
  });
  await voice.emit(transcript("Final"));
  expect(body(voice)).toBe("Final");
  voice.close();
});

it("reconciles a confirmed send after receipt saving fails", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  await controller.send(async () => false);
  const voice = mount();
  await voice.emit(transcript("Partial"));
  const pendingId = controller.snapshot().draft.pending?.id;
  harness.put.mockRejectedValueOnce(new Error("Disk full"));
  vi.spyOn(backend, "sendStatus").mockResolvedValue({
    state: "sent",
    message: null,
  });
  const check = Array.from(
    voice.container.querySelectorAll<HTMLButtonElement>("button"),
  ).find((button) => button.textContent === "Check result");
  if (!check) throw new Error("Check result button is missing");
  await act(async () => {
    check.click();
    await vi.waitFor(() => {
      expect(controller.snapshot().busy).toBe(false);
      expect(controller.snapshot().draft.pending?.id).toBe(pendingId);
    });
  });
  expect(body(voice)).toBe("Partial");
  expect(controller.snapshot().submissionResult).toEqual({
    id: pendingId,
    state: "sent",
  });
  await voice.emit(transcript("Final"));
  expect(body(voice)).toBe("Final");

  const discard = Array.from(
    voice.container.querySelectorAll<HTMLButtonElement>("button"),
  ).find((button) => button.textContent === "Retry");
  if (!discard) throw new Error("Retry button is missing");
  await act(async () => {
    discard.click();
    await vi.waitFor(() => {
      expect(controller.snapshot().busy).toBe(false);
      expect(controller.snapshot().draft.pending).toBeNull();
    });
  });
  expect(body(voice)).toBe("Final");
  voice.close();
});

it("retries send-result confirmation after receipt persistence fails", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  await controller.send(async () => false);
  const voice = mount();
  await voice.emit(transcript("Partial"));
  const pendingId = controller.snapshot().draft.pending?.id;
  harness.put.mockRejectedValueOnce(new Error("Disk full"));
  vi.spyOn(backend, "sendStatus").mockResolvedValue({
    state: "sent",
    message: null,
  });
  const check = () => {
    const button = Array.from(
      voice.container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((item) => item.textContent === "Check result");
    if (!button) throw new Error("Check result button is missing");
    return button;
  };
  await act(async () => {
    check().click();
    await vi.waitFor(() => {
      expect(controller.snapshot().busy).toBe(false);
      expect(controller.snapshot().draft.pending?.id).toBe(pendingId);
    });
  });
  expect(body(voice)).toBe("Partial");
  expect(controller.snapshot().submissionResult).toEqual({
    id: pendingId,
    state: "sent",
  });
  await voice.emit(transcript("Final"));
  expect(body(voice)).toBe("Final");
  const retry = Array.from(
    voice.container.querySelectorAll<HTMLButtonElement>("button"),
  ).find((item) => item.textContent === "Retry");
  if (!retry) throw new Error("Retry button is missing");
  await act(async () => {
    retry.click();
    await vi.waitFor(() => {
      expect(controller.snapshot().busy).toBe(false);
      expect(controller.snapshot().draft.pending).toBeNull();
    });
  });
  expect(body(voice)).toBe("Final");
  voice.close();
});

it("retains voice text through unknown status, reload and cancellation", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  await controller.send(async () => false);
  const pending = controller.snapshot().draft.pending;
  if (!pending) throw new Error("Pending submission is missing");
  const id = pending.id;
  const voice = mount();
  await voice.emit(transcript(" new words"));
  await vi.waitFor(() => expect(controller.snapshot().saving).toBe(false));
  const lastWrite = harness.put.mock.calls[harness.put.mock.calls.length - 1];
  if (!lastWrite) throw new Error("Draft write is missing");
  const saved = structuredClone(lastWrite[1]) as Draft;
  const restored = new DraftController(1, saved);
  vi.spyOn(backend, "sendStatus").mockResolvedValue({
    state: "unknown",
    message: null,
  });
  await restored.checkResult();
  expect(restored.snapshot().draft.pending?.id).toBe(id);
  expect(restored.snapshot().draft.body).toBe("Saved message new words");
  const confirmed = new DraftController(1, saved);
  vi.spyOn(backend, "sendStatus").mockResolvedValue({
    state: "sent",
    message: null,
  });
  await confirmed.checkResult();
  expect(confirmed.snapshot().draft.pending).toBeNull();
  expect(confirmed.snapshot().draft.body).toBe(" new words");
  vi.spyOn(backend, "cancelSend").mockResolvedValue({
    state: "cancelled",
    message: null,
  });
  await restored.discardAttempt();
  expect(backend.cancelSend).toHaveBeenCalledWith(1, id);
  expect(restored.snapshot().draft.pending).toBeNull();
  expect(restored.snapshot().draft.body).toBe("Saved message new words");
});

it("rejects old callbacks after edits, Discussion changes and unmount", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const first = mount();
  await first.emit(transcript(" first"));
  const lateAfterUnmount = harness.callbacks[harness.callbacks.length - 1];
  first.close();
  act(() => lateAfterUnmount(transcript(" late")));
  expect(controller.snapshot().draft.body).toBe("Saved message first");

  const second = mount();
  await second.ready();
  const input = second.container.querySelector<HTMLTextAreaElement>(
    'textarea[aria-label="Message"]',
  );
  if (!input) throw new Error("Composer textarea is missing");
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )?.set;
    setter?.call(input, "User edit");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await vi.waitFor(() => expect(controller.snapshot().saving).toBe(false));
  });
  expect(controller.snapshot().draft.body).toBe("User edit");
  expect(body(second)).toBe("User edit");
  const editedCallback = harness.callbacks[harness.callbacks.length - 1];
  act(() => editedCallback(transcript(" overwrite")));
  expect(controller.snapshot().draft.body).toBe("User edit");

  const other = new DraftController(2, draft("Second Discussion"));
  harness.controllers.set(2, other);
  await second.start();
  const oldDiscussionCallback = harness.callbacks[harness.callbacks.length - 1];
  second.render(2);
  expect(harness.cancel).toHaveBeenCalledTimes(3);
  expect(second.container.textContent).not.toContain("Listening…");
  act(() => oldDiscussionCallback(transcript(" late")));
  expect(body(second)).toBe("Second Discussion");

  await second.start();
  const activeCallback = harness.callbacks[harness.callbacks.length - 1];
  second.close();
  expect(harness.cancel).toHaveBeenCalledTimes(4);
  act(() => activeCallback(transcript(" late")));
  expect(other.snapshot().draft.body).toBe("Second Discussion");
});
