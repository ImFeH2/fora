import { Children, isValidElement, type ReactNode } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import { Composer } from "@/features/discussions/composer";
import { type Draft, DraftController } from "@/features/discussions/draft";
import { backend } from "@/lib/backend";
import type { VoiceEvent } from "@/lib/voice";

const harness = vi.hoisted(() => ({
  controller: null as DraftController | null,
  effects: [] as (() => (() => void) | undefined)[],
  callbacks: [] as ((event: VoiceEvent) => void)[],
  cancel: vi.fn(),
  put: vi.fn(),
}));
vi.mock("react", async (original) => ({
  ...(await original<typeof import("react")>()),
  useState: (value: unknown) => [value, vi.fn()],
  useRef: (value: unknown) => ({ current: value }),
  useCallback: (value: unknown) => value,
  useMemo: (factory: () => unknown) => factory(),
  useId: () => "voice-test",
  useLayoutEffect: () => {},
  useEffect: (effect: () => (() => void) | undefined) =>
    harness.effects.push(effect),
}));
vi.mock("idb", () => ({
  openDB: async () => ({ put: harness.put, close: () => {} }),
}));
vi.mock("@/features/discussions/draft", async (original) => ({
  ...(await original<typeof import("@/features/discussions/draft")>()),
  useDraft: () => ({
    controller: harness.controller,
    view: harness.controller?.snapshot(),
    error: null,
  }),
}));
vi.mock("@/lib/voice", () => ({
  VoiceRecording: class {
    constructor(callback: (event: VoiceEvent) => void) {
      harness.callbacks.push(callback);
    }
    start = async () => {};
    stop = vi.fn();
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
function button(node: ReactNode, label: string): (() => void) | undefined {
  let found: (() => void) | undefined;
  Children.forEach(node, (child) => {
    if (
      !isValidElement<{
        children?: ReactNode;
        "aria-label"?: string;
        onClick?: () => void;
      }>(child)
    )
      return;
    if (child.props["aria-label"] === label) found = child.props.onClick;
    found ??= button(child.props.children, label);
  });
  return found;
}
function mount() {
  const node = Composer({
    discussionId: 1,
    members: [],
    memberIds: new Set(),
    busy: false,
    placeholder: "Message",
    onSend: async () => true,
    onHeightChange: () => {},
  });
  const cleanups = harness.effects.splice(0).map((effect) => effect());
  const start = button(node, "Start voice input");
  if (!start) throw new Error("Voice input start button is missing");
  start();
  const emit = harness.callbacks[harness.callbacks.length - 1];
  if (!emit) throw new Error("Voice recording callback is missing");
  return {
    emit,
    close: () => {
      cleanups.forEach((cleanup) => {
        cleanup?.();
      });
    },
  };
}
function transcript(text: string): VoiceEvent {
  return { type: "transcript", text };
}

beforeEach(() => {
  vi.restoreAllMocks();
  harness.effects = [];
  harness.callbacks = [];
  harness.cancel.mockReset();
  harness.put.mockReset().mockResolvedValue(undefined);
  harness.controller = new DraftController(1, draft());
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
  voice.emit(transcript(" partial"));
  voice.emit(transcript(" final"));
  expect(controller.snapshot().draft.pending).toEqual(pending);
  finish(true);
  await operation;
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
  finish(true);
  await operation;
  expect(controller.snapshot().draft.body).toBe("");
  voice.emit(transcript("New words"));
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
  voice.emit(transcript("Partial"));
  finish(true);
  await operation;
  voice.emit(transcript("Final"));
  expect(controller.snapshot().draft.body).toBe("Final");
  expect(controller.snapshot().draft.pending).toBeNull();
});

it("reconciles voice text after send-result confirmation", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  await controller.send(async () => false);
  let resolveStatus!: (result: { state: "sent"; message: null }) => void;
  vi.spyOn(backend, "sendStatus").mockImplementation(
    () =>
      new Promise((resolve) => {
        resolveStatus = resolve;
      }),
  );
  const checking = controller.checkResult();
  const voice = mount();
  resolveStatus({ state: "sent", message: null });
  await checking;
  voice.emit(transcript("Final"));
  expect(controller.snapshot().draft.pending).toBeNull();
  expect(controller.snapshot().draft.body).toBe("Final");
});

it("retains voice text through unknown status, reload and cancellation", async () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  await controller.send(async () => false);
  const pending = controller.snapshot().draft.pending;
  if (!pending) throw new Error("Pending submission is missing");
  const id = pending.id;
  const voice = mount();
  voice.emit(transcript(" new words"));
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
  vi.spyOn(backend, "cancelSend").mockResolvedValue({
    state: "cancelled",
    message: null,
  });
  await restored.discardAttempt();
  expect(backend.cancelSend).toHaveBeenCalledWith(1, id);
  expect(restored.snapshot().draft.pending).toBeNull();
  expect(restored.snapshot().draft.body).toBe("Saved message new words");
});

it("rejects callbacks after leaving a Discussion and after user edits", () => {
  const controller = harness.controller;
  if (!controller) throw new Error("Draft controller is missing");
  const first = mount();
  first.emit(transcript(" first"));
  first.close();
  first.emit(transcript(" late"));
  expect(controller.snapshot().draft.body).toBe("Saved message first");
  const second = mount();
  controller.setBody("User edit");
  second.emit(transcript(" overwrite"));
  expect(controller.snapshot().draft.body).toBe("User edit");
  expect(harness.cancel).toHaveBeenCalledTimes(2);
});
