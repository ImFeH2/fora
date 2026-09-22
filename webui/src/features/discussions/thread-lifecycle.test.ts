import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearToasts, readToasts } from "@/components/ui/toast";
import { useThreadData } from "@/features/discussions/thread-data";
import { type BackendEvent, backend, type DiscussionPage } from "@/lib/backend";

const lifecycle = vi.hoisted(() => ({
  effects: [] as (() => undefined | (() => void))[],
}));
vi.mock("react", async (original) => ({
  ...(await original<typeof import("react")>()),
  useState: (initial: unknown) => [initial, vi.fn()],
  useRef: (initial: unknown) => ({ current: initial }),
  useCallback: (callback: unknown) => callback,
  useEffect: (effect: () => undefined | (() => void)) => {
    lifecycle.effects.push(effect);
  },
}));

const result: DiscussionPage = {
  id: 1,
  messages: [
    {
      id: 1,
      sender_id: 2,
      sender_name: "Helper",
      body: "Unread",
      mentions: [],
      attachments: [],
      created_at: "now",
    },
  ],
  latest_id: 1,
  read_through: 0,
  first_unread_id: 1,
  has_before: false,
  has_after: false,
  previous_sender_id: null,
  awaiting_ack: [],
  acknowledged: [],
  pending_count: 0,
  metadata: { topic: "Thread", archived: false, members: [] },
};
let event: (value: BackendEvent) => void;
let cleanups: (() => void)[];

beforeEach(() => {
  lifecycle.effects = [];
  cleanups = [];
  clearToasts();
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    event = listener;
    return vi.fn();
  });
  vi.spyOn(backend, "discussionPage").mockResolvedValue(result);
  vi.spyOn(backend, "markRead").mockResolvedValue({ read_through: 1 });
});
afterEach(() => {
  for (const cleanup of cleanups) cleanup();
  clearToasts();
  vi.restoreAllMocks();
});
function mount() {
  const hook = useThreadData(1, 1);
  for (const effect of lifecycle.effects.splice(0)) {
    const cleanup = effect();
    if (cleanup) cleanups.push(cleanup);
  }
  return hook;
}
async function settle() {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

describe("thread request lifetimes with hook effects driven explicitly", () => {
  it("fetches entry without reading; only explicit visible reading submits a watermark", async () => {
    const hook = mount();
    await settle();
    expect(backend.discussionPage).toHaveBeenCalledWith(1, {
      entry: true,
      limit: 50,
    });
    expect(backend.markRead).not.toHaveBeenCalled();
    await hook.markRead(1);
    expect(backend.markRead).toHaveBeenCalledWith(1, 1);
    expect(hook.current.current?.readThrough).toBe(1);
  });

  it("discards old responses after the keyed thread session unmounts", async () => {
    const pending = deferred<DiscussionPage>();
    vi.mocked(backend.discussionPage).mockReturnValueOnce(pending.promise);
    const old = mount();
    cleanups[0]();
    pending.resolve(result);
    await settle();
    expect(old.current.current).toBeNull();
    expect(backend.markRead).not.toHaveBeenCalled();
  });

  it("does not toast errors from a thread that has already been left", async () => {
    const pending = deferred<DiscussionPage>();
    vi.mocked(backend.discussionPage).mockReturnValueOnce(pending.promise);
    mount();
    cleanups[0]();
    pending.reject(new Error("old failure"));
    await settle();
    expect(readToasts().filter((item) => item.open)).toEqual([]);
  });

  it("filters other discussions and distinguishes the updated event's id", async () => {
    mount();
    await settle();
    event({ type: "message.created", discussion_id: 2, id: 4 });
    event({ type: "discussion.updated", id: 2 });
    expect(backend.discussionPage).toHaveBeenCalledTimes(1);
    event({ type: "discussion.updated", id: 1 });
    await settle();
    expect(backend.discussionPage).toHaveBeenCalledTimes(2);
  });

  it("preserves newer event metadata across an in-flight initial response", async () => {
    const pending = deferred<DiscussionPage>();
    vi.mocked(backend.discussionPage).mockReturnValueOnce(pending.promise);
    const hook = mount();
    event({ type: "message.created", discussion_id: 1, id: 5 });
    pending.resolve(result);
    await settle();
    expect(hook.current.current?.latest).toBe(5);
    expect(hook.current.current?.divider).toBe(1);
    expect(hook.current.current?.hasAfter).toBe(true);
    expect(backend.markRead).not.toHaveBeenCalled();
  });

  it("leaves failed pages for explicit Toast retry without a hidden retry loop", async () => {
    vi.mocked(backend.discussionPage).mockRejectedValueOnce(
      new Error("page failed"),
    );
    mount();
    await settle();
    expect(backend.discussionPage).toHaveBeenCalledTimes(1);
    const failure = readToasts().find((item) => item.open);
    expect(failure?.duration).toBeNull();
    failure?.action?.onClick();
    await settle();
    expect(backend.discussionPage).toHaveBeenCalledTimes(2);
  });

  it("deduplicates backfill requests, preserves the divider and never reads prefetched history", async () => {
    vi.mocked(backend.discussionPage).mockResolvedValueOnce({
      ...result,
      messages: [{ ...result.messages[0], id: 148 }],
      first_unread_id: 148,
      latest_id: 148,
      read_through: 147,
      has_before: true,
    });
    const hook = mount();
    await settle();
    const pending = deferred<DiscussionPage>();
    vi.mocked(backend.discussionPage).mockReturnValueOnce(pending.promise);
    const first = hook.request("before");
    await hook.request("before");
    expect(backend.discussionPage).toHaveBeenCalledTimes(2);
    expect(backend.discussionPage).toHaveBeenLastCalledWith(1, {
      before: 148,
      limit: 50,
    });
    pending.resolve({ ...result, latest_id: 148, read_through: 147 });
    await first;
    expect(hook.current.current?.messages.map((message) => message.id)).toEqual(
      [1, 148],
    );
    expect(hook.current.current?.divider).toBe(148);
    expect(hook.current.current?.readThrough).toBe(147);
    expect(backend.markRead).not.toHaveBeenCalled();
    await hook.request("before");
    expect(backend.discussionPage).toHaveBeenCalledTimes(2);
  });

  it("discards an in-flight backfill after leaving the thread", async () => {
    vi.mocked(backend.discussionPage).mockResolvedValueOnce({
      ...result,
      has_before: true,
    });
    const hook = mount();
    await settle();
    const snapshot = hook.current.current;
    const pending = deferred<DiscussionPage>();
    vi.mocked(backend.discussionPage).mockReturnValueOnce(pending.promise);
    const request = hook.request("before");
    cleanups[0]();
    pending.resolve({ ...result, messages: [] });
    await request;
    expect(hook.current.current).toBe(snapshot);
    expect(backend.markRead).not.toHaveBeenCalled();
  });

  it("keeps a failed reading submission unconfirmed until explicit retry", async () => {
    vi.mocked(backend.markRead).mockRejectedValueOnce(new Error("read failed"));
    const hook = mount();
    await settle();
    await expect(hook.markRead(1)).rejects.toThrow("read failed");
    expect(hook.current.current?.readThrough).toBe(0);
    await expect(hook.markRead(1)).rejects.toThrow("Retry saving");
    expect(backend.markRead).toHaveBeenCalledTimes(1);
    readToasts()
      .find((item) => item.id.endsWith("-read"))
      ?.action?.onClick();
    await settle();
    expect(hook.current.current?.readThrough).toBe(1);
  });
});
