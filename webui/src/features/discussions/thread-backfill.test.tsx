import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThreadPage } from "@/features/discussions/thread";
import { mergePage, type ThreadData } from "@/features/discussions/thread-data";

const harness = vi.hoisted(() => ({
  effects: [] as (() => undefined | (() => void))[],
  frames: [] as FrameRequestCallback[],
  resize: () => {},
  root: null as unknown as HTMLDivElement,
  total: 200,
  data: null as ThreadData | null,
  failed: false,
  loading: false,
  request: vi.fn(),
  read: vi.fn().mockResolvedValue(undefined),
  position: vi.fn(),
  offset: vi.fn((offset: number) => {
    harness.root.scrollTop = offset;
  }),
  positioned: vi.fn(),
  markerTop: 300,
  layoutChanges: [] as number[],
  visibleMessages: [] as HTMLElement[],
  disconnect: vi.fn(),
}));
vi.mock("react", async (original) => ({
  ...(await original<typeof import("react")>()),
  useRef: (value: unknown) => ({
    current: value === null ? harness.root : value,
  }),
  useEffect: (effect: () => undefined | (() => void)) =>
    harness.effects.push(effect),
  useLayoutEffect: (effect: () => undefined | (() => void)) =>
    harness.effects.push(effect),
}));
vi.mock("@/app/organization", () => ({
  useOrganization: () => ({
    members: [],
    humanId: 1,
    discussions: [],
    refresh: vi.fn(),
  }),
}));
vi.mock("@/app/router", () => ({ useNavigate: () => vi.fn() }));
vi.mock("@/features/discussions/composer", () => ({ Composer: () => null }));
vi.mock("@/components/ui/tooltip", () => ({ Tooltip: () => null }));
vi.mock("@/components/ui/menu", () => ({ OverflowMenu: () => null }));
vi.mock("@/features/discussions/thread-data", async (original) => ({
  ...(await original<typeof import("@/features/discussions/thread-data")>()),
  useThreadData: () => ({
    data: harness.data,
    current: {
      get current() {
        return harness.data;
      },
    },
    get failed() {
      return harness.failed;
    },
    get loading() {
      return harness.loading;
    },
    position: "entry",
    positioned: harness.positioned,
    request: harness.request,
    markRead: harness.read,
    view: vi.fn(),
  }),
}));
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: () => ({
    getVirtualItems: () => [],
    getTotalSize: () => harness.total,
    isAtEnd: () => true,
    scrollToIndex: harness.position,
    scrollToOffset: harness.offset,
    scrollToEnd: harness.position,
  }),
}));

function data(count = 3, hasBefore = true, unread: number | null = 148) {
  return mergePage(
    null,
    {
      id: 1,
      messages: Array.from({ length: count }, (_, index) => ({
        id: 151 - count + index,
        sender_id: 2,
        sender_name: "Helper",
        body: "Short",
        mentions: [],
        attachments: [],
        created_at: "now",
      })),
      latest_id: count ? 150 : 0,
      read_through: 147,
      first_unread_id: unread,
      has_before: hasBefore,
      has_after: false,
      previous_sender_id: 2,
      awaiting_ack: [],
      acknowledged: [],
      pending_count: 0,
      metadata: { topic: "Thread", archived: false, members: [] },
    },
    "entry",
  );
}

let cleanups: (() => void)[];
beforeEach(() => {
  vi.clearAllMocks();
  harness.effects = [];
  harness.frames = [];
  harness.failed = false;
  harness.loading = false;
  harness.total = 200;
  harness.data = data();
  harness.markerTop = 300;
  harness.layoutChanges = [];
  harness.visibleMessages = [];
  harness.root = {
    clientHeight: 600,
    scrollHeight: 2000,
    scrollTop: 0,
    getBoundingClientRect: () => ({ top: 0, bottom: 600 }),
    querySelector: () => ({
      getBoundingClientRect: () => ({
        top: harness.markerTop - harness.root.scrollTop,
      }),
    }),
    querySelectorAll: () => harness.visibleMessages,
  } as unknown as HTMLDivElement;
  cleanups = [];
  vi.stubGlobal("document", {
    visibilityState: "visible",
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  });
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) =>
    harness.frames.push(callback),
  );
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(callback: () => void) {
        harness.resize = callback;
      }
      observe = vi.fn();
      disconnect = harness.disconnect;
    },
  );
});
afterEach(() => {
  for (const cleanup of cleanups) cleanup();
  vi.unstubAllGlobals();
});
function mount() {
  renderToStaticMarkup(<ThreadPage id={1} />);
  for (const effect of harness.effects) {
    const cleanup = effect();
    if (cleanup) cleanups.push(cleanup);
  }
}
function nextFrame(timestamp: number) {
  harness.markerTop += harness.layoutChanges.shift() ?? 0;
  for (const callback of harness.frames.splice(0)) callback(timestamp);
}
function frame() {
  for (let index = 0; harness.frames.length && index < 10; index++)
    nextFrame(index);
}

describe("short thread backfill sampling", () => {
  it.each([1, 3])(
    "backfills %s unread rows after positioning without reading prefetched rows",
    (count) => {
      harness.data = data(count, true, 151 - count);
      mount();
      harness.resize();
      expect(harness.request).not.toHaveBeenCalled();
      frame();
      expect(harness.position).toHaveBeenCalledWith(0, { align: "center" });
      expect(harness.offset).toHaveBeenLastCalledWith(32);
      expect(harness.root.scrollTop).toBe(32);
      expect(harness.request).toHaveBeenCalledWith("before");
      expect(harness.read).not.toHaveBeenCalled();
    },
  );

  it("waits for two stable frames after three changing layout frames", () => {
    harness.layoutChanges = [20, 20, 20, 0, 0];
    harness.visibleMessages = [
      {
        dataset: { messageId: "148" },
        getBoundingClientRect: () => ({ top: 100, bottom: 200 }),
      } as unknown as HTMLElement,
    ];
    mount();

    nextFrame(0);
    nextFrame(1);
    nextFrame(2);
    nextFrame(3);
    expect(harness.positioned).not.toHaveBeenCalled();
    expect(harness.read).not.toHaveBeenCalled();

    nextFrame(4);
    expect(harness.positioned).toHaveBeenCalledOnce();
    expect(harness.read).toHaveBeenCalledWith(148);
  });

  it("waits for a readable viewport before sampling the entry page", () => {
    Object.defineProperty(harness.root, "clientHeight", {
      value: 0,
      configurable: true,
    });
    mount();
    frame();
    expect(harness.request).not.toHaveBeenCalled();
    expect(harness.read).not.toHaveBeenCalled();
    Object.defineProperty(harness.root, "clientHeight", {
      value: 600,
      configurable: true,
    });
    frame();
    expect(harness.request).toHaveBeenCalledWith("before");
    expect(harness.read).not.toHaveBeenCalled();
  });

  it("backfills a short no-unread entry positioned at the end", () => {
    harness.data = data(3, true, null);
    mount();
    frame();
    expect(harness.position).toHaveBeenCalledWith();
    expect(harness.request).toHaveBeenCalledWith("before");
  });

  it.each([0, 3])("stops with %s rows when history is exhausted", (count) => {
    harness.data = data(count, false, null);
    mount();
    frame();
    expect(harness.request).not.toHaveBeenCalled();
  });

  it("uses total virtual height rather than the empty rendered subset", () => {
    harness.total = 2000;
    mount();
    frame();
    expect(harness.request).not.toHaveBeenCalled();
  });

  it("continues short batches and stops once scrollable or exhausted", () => {
    mount();
    frame();
    harness.total = 400;
    harness.resize();
    expect(harness.request).toHaveBeenCalledTimes(2);
    harness.total = 601;
    harness.resize();
    expect(harness.request).toHaveBeenCalledTimes(2);
    harness.total = 400;
    harness.data = data(3, false);
    harness.resize();
    expect(harness.request).toHaveBeenCalledTimes(2);
  });

  it("reacts to viewport growth and exact fit but ignores zero-height layout", () => {
    harness.total = 700;
    mount();
    frame();
    expect(harness.request).not.toHaveBeenCalled();
    Object.defineProperty(harness.root, "clientHeight", {
      value: 700,
      configurable: true,
    });
    harness.resize();
    expect(harness.request).toHaveBeenCalledTimes(1);
    Object.defineProperty(harness.root, "clientHeight", { value: 0 });
    harness.total = 0;
    harness.resize();
    expect(harness.request).toHaveBeenCalledTimes(1);
  });

  it("does not retry failed pages or request during a load", () => {
    harness.loading = true;
    mount();
    frame();
    expect(harness.request).not.toHaveBeenCalled();
    harness.loading = false;
    harness.failed = true;
    harness.resize();
    harness.resize();
    expect(harness.request).not.toHaveBeenCalled();
    harness.failed = false;
    harness.resize();
    expect(harness.request).toHaveBeenCalledTimes(1);
  });

  it("disconnects the resize observer when the session leaves", () => {
    mount();
    for (const cleanup of cleanups.splice(0)) cleanup();
    expect(harness.disconnect).toHaveBeenCalledTimes(1);
  });
});
