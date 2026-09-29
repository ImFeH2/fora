import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  expect,
  it,
  vi,
} from "vitest";
import type { ThreadData } from "@/features/discussions/thread-data";
import type { DiscussionSummary, Member } from "@/lib/backend";

const harness = vi.hoisted(() => ({
  thread: null as ThreadData | null,
  markRead: vi.fn(),
  invalidate: vi.fn(),
  request: vi.fn(),
  view: vi.fn(),
  positioned: vi.fn(),
}));

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: () => ({
    getVirtualItems: () =>
      (harness.thread?.messages ?? []).map((message, index) => ({
        key: message.id,
        index,
        start: index * 96,
      })),
    getTotalSize: () => (harness.thread?.messages.length ?? 0) * 96,
    measureElement: () => {},
    isAtEnd: () => true,
    scrollToEnd: vi.fn(),
    scrollToOffset: vi.fn(),
    scrollToIndex: vi.fn(),
  }),
}));

vi.mock("@/components/ui/menu", () => ({
  OverflowMenu: ({
    actions,
  }: {
    actions: {
      id: string;
      label: string;
      disabled?: boolean;
      onSelect: () => void;
    }[];
  }) => (
    <div>
      {actions.map((action) => (
        <button
          key={action.id}
          type="button"
          disabled={action.disabled}
          onClick={action.onSelect}
        >
          {action.label}
        </button>
      ))}
    </div>
  ),
}));

vi.mock("@/features/discussions/thread-data", () => ({
  useThreadData: () => ({
    data: harness.thread,
    missing: false,
    failed: false,
    loading: false,
    position: null,
    positioned: harness.positioned,
    request: harness.request,
    markRead: harness.markRead,
    invalidate: harness.invalidate,
    view: harness.view,
    current: { current: harness.thread },
    live: { current: true },
  }),
}));

vi.mock("@/features/discussions/composer", () => ({
  Composer: () => null,
}));

let cleanup: typeof import("@testing-library/react").cleanup;
let fireEvent: typeof import("@testing-library/react").fireEvent;
let render: typeof import("@testing-library/react").render;
let waitFor: typeof import("@testing-library/react").waitFor;
let DiscussionsPage: typeof import("@/features/discussions/list").DiscussionsPage;
let ThreadPage: typeof import("@/features/discussions/thread").ThreadPage;
let OrganizationProvider: typeof import("@/app/organization").OrganizationProvider;
let RouterProvider: typeof import("@/app/router").RouterProvider;
let TooltipProvider: typeof import("@/components/ui/tooltip").TooltipProvider;
let backend: typeof import("@/lib/backend").backend;
let clearToasts: typeof import("@/components/ui/toast").clearToasts;
let readToasts: typeof import("@/components/ui/toast").readToasts;
let closeDom: () => void;

const member: Member = {
  id: 1,
  name: "You",
  type: "human",
  state: "idle",
};
const helper: Member = {
  id: 2,
  name: "Helper",
  type: "agent",
  state: "idle",
};
const discussion: DiscussionSummary = {
  id: 1,
  topic: "Release",
  member_ids: [1, 2],
  archived: false,
  unread: 0,
};

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
  vi.stubGlobal("HTMLElement", dom.window.HTMLElement);
  vi.stubGlobal("Element", dom.window.Element);
  vi.stubGlobal("Node", dom.window.Node);
  vi.stubGlobal("MutationObserver", dom.window.MutationObserver);
  vi.stubGlobal("Event", dom.window.Event);
  vi.stubGlobal("EventTarget", dom.window.EventTarget);
  vi.stubGlobal("CustomEvent", dom.window.CustomEvent);
  vi.stubGlobal("MouseEvent", dom.window.MouseEvent);
  vi.stubGlobal("KeyboardEvent", dom.window.KeyboardEvent);
  vi.stubGlobal("AbortController", dom.window.AbortController);
  vi.stubGlobal("AbortSignal", dom.window.AbortSignal);
  vi.stubGlobal(
    "getComputedStyle",
    dom.window.getComputedStyle.bind(dom.window),
  );
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  vi.stubGlobal("requestAnimationFrame", vi.fn());
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  ({ cleanup, fireEvent, render, waitFor } = await import(
    "@testing-library/react"
  ));
  ({ DiscussionsPage } = await import("@/features/discussions/list"));
  ({ ThreadPage } = await import("@/features/discussions/thread"));
  ({ OrganizationProvider } = await import("@/app/organization"));
  ({ RouterProvider } = await import("@/app/router"));
  ({ TooltipProvider } = await import("@/components/ui/tooltip"));
  ({ backend } = await import("@/lib/backend"));
  ({ clearToasts, readToasts } = await import("@/components/ui/toast"));
}, 30000);

function threadData(overrides: Partial<ThreadData> = {}): ThreadData {
  return {
    id: 1,
    topic: discussion.topic,
    archived: discussion.archived,
    members: [member, helper],
    messages: [
      {
        id: 1,
        sender_id: helper.id,
        sender_name: helper.name,
        body: "@You please review",
        mentions: [{ member_id: member.id, position: 0, length: 4 }],
        attachments: [],
        created_at: "2026-01-01T00:00:00Z",
        pending: true,
        acknowledged: false,
      },
    ],
    divider: null,
    latest: 1,
    readThrough: 0,
    pendingCount: 1,
    hasBefore: false,
    hasAfter: false,
    previousSender: null,
    ...overrides,
  };
}

function organization() {
  return {
    members: [member, helper],
    humanId: member.id,
    discussions: [discussion],
    refresh: vi.fn(async () => {}),
  };
}

function listView() {
  return render(
    <TooltipProvider>
      <RouterProvider>
        <OrganizationProvider value={organization()}>
          <DiscussionsPage />
        </OrganizationProvider>
      </RouterProvider>
    </TooltipProvider>,
  );
}

function threadView() {
  return render(
    <TooltipProvider>
      <RouterProvider>
        <OrganizationProvider value={organization()}>
          <ThreadPage id={discussion.id} />
        </OrganizationProvider>
      </RouterProvider>
    </TooltipProvider>,
  );
}

function button(view: { container: HTMLElement }, name: string) {
  const item = Array.from(view.container.querySelectorAll("button")).find(
    (candidate) => candidate.textContent?.trim() === name,
  );
  if (!item) throw new Error(`${name} button is missing`);
  return item as HTMLButtonElement;
}

function openToast(id: string) {
  return readToasts().find((item) => item.id === id && item.open);
}

function flushScheduledWork() {
  return new Promise<void>((resolve) => setImmediate(resolve));
}

beforeEach(() => {
  harness.thread = threadData();
  harness.markRead.mockReset().mockResolvedValue(undefined);
  harness.invalidate.mockReset().mockResolvedValue(undefined);
  harness.request.mockReset().mockResolvedValue(undefined);
  harness.view.mockReset();
  harness.positioned.mockReset();
  clearToasts();
  vi.spyOn(backend, "onEvent").mockReturnValue(vi.fn());
  vi.spyOn(backend, "discussions").mockResolvedValue([discussion]);
});

afterEach(async () => {
  cleanup();
  await flushScheduledWork();
  clearToasts();
  vi.restoreAllMocks();
});

afterAll(async () => {
  await flushScheduledWork();
  closeDom();
  vi.unstubAllGlobals();
});

it("clears a list archive failure after the original action succeeds", async () => {
  vi.spyOn(backend, "archiveDiscussion")
    .mockRejectedValueOnce(new Error("Archive unavailable"))
    .mockResolvedValueOnce({ id: discussion.id });
  const view = listView();
  await waitFor(() => expect(button(view, "Archive")).toBeDefined());

  fireEvent.click(button(view, "Archive"));
  await waitFor(() =>
    expect(openToast("discussion-archive:1")).toMatchObject({
      title: "Could not archive Discussion",
      description: "Archive unavailable",
    }),
  );

  await waitFor(() => expect(button(view, "Archive").disabled).toBe(false));
  fireEvent.click(button(view, "Archive"));
  await waitFor(() =>
    expect(backend.archiveDiscussion).toHaveBeenCalledTimes(2),
  );
  expect(openToast("discussion-archive:1")).toBeUndefined();
});

it("clears a detail archive failure after the original action succeeds", async () => {
  vi.spyOn(backend, "archiveDiscussion")
    .mockRejectedValueOnce(new Error("Archive unavailable"))
    .mockResolvedValueOnce({ id: discussion.id });
  const view = threadView();
  await waitFor(() => expect(button(view, "Archive")).toBeDefined());

  fireEvent.click(button(view, "Archive"));
  await waitFor(() =>
    expect(
      openToast("discussion-write:1:Could not archive Discussion"),
    ).toMatchObject({
      title: "Could not archive Discussion",
      description: "Archive unavailable",
    }),
  );

  fireEvent.click(button(view, "Archive"));
  await waitFor(() =>
    expect(backend.archiveDiscussion).toHaveBeenCalledTimes(2),
  );
  expect(
    openToast("discussion-write:1:Could not archive Discussion"),
  ).toBeUndefined();
});

it("clears a mark handled failure after the original action succeeds", async () => {
  vi.spyOn(backend, "ack")
    .mockRejectedValueOnce(new Error("Acknowledgement unavailable"))
    .mockResolvedValueOnce({ acked: 1 });
  const view = threadView();
  await waitFor(() => expect(button(view, "Mark handled")).toBeDefined());

  fireEvent.click(button(view, "Mark handled"));
  await waitFor(() =>
    expect(
      openToast("discussion-write:1:Could not mark handled"),
    ).toMatchObject({
      title: "Could not mark handled",
      description: "Acknowledgement unavailable",
    }),
  );

  await waitFor(() =>
    expect(button(view, "Mark handled").disabled).toBe(false),
  );
  fireEvent.click(button(view, "Mark handled"));
  await waitFor(() => expect(backend.ack).toHaveBeenCalledTimes(2));
  expect(
    openToast("discussion-write:1:Could not mark handled"),
  ).toBeUndefined();
});

it("clears an undo failure after the original action succeeds", async () => {
  harness.thread = threadData({
    messages: [
      {
        ...threadData().messages[0],
        pending: false,
        acknowledged: true,
      },
    ],
    pendingCount: 0,
  });
  vi.spyOn(backend, "revokeAck")
    .mockRejectedValueOnce(new Error("Undo unavailable"))
    .mockResolvedValueOnce({ revoked: 1 });
  const view = threadView();
  await waitFor(() => expect(button(view, "Undo")).toBeDefined());

  fireEvent.click(button(view, "Undo"));
  await waitFor(() =>
    expect(openToast("discussion-write:1:Could not undo")).toMatchObject({
      title: "Could not undo",
      description: "Undo unavailable",
    }),
  );

  await waitFor(() => expect(button(view, "Undo").disabled).toBe(false));
  fireEvent.click(button(view, "Undo"));
  await waitFor(() => expect(backend.revokeAck).toHaveBeenCalledTimes(2));
  expect(openToast("discussion-write:1:Could not undo")).toBeUndefined();
});

it("clears a batch handling failure after the original action succeeds", async () => {
  vi.spyOn(backend, "ackPending")
    .mockRejectedValueOnce(new Error("Batch acknowledgement unavailable"))
    .mockResolvedValueOnce({ acked: 1, read_through: 1, pending_count: 0 });
  const view = threadView();
  await waitFor(() => expect(button(view, "Mark all handled")).toBeDefined());

  fireEvent.click(button(view, "Mark all handled"));
  await waitFor(() =>
    expect(
      openToast("discussion-write:1:Could not mark all handled"),
    ).toMatchObject({
      title: "Could not mark all handled",
      description: "Batch acknowledgement unavailable",
    }),
  );

  await waitFor(() =>
    expect(button(view, "Mark all handled").disabled).toBe(false),
  );
  fireEvent.click(button(view, "Mark all handled"));
  await waitFor(() => expect(backend.ackPending).toHaveBeenCalledTimes(2));
  expect(
    openToast("discussion-write:1:Could not mark all handled"),
  ).toBeUndefined();
});
