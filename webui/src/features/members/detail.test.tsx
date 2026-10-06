import { type ReactNode, StrictMode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { OrganizationProvider } from "@/app/organization";
import { RouterProvider } from "@/app/router";
import { ConfirmDialog } from "@/components/ui/dialog";
import { Avatar } from "@/components/ui/index";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TreeView } from "@/features/library/tree-view";
import {
  AgentDetailStatus,
  HistorySection,
  MemberPage,
  WorkspaceContent,
  WorkspaceSection,
} from "@/features/members/detail";
import * as saver from "@/features/settings/saver";
import {
  type AgentDetail,
  type AgentHistoryRead,
  type AgentHistoryRequestSummary,
  type AgentRun,
  backend,
  type HistoryBinary,
  type HistoryImage,
  type HistoryText,
  type LibraryEntry,
} from "@/lib/backend";
import { formatTime } from "@/lib/format";

vi.mock("@/features/settings/model", () => ({
  AgentModelPanel: () => null,
}));

vi.mock("@/components/ui/dialog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/ui/dialog")>()),
  ConfirmDialog: vi.fn(() => null),
  Modal: ({
    children,
    footer,
    open,
  }: {
    children?: ReactNode;
    footer?: ReactNode;
    open: boolean;
  }) =>
    open ? (
      <div>
        {children}
        {footer}
      </div>
    ) : null,
}));

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

const detail: AgentDetail = {
  id: 2,
  workspace: [],
  runs: [],
  usage: {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    requests: 0,
    total_tokens: 0,
  },
  token_limit: 0,
  over_token_limit: false,
  idle: false,
  idle_streak: 0,
  no_tool_streak: 0,
  pause_reason: null,
  window: { number: 1, since_sequence: 0, reset_at: null, reason: null },
};
const entries: LibraryEntry[] = [
  {
    path: "notes",
    kind: "directory",
    size: 0,
    modified_at: "2026-01-01T00:00:00Z",
  },
  {
    path: "notes/MEMORY.md",
    kind: "file",
    size: 5,
    modified_at: "2026-01-01T00:00:00Z",
  },
];

function status(value: AgentDetail) {
  return renderToStaticMarkup(
    <TooltipProvider>
      <AgentDetailStatus detail={value} />
    </TooltipProvider>,
  );
}

describe("Agent deletion", () => {
  it("describes leaving Discussions and keeps the delete action", async () => {
    const refresh = vi.fn(async () => {});
    const remove = vi
      .spyOn(backend, "deleteAgent")
      .mockResolvedValue({ id: 2 });
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <RouterProvider>
          <OrganizationProvider
            value={{
              members: [
                { id: 2, name: "Helper", type: "agent", state: "idle" },
              ],
              humanId: 1,
              discussions: [],
              refresh,
            }}
          >
            <MemberPage id={2} />
          </OrganizationProvider>
        </RouterProvider>
      </TooltipProvider>,
    );
    expect(html).toContain(
      renderToStaticMarkup(<Avatar memberId={2} size="lg" />),
    );
    const confirmation = vi.mocked(ConfirmDialog).mock.calls[0][0];
    expect(confirmation.title).toBe("Delete Helper?");
    expect(confirmation.description).toBe(
      "It leaves every Discussion and stops running.",
    );
    expect(confirmation.confirmLabel).toBe("Delete Agent");
    await confirmation.onConfirm();
    expect(remove).toHaveBeenCalledExactlyOnceWith(2);
    expect(refresh).toHaveBeenCalledOnce();
  });
});

describe("Agent detail status", () => {
  it("shows pause intent while the current Turn completes", () => {
    expect(
      status({ ...detail, state: "running", pause_requested: true }),
    ).toContain("Pause requested · Current Turn will finish");
    expect(
      status({ ...detail, state: "paused", pause_requested: true }),
    ).not.toContain("Current Turn will finish");
    expect(status(detail)).not.toContain("Pause requested");
  });
  it("uses the kernel idle flag even below the previous threshold", () => {
    expect(status({ ...detail, idle: true, idle_streak: 1 })).toContain(
      "1 idle Turn",
    );
    expect(status({ ...detail, idle: false, idle_streak: 20 })).not.toContain(
      "idle Turn",
    );
    expect(status({ ...detail, idle: true, idle_streak: 1000 })).toContain(
      "1,000 idle Turns",
    );
  });

  it("shows later windows but not the initial window", () => {
    expect(status(detail)).not.toContain("Window");
    const html = status({
      ...detail,
      window: {
        number: 1200,
        since_sequence: 10,
        reset_at: "2026-01-01T00:00:00Z",
        reason: "budget",
      },
    });
    expect(html).toContain("Window 1,200");
    expect(html).toContain("<button");
    expect(html).not.toContain("title=");
  });
});

describe("Agent history", () => {
  it("shows the complete Turn entry and request count", () => {
    const run: AgentRun = {
      sequence: 4,
      run_id: "run-4",
      status: "failed",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: "2026-01-01T00:01:00Z",
      last_saved_at: "2026-01-01T00:00:30Z",
      window_number: 2,
      request_count: 2,
      usage: null,
      error: "Provider failed",
      effects: [],
    };
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <HistorySection agentId={2} initialRuns={[run]} />
      </TooltipProvider>,
    );
    expect(html).toContain("History");
    expect(html).toContain("2 model requests");
    expect(html).toContain("View full context");
    expect(html).toContain("Provider failed");
  });
});

describe("Agent history request fields", () => {
  let act: typeof import("@testing-library/react").act;
  let cleanup: typeof import("@testing-library/react").cleanup;
  let fireEvent: typeof import("@testing-library/react").fireEvent;
  let render: typeof import("@testing-library/react").render;
  let screen: typeof import("@testing-library/react").screen;
  let within: typeof import("@testing-library/react").within;
  let closeDom: (() => void) | undefined;

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
    vi.stubGlobal("Node", dom.window.Node);
    vi.stubGlobal("MutationObserver", dom.window.MutationObserver);
    vi.stubGlobal("Event", dom.window.Event);
    vi.stubGlobal("EventTarget", dom.window.EventTarget);
    vi.stubGlobal("CustomEvent", dom.window.CustomEvent);
    vi.stubGlobal("MouseEvent", dom.window.MouseEvent);
    vi.stubGlobal("KeyboardEvent", dom.window.KeyboardEvent);
    vi.stubGlobal("AbortController", dom.window.AbortController);
    vi.stubGlobal("AbortSignal", dom.window.AbortSignal);
    vi.stubGlobal("getComputedStyle", dom.window.getComputedStyle);
    dom.window.requestAnimationFrame = vi.fn();
    dom.window.cancelAnimationFrame = vi.fn();
    vi.stubGlobal("requestAnimationFrame", dom.window.requestAnimationFrame);
    vi.stubGlobal("cancelAnimationFrame", dom.window.cancelAnimationFrame);
    const testing = await import("@testing-library/react");
    act = testing.act;
    cleanup = testing.cleanup;
    fireEvent = testing.fireEvent;
    render = testing.render;
    screen = testing.screen;
    within = testing.within;
  }, 30000);

  afterEach(() => cleanup());

  afterAll(() => {
    closeDom?.();
    vi.unstubAllGlobals();
  });

  function page(id: number, strict = false) {
    const content = (
      <TooltipProvider>
        <RouterProvider>
          <OrganizationProvider
            value={{
              members: [
                { id: 2, name: "Alpha", type: "agent", state: "idle" },
                { id: 3, name: "Beta", type: "agent", state: "idle" },
              ],
              humanId: 1,
              discussions: [],
              refresh: async () => {},
            }}
          >
            <MemberPage id={id} />
          </OrganizationProvider>
        </RouterProvider>
      </TooltipProvider>
    );
    return strict ? <StrictMode>{content}</StrictMode> : content;
  }

  function pendingDetail() {
    let resolve!: (value: AgentDetail) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<AgentDetail>((yes, no) => {
      resolve = yes;
      reject = no;
    });
    return { promise, resolve, reject };
  }

  function events() {
    const listeners = new Set<Parameters<typeof backend.onEvent>[0]>();
    const stops = vi.fn();
    vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
        stops();
      };
    });
    return {
      listeners,
      stops,
      emit: (event: Parameters<Parameters<typeof backend.onEvent>[0]>[0]) => {
        for (const listener of listeners) listener(event);
      },
    };
  }

  function tokens(value: number, id = 2): AgentDetail {
    return {
      ...detail,
      id,
      usage: { ...detail.usage, total_tokens: value },
    };
  }

  it("filters unrelated events and merges bursts while publishing every completed read", async () => {
    const stream = events();
    const first = pendingDetail();
    const second = pendingDetail();
    const third = pendingDetail();
    const read = vi
      .spyOn(backend, "agentDetail")
      .mockImplementationOnce(() => first.promise)
      .mockImplementationOnce(() => second.promise)
      .mockImplementationOnce(() => third.promise);
    render(page(2));
    await act(async () => {});
    expect(read).toHaveBeenCalledExactlyOnceWith(2);
    await act(async () => {
      for (let index = 0; index < 20; index += 1)
        stream.emit({ type: "turn.progress", agent_id: 3 });
    });
    expect(read).toHaveBeenCalledTimes(1);
    await act(async () => {
      for (let index = 0; index < 20; index += 1)
        stream.emit({ type: "turn.progress", agent_id: 2 });
      stream.emit({ type: "settings.updated" });
      stream.emit({ type: "organization.changed" });
    });
    expect(read).toHaveBeenCalledTimes(1);
    await act(async () => first.resolve(tokens(101)));
    expect(screen.getByText("101")).toBeTruthy();
    expect(read).toHaveBeenCalledTimes(2);
    await act(async () => {
      for (let index = 0; index < 20; index += 1)
        stream.emit({ type: "turn.progress", agent_id: 2 });
      second.resolve(tokens(202));
    });
    expect(screen.getByText("202")).toBeTruthy();
    expect(read).toHaveBeenCalledTimes(3);
    const finalRun: AgentRun = {
      sequence: 4,
      run_id: "run-4",
      status: "failed",
      started_at: "2026-10-06T13:00:00Z",
      completed_at: "2026-10-06T13:02:00Z",
      last_saved_at: "2026-10-06T13:01:59Z",
      usage: null,
      error: "Final Turn error",
      effects: [],
    };
    await act(async () => third.resolve({ ...tokens(303), runs: [finalRun] }));
    expect(screen.getByText("303")).toBeTruthy();
    expect(screen.getByText("Final Turn error")).toBeTruthy();
    expect(screen.getByText("failed")).toBeTruthy();
    expect(
      screen.getByText(`Saved ${formatTime(finalRun.last_saved_at as string)}`),
    ).toBeTruthy();
    expect(
      screen.getByText(
        `Finished ${formatTime(finalRun.completed_at as string)}`,
      ),
    ).toBeTruthy();
    expect(read).toHaveBeenCalledTimes(3);
    await act(async () => stream.emit({ type: "turn.completed", agent_id: 3 }));
    expect(read).toHaveBeenCalledTimes(3);
  });

  it.each([false, true])(
    "isolates late results after switching Agent (failure=%s)",
    async (failure) => {
      const stream = events();
      const old = pendingDetail();
      const current = pendingDetail();
      const report = vi
        .spyOn(saver, "reportLoadFailure")
        .mockImplementation(() => {});
      const read = vi
        .spyOn(backend, "agentDetail")
        .mockResolvedValueOnce(tokens(101))
        .mockImplementationOnce(() => old.promise)
        .mockImplementationOnce(() => current.promise);
      const view = render(page(2));
      await act(async () => {});
      expect(screen.getByText("101")).toBeTruthy();
      await act(async () =>
        stream.emit({ type: "turn.progress", agent_id: 2 }),
      );
      view.rerender(page(3));
      await act(async () => {});
      expect(screen.queryByText("101")).toBeNull();
      expect(read.mock.calls.map(([id]) => id)).toEqual([2, 2, 3]);
      await act(async () => {
        if (failure) old.reject(new Error("stale Agent failure"));
        else old.resolve(tokens(999));
        current.resolve(tokens(303, 3));
      });
      expect(screen.getByText("303")).toBeTruthy();
      expect(screen.queryByText("999")).toBeNull();
      expect(report).not.toHaveBeenCalled();
      expect(stream.listeners.size).toBe(1);
    },
  );

  it("recovers an initial failure through Retry and displays empty history", async () => {
    events();
    let retry: (() => void) | undefined;
    vi.spyOn(saver, "reportLoadFailure").mockImplementation(
      (_id, _error, action) => {
        retry = action;
      },
    );
    const read = vi
      .spyOn(backend, "agentDetail")
      .mockRejectedValueOnce(new Error("initial failure"))
      .mockResolvedValueOnce(tokens(101));
    render(page(2));
    await act(async () => {});
    expect(screen.queryByText("No Turns yet")).toBeNull();
    expect(retry).toBeDefined();
    await act(async () => retry?.());
    expect(read).toHaveBeenCalledTimes(2);
    expect(screen.getByText("101")).toBeTruthy();
    expect(screen.getByText("No Turns yet")).toBeTruthy();
  });

  it("retains content on failure and merges Retry through the same refresh lifecycle", async () => {
    const stream = events();
    const recovery = pendingDetail();
    let retry: (() => void) | undefined;
    const report = vi
      .spyOn(saver, "reportLoadFailure")
      .mockImplementation((_id, _error, action) => {
        retry = action;
      });
    const read = vi
      .spyOn(backend, "agentDetail")
      .mockResolvedValueOnce(tokens(101))
      .mockRejectedValueOnce(new Error("read failed"))
      .mockImplementationOnce(() => recovery.promise)
      .mockResolvedValueOnce(tokens(303));
    render(page(2));
    await act(async () => {});
    await act(async () => stream.emit({ type: "turn.failed", agent_id: 2 }));
    expect(screen.getByText("101")).toBeTruthy();
    expect(report).toHaveBeenCalledWith(
      "agent-load:2",
      expect.any(Error),
      expect.any(Function),
    );
    await act(async () => {
      retry?.();
      retry?.();
      stream.emit({ type: "turn.progress", agent_id: 2 });
    });
    expect(read).toHaveBeenCalledTimes(3);
    await act(async () => recovery.resolve(tokens(202)));
    expect(screen.getByText("303")).toBeTruthy();
    expect(read).toHaveBeenCalledTimes(4);
  });

  it.each([false, true])(
    "isolates the old connection during reconnect (failure=%s)",
    async (failure) => {
      const stream = events();
      const old = pendingDetail();
      const current = pendingDetail();
      const followup = pendingDetail();
      const report = vi
        .spyOn(saver, "reportLoadFailure")
        .mockImplementation(() => {});
      const read = vi
        .spyOn(backend, "agentDetail")
        .mockImplementationOnce(() => old.promise)
        .mockImplementationOnce(() => current.promise)
        .mockImplementationOnce(() => followup.promise);
      render(page(2));
      await act(async () => {});
      await act(async () => {
        stream.emit({ type: "connection.closed" });
        stream.emit({ type: "turn.progress", agent_id: 2 });
      });
      expect(read).toHaveBeenCalledTimes(1);
      await act(async () => stream.emit({ type: "connection.restored" }));
      expect(read).toHaveBeenCalledTimes(2);
      await act(async () => {
        stream.emit({ type: "turn.progress", agent_id: 2 });
        if (failure) old.reject(new Error("old connection failure"));
        else old.resolve(tokens(999));
      });
      expect(read).toHaveBeenCalledTimes(2);
      expect(screen.queryByText("999")).toBeNull();
      expect(report).not.toHaveBeenCalled();
      await act(async () => current.resolve(tokens(202)));
      expect(screen.getByText("202")).toBeTruthy();
      expect(read).toHaveBeenCalledTimes(3);
      await act(async () => followup.resolve(tokens(303)));
      expect(screen.getByText("303")).toBeTruthy();
    },
  );

  it.each([false, true])(
    "cleans StrictMode subscriptions and ignores completion after unmount (failure=%s)",
    async (failure) => {
      const stream = events();
      const pending = pendingDetail();
      const report = vi
        .spyOn(saver, "reportLoadFailure")
        .mockImplementation(() => {});
      const read = vi
        .spyOn(backend, "agentDetail")
        .mockImplementation(() => pending.promise);
      const view = render(page(2, true));
      await act(async () => {});
      expect(read).toHaveBeenCalledExactlyOnceWith(2);
      expect(stream.listeners.size).toBe(1);
      view.unmount();
      expect(stream.listeners.size).toBe(0);
      expect(stream.stops).toHaveBeenCalledTimes(2);
      await act(async () => {
        if (failure) pending.reject(new Error("unmounted failure"));
        else pending.resolve(tokens(999));
      });
      expect(report).not.toHaveBeenCalled();
      expect(read).toHaveBeenCalledTimes(1);
    },
  );

  it("resets long request fields before continuing the selected request", async () => {
    const summary = (ordinal: number): AgentHistoryRequestSummary => ({
      ordinal,
      request_id: `request-${ordinal}`,
      run_id: "run-4",
      window_number: 2,
      status: "completed",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: "2026-01-01T00:01:00Z",
      streaming: false,
      input_length: 0,
      parameters_length: 10,
      settings_length: 10,
      model_length: 10,
      response_length: 0,
      input_count: 0,
      response_count: 0,
      related_count: 0,
      error: null,
    });
    const field = (
      source: HistoryText["source"],
      ordinal: number,
      value: string,
      hasMore: boolean,
    ): HistoryText => ({
      kind: "text",
      source,
      request_ordinal: ordinal,
      message_index: 0,
      path: ["value"],
      offset: 0,
      next_offset: value.length,
      total_bytes: value.length + (hasMore ? 5 : 0),
      value,
      has_more: hasMore,
    });
    const emptyMessages = {
      messages: [],
      offset: 0,
      total: 0,
      has_more: false,
    };
    const read = (ordinal: number): AgentHistoryRead => ({
      agent_id: 2,
      run: {
        sequence: 4,
        run_id: "run-4",
        status: "failed",
        started_at: "2026-01-01T00:00:00Z",
        completed_at: "2026-01-01T00:01:00Z",
        last_saved_at: "2026-01-01T00:00:30Z",
        window_number: 2,
        window_reset_at: null,
        window_reason: null,
        request_count: 2,
        usage: null,
        error: null,
        legacy: false,
      },
      windows: [],
      windows_has_more: false,
      windows_next_after: null,
      requests: [summary(1), summary(2)],
      requests_has_more: false,
      requests_next_after: null,
      messages: emptyMessages,
      missing: [],
      request: {
        summary: summary(ordinal),
        input: emptyMessages,
        parameters: field(
          "parameters",
          ordinal,
          `request-${ordinal}-parameters`,
          true,
        ),
        settings: field(
          "settings",
          ordinal,
          `request-${ordinal}-settings`,
          false,
        ),
        model: field("model", ordinal, `request-${ordinal}-model`, false),
        response: null,
        related: emptyMessages,
      },
    });
    vi.spyOn(backend, "agentHistoryRead").mockImplementation(
      async (_agentId, _sequence, ordinal) => read(ordinal ?? 1),
    );
    const textRead = vi
      .spyOn(backend, "agentHistoryText")
      .mockImplementation(async (_agentId, _sequence, reference) => ({
        offset: reference.offset,
        next_offset: reference.offset + 5,
        total_bytes: reference.offset + 5,
        value: `request-${reference.request_ordinal}-tail`,
        has_more: false,
      }));

    render(
      <TooltipProvider>
        <HistorySection
          agentId={2}
          initialRuns={[
            {
              sequence: 4,
              run_id: "run-4",
              status: "failed",
              started_at: "2026-01-01T00:00:00Z",
              completed_at: "2026-01-01T00:01:00Z",
              last_saved_at: "2026-01-01T00:00:30Z",
              window_number: 2,
              request_count: 2,
              usage: null,
              error: null,
              effects: [],
            },
          ]}
        />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "View full context" }));
    expect(await screen.findByText("request-1-parameters")).toBeTruthy();

    const parametersPanel = () => {
      const heading = screen.getByRole("heading", {
        name: "Model request parameters",
      });
      const section = heading.closest("section");
      if (!(section instanceof HTMLElement))
        throw new Error("Request parameters section is missing");
      return within(section);
    };

    fireEvent.click(
      parametersPanel().getByRole("button", { name: "Load more text" }),
    );
    expect(await screen.findByText(/request-1-tail/)).toBeTruthy();

    fireEvent.click(screen.getByRole("tab", { name: /Request 2/ }));
    expect(await screen.findByText("request-2-parameters")).toBeTruthy();
    expect(await screen.findByText("request-2-settings")).toBeTruthy();
    expect(await screen.findByText("request-2-model")).toBeTruthy();
    expect(screen.queryByText(/request-1-tail/)).toBeNull();
    expect(screen.queryByText("request-1-parameters")).toBeNull();
    expect(screen.queryByText("request-1-settings")).toBeNull();
    expect(screen.queryByText("request-1-model")).toBeNull();

    fireEvent.click(
      parametersPanel().getByRole("button", { name: "Load more text" }),
    );
    expect(await screen.findByText(/request-2-tail/)).toBeTruthy();
    expect(screen.queryByText(/request-1-tail/)).toBeNull();
    expect(textRead).toHaveBeenLastCalledWith(
      2,
      4,
      expect.objectContaining({
        request_ordinal: 2,
        offset: expect.any(Number),
      }),
    );
  });

  it("resets long responses and images when changing requests", async () => {
    const summary = (ordinal: number): AgentHistoryRequestSummary => ({
      ordinal,
      request_id: `request-${ordinal}`,
      run_id: "run-4",
      window_number: 2,
      status: "completed",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: "2026-01-01T00:01:00Z",
      streaming: false,
      input_length: 0,
      parameters_length: 10,
      settings_length: 10,
      model_length: 10,
      response_length: 20,
      input_count: 0,
      response_count: 2,
      related_count: 0,
      error: null,
    });
    const field = (
      source: HistoryText["source"],
      ordinal: number,
      value: string,
      hasMore: boolean,
    ): HistoryText => ({
      kind: "text",
      source,
      request_ordinal: ordinal,
      message_index: 0,
      path: ["value"],
      offset: 0,
      next_offset: value.length,
      total_bytes: value.length + (hasMore ? 5 : 0),
      value,
      has_more: hasMore,
    });
    const image = (ordinal: number): HistoryBinary => ({
      kind: "binary",
      source: "response",
      request_ordinal: ordinal,
      message_index: 1,
      path: ["image"],
      media_type: "image/png",
      size: 3,
      identifier: `image-${ordinal}`,
    });
    const emptyMessages = {
      messages: [],
      offset: 0,
      total: 0,
      has_more: false,
    };
    const read = (ordinal: number): AgentHistoryRead => ({
      agent_id: 2,
      run: {
        sequence: 4,
        run_id: "run-4",
        status: "failed",
        started_at: "2026-01-01T00:00:00Z",
        completed_at: "2026-01-01T00:01:00Z",
        last_saved_at: "2026-01-01T00:00:30Z",
        window_number: 2,
        window_reset_at: null,
        window_reason: null,
        request_count: 2,
        usage: null,
        error: null,
        legacy: false,
      },
      windows: [],
      windows_has_more: false,
      windows_next_after: null,
      requests: [summary(1), summary(2)],
      requests_has_more: false,
      requests_next_after: null,
      messages: emptyMessages,
      missing: [],
      request: {
        summary: summary(ordinal),
        input: emptyMessages,
        parameters: field(
          "parameters",
          ordinal,
          `request-${ordinal}-parameters`,
          false,
        ),
        settings: field(
          "settings",
          ordinal,
          `request-${ordinal}-settings`,
          false,
        ),
        model: field("model", ordinal, `request-${ordinal}-model`, false),
        response: {
          messages: [
            field("response", ordinal, `request-${ordinal}-response`, true),
            image(ordinal),
          ],
          offset: 0,
          total: 2,
          has_more: false,
        },
        related: emptyMessages,
      },
    });
    vi.spyOn(backend, "agentHistoryRead").mockImplementation(
      async (_agentId, _sequence, ordinal) => read(ordinal ?? 1),
    );
    const textRead = vi
      .spyOn(backend, "agentHistoryText")
      .mockImplementation(async (_agentId, _sequence, reference) => ({
        offset: reference.offset,
        next_offset: reference.offset + 5,
        total_bytes: reference.offset + 5,
        value: `request-${reference.request_ordinal}-response-tail`,
        has_more: false,
      }));
    let resolveSecondImage!: (value: HistoryImage) => void;
    const secondImage = new Promise<HistoryImage>((resolve) => {
      resolveSecondImage = resolve;
    });
    vi.spyOn(backend, "agentHistoryImage").mockImplementation(
      async (_agentId, _sequence, reference) => {
        if (reference.request_ordinal === 2) return secondImage;
        return {
          media_type: "image/png",
          size: 3,
          identifier: "image-1",
          data: "image-one",
        };
      },
    );

    render(
      <TooltipProvider>
        <HistorySection
          agentId={2}
          initialRuns={[
            {
              sequence: 4,
              run_id: "run-4",
              status: "failed",
              started_at: "2026-01-01T00:00:00Z",
              completed_at: "2026-01-01T00:01:00Z",
              last_saved_at: "2026-01-01T00:00:30Z",
              window_number: 2,
              request_count: 2,
              usage: null,
              error: null,
              effects: [],
            },
          ]}
        />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "View full context" }));
    expect(await screen.findByText("request-1-response")).toBeTruthy();
    expect(await screen.findByAltText("Saved model content")).toBeTruthy();

    const responsePanel = () => {
      const heading = screen.getByRole("heading", { name: "Model response" });
      const section = heading.closest("section");
      if (!(section instanceof HTMLElement))
        throw new Error("Model response section is missing");
      return within(section);
    };

    fireEvent.click(
      responsePanel().getByRole("button", { name: "Load more text" }),
    );
    expect(await screen.findByText(/request-1-response-tail/)).toBeTruthy();

    fireEvent.click(screen.getByRole("tab", { name: /Request 2/ }));
    expect(await screen.findByText("request-2-response")).toBeTruthy();
    expect(screen.queryByText(/request-1-response-tail/)).toBeNull();
    expect(screen.queryByAltText("Saved model content")).toBeNull();
    expect(screen.getByText(/Loading image/)).toBeTruthy();

    resolveSecondImage({
      media_type: "image/png",
      size: 3,
      identifier: "image-2",
      data: "image-two",
    });
    const nextImage = await screen.findByAltText("Saved model content");
    expect(nextImage.getAttribute("src")).toBe(
      "data:image/png;base64,image-two",
    );

    fireEvent.click(
      responsePanel().getByRole("button", { name: "Load more text" }),
    );
    expect(await screen.findByText(/request-2-response-tail/)).toBeTruthy();
    expect(screen.queryByText(/request-1-response-tail/)).toBeNull();
    expect(textRead).toHaveBeenLastCalledWith(
      2,
      4,
      expect.objectContaining({
        source: "response",
        request_ordinal: 2,
        offset: expect.any(Number),
      }),
    );
  });

  it("keeps a live request when the initial history read finishes later", async () => {
    const emptyMessages = {
      messages: [],
      offset: 0,
      total: 0,
      has_more: false,
    };
    const summary: AgentHistoryRequestSummary = {
      ordinal: 1,
      request_id: "request-1",
      run_id: "run-4",
      window_number: 1,
      status: "pending",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: null,
      streaming: false,
      input_length: 0,
      parameters_length: 2,
      settings_length: 2,
      model_length: 2,
      response_length: 0,
      input_count: 0,
      response_count: 0,
      related_count: 0,
      error: null,
    };
    const field = (source: HistoryText["source"]): HistoryText => ({
      kind: "text",
      source,
      request_ordinal: 1,
      message_index: 0,
      path: [],
      offset: 0,
      next_offset: 2,
      total_bytes: 2,
      value: "{}",
      has_more: false,
    });
    const run = {
      sequence: 4,
      run_id: "run-4",
      status: "running",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: null,
      last_saved_at: "2026-01-01T00:00:01Z",
      window_number: 1,
      window_reset_at: null,
      window_reason: null,
      request_count: 1,
      usage: null,
      error: null,
      legacy: false,
    };
    const initialRead: AgentHistoryRead = {
      agent_id: 2,
      run: { ...run, request_count: 0, legacy: true },
      windows: [],
      windows_has_more: false,
      windows_next_after: null,
      requests: [],
      requests_has_more: false,
      requests_next_after: null,
      messages: emptyMessages,
      missing: [],
    };
    const liveRead: AgentHistoryRead = {
      agent_id: 2,
      run,
      windows: [],
      windows_has_more: false,
      windows_next_after: null,
      requests: [summary],
      requests_has_more: false,
      requests_next_after: null,
      messages: emptyMessages,
      missing: [],
      request: {
        summary,
        input: emptyMessages,
        parameters: field("parameters"),
        settings: field("settings"),
        model: field("model"),
        response: null,
        related: emptyMessages,
      },
    };
    let resolveInitial!: (value: AgentHistoryRead) => void;
    const initial = new Promise<AgentHistoryRead>((resolve) => {
      resolveInitial = resolve;
    });
    let reads = 0;
    vi.spyOn(backend, "agentHistoryRead").mockImplementation(async () => {
      reads += 1;
      return reads === 1 ? initial : liveRead;
    });
    let listener: Parameters<typeof backend.onEvent>[0] | undefined;
    vi.spyOn(backend, "onEvent").mockImplementation((next) => {
      listener = next;
      return vi.fn();
    });

    render(
      <TooltipProvider>
        <HistorySection
          agentId={2}
          initialRuns={[
            {
              sequence: 4,
              run_id: "run-4",
              status: "running",
              started_at: "2026-01-01T00:00:00Z",
              completed_at: null,
              last_saved_at: "2026-01-01T00:00:01Z",
              window_number: 1,
              request_count: 1,
              usage: null,
              error: null,
              effects: [],
            },
          ]}
        />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: /#4/ }));
    fireEvent.click(screen.getByRole("button", { name: "View full context" }));
    listener?.({ type: "turn.progress", agent_id: 2, sequence: 4 });
    expect(await screen.findByRole("tab", { name: /Request 1/ })).toBeTruthy();

    resolveInitial(initialRead);
    await Promise.resolve();
    expect(screen.getByRole("tab", { name: /Request 1/ })).toBeTruthy();
  });

  it("keeps a live request when a stale initial history failure finishes later", async () => {
    const emptyMessages = {
      messages: [],
      offset: 0,
      total: 0,
      has_more: false,
    };
    const summary: AgentHistoryRequestSummary = {
      ordinal: 1,
      request_id: "request-1",
      run_id: "run-4",
      window_number: 1,
      status: "pending",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: null,
      streaming: false,
      input_length: 0,
      parameters_length: 2,
      settings_length: 2,
      model_length: 2,
      response_length: 0,
      input_count: 0,
      response_count: 0,
      related_count: 0,
      error: null,
    };
    const field = (source: HistoryText["source"]): HistoryText => ({
      kind: "text",
      source,
      request_ordinal: 1,
      message_index: 0,
      path: [],
      offset: 0,
      next_offset: 2,
      total_bytes: 2,
      value: "{}",
      has_more: false,
    });
    const run = {
      sequence: 4,
      run_id: "run-4",
      status: "running",
      started_at: "2026-01-01T00:00:00Z",
      completed_at: null,
      last_saved_at: "2026-01-01T00:00:01Z",
      window_number: 1,
      window_reset_at: null,
      window_reason: null,
      request_count: 1,
      usage: null,
      error: null,
      legacy: false,
    };
    const liveRead: AgentHistoryRead = {
      agent_id: 2,
      run,
      windows: [],
      windows_has_more: false,
      windows_next_after: null,
      requests: [summary],
      requests_has_more: false,
      requests_next_after: null,
      messages: emptyMessages,
      missing: [],
      request: {
        summary,
        input: emptyMessages,
        parameters: field("parameters"),
        settings: field("settings"),
        model: field("model"),
        response: null,
        related: emptyMessages,
      },
    };
    let rejectInitial!: (reason?: unknown) => void;
    const initial = new Promise<AgentHistoryRead>((_, reject) => {
      rejectInitial = reject;
    });
    let reads = 0;
    vi.spyOn(backend, "agentHistoryRead").mockImplementation(async () => {
      reads += 1;
      return reads === 1 ? initial : liveRead;
    });
    let listener: Parameters<typeof backend.onEvent>[0] | undefined;
    vi.spyOn(backend, "onEvent").mockImplementation((next) => {
      listener = next;
      return vi.fn();
    });

    render(
      <TooltipProvider>
        <HistorySection
          agentId={2}
          initialRuns={[
            {
              sequence: 4,
              run_id: "run-4",
              status: "running",
              started_at: "2026-01-01T00:00:00Z",
              completed_at: null,
              last_saved_at: "2026-01-01T00:00:01Z",
              window_number: 1,
              request_count: 1,
              usage: null,
              error: null,
              effects: [],
            },
          ]}
        />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: /#4/ }));
    fireEvent.click(screen.getByRole("button", { name: "View full context" }));
    listener?.({ type: "turn.progress", agent_id: 2, sequence: 4 });
    expect(await screen.findByRole("tab", { name: /Request 1/ })).toBeTruthy();

    rejectInitial(new Error("stale initial failure"));
    await Promise.resolve();
    expect(screen.getByRole("tab", { name: /Request 1/ })).toBeTruthy();
    expect(screen.queryByText("stale initial failure")).toBeNull();
  });

  it("ignores a history failure after the modal closes", async () => {
    let rejectRead!: (reason?: unknown) => void;
    const pending = new Promise<AgentHistoryRead>((_, reject) => {
      rejectRead = reject;
    });
    vi.spyOn(backend, "agentHistoryRead").mockImplementation(
      async () => pending,
    );

    render(
      <TooltipProvider>
        <HistorySection
          agentId={2}
          initialRuns={[
            {
              sequence: 4,
              run_id: "run-4",
              status: "running",
              started_at: "2026-01-01T00:00:00Z",
              completed_at: null,
              last_saved_at: "2026-01-01T00:00:01Z",
              window_number: 1,
              request_count: 0,
              usage: null,
              error: null,
              effects: [],
            },
          ]}
        />
      </TooltipProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: /#4/ }));
    fireEvent.click(screen.getByRole("button", { name: "View full context" }));
    expect(await screen.findByText("Loading Turn history")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByText("Loading Turn history")).toBeNull();

    rejectRead(new Error("closed history failure"));
    await Promise.resolve();
    expect(screen.queryByText("closed history failure")).toBeNull();
  });
});

describe("Workspace tree", () => {
  it("uses the shared tree with collapsed folders and file counts", () => {
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <WorkspaceSection agentId={2} entries={entries} />
      </TooltipProvider>,
    );
    expect(html).toMatch(/<h2\b[^>]*>Workspace<\/h2>/);
    expect(html).toContain('aria-label="Workspace files"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain("notes");
    expect(html).toContain(">1</p>");
    expect(html).not.toContain("MEMORY.md");
    expect(html).not.toContain("Actions");
  });

  it("opens files without hashes in read-only mode and suppresses all actions", () => {
    const html = renderToStaticMarkup(
      <TooltipProvider>
        <TreeView
          entries={[...entries, { ...entries[1], path: "notes/.image.png" }]}
          expanded={new Set(["notes"])}
          onToggle={() => {}}
          onOpen={() => {}}
          readOnly
          rowActions={() => [
            { id: "delete", label: "Delete", onSelect: () => {} },
          ]}
        />
      </TooltipProvider>,
    );
    const buttons = html.match(/<button\b[^>]*>[\s\S]*?<\/button>/g) ?? [];
    expect(buttons.join("")).toContain(">MEMORY.md<");
    expect(buttons.join("")).toContain(">.image.png<");
    expect(html).toContain('aria-expanded="true"');
    expect(html).not.toContain("Delete");
    expect(html).not.toContain("Actions");
  });

  it("preserves the empty Workspace state", () => {
    expect(
      renderToStaticMarkup(<WorkspaceSection agentId={2} entries={[]} />),
    ).toContain("No Workspace files");
  });

  it("renders escaped, read-only content and its UTF-8 byte size without actions", () => {
    const html = renderToStaticMarkup(
      <WorkspaceContent
        file={{ path: "notes/MEMORY.md", content: "<中文>\n", hash: "hash" }}
      />,
    );
    expect(html).toContain("9 B");
    expect(html).toContain("&lt;中文&gt;\n</pre>");
    expect(html).not.toContain("textarea");
    expect(html).not.toContain("button");
    expect(html).not.toContain("contenteditable");
  });
});
