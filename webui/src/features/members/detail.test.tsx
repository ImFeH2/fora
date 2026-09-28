import type { ReactNode } from "react";
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
import {
  type AgentDetail,
  type AgentHistoryRead,
  type AgentHistoryRequestSummary,
  type AgentRun,
  backend,
  type HistoryText,
  type LibraryEntry,
} from "@/lib/backend";

vi.mock("@/components/ui/dialog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/ui/dialog")>()),
  ConfirmDialog: vi.fn(() => null),
  Modal: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
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
