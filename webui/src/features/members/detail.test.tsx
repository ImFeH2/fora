import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
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
  type AgentRun,
  backend,
  type LibraryEntry,
} from "@/lib/backend";

vi.mock("@/components/ui/dialog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/ui/dialog")>()),
  ConfirmDialog: vi.fn(() => null),
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
