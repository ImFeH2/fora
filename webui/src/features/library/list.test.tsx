import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  createLibraryEntry,
  creationDestination,
  creationNameError,
  deleteConsequence,
  LibraryContents,
  libraryActions,
} from "@/features/library/list";
import { BackendError, backend, type LibraryEntry } from "@/lib/backend";

const entries: LibraryEntry[] = [
  {
    path: "runbooks",
    kind: "directory",
    size: 0,
    modified_at: "2026-01-01T00:00:00Z",
  },
  {
    path: "runbooks/deep",
    kind: "directory",
    size: 0,
    modified_at: "2026-01-01T00:00:00Z",
  },
  {
    path: "runbooks/deep/notes.md",
    kind: "file",
    size: 1200,
    modified_at: "2026-01-01T00:00:00Z",
  },
  {
    path: "runbooks/image.png",
    kind: "file",
    size: 2048,
    modified_at: "2026-01-01T00:00:00Z",
  },
  {
    path: "empty",
    kind: "directory",
    size: 0,
    modified_at: "2026-01-01T00:00:00Z",
  },
];

const handlers = () => ({ create: vi.fn(), rename: vi.fn(), remove: vi.fn() });

function render(
  expanded: string[] = [],
  query = "",
  data = entries,
  disabled = false,
) {
  return renderToStaticMarkup(
    <TooltipProvider>
      <LibraryContents
        entries={data}
        expanded={new Set(expanded)}
        query={query}
        onToggle={() => {}}
        onOpen={() => {}}
        onCreate={() => {}}
        onClearQuery={() => {}}
        disabled={disabled}
        rowActions={(entry) => libraryActions(entry, handlers())}
      />
    </TooltipProvider>,
  );
}

describe("Library tree", () => {
  it("renders collapsed folders with recursive file counts and retains empty folders", () => {
    const html = render();
    expect(html).toContain('aria-label="Library documents"');
    expect(html).toContain("2 documents");
    expect(html).toContain("runbooks");
    expect(html).toContain("empty");
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain(">2</p>");
    expect(html).toContain(">0</p>");
    expect(html).not.toContain("notes.md");
    expect(html).not.toContain("image.png");
  });

  it("renders only expanded levels", () => {
    const html = render(["runbooks"]);
    expect(html).toContain('aria-expanded="true"');
    expect(html).toContain("deep");
    expect(html).toContain("image.png");
    expect(html).not.toContain("notes.md");
    expect(render(["runbooks", "runbooks/deep"])).toContain("notes.md");
  });

  it("links all files without content hashes and keeps their actions and size", () => {
    const html = render([], "image");
    expect(html).not.toContain("Unreadable");
    const buttons = html.match(/<button\b[^>]*>[\s\S]*?<\/button>/g) ?? [];
    expect(buttons.join("")).toContain(">image.png<");
    expect(html).toContain('aria-label="Actions for runbooks/image.png"');
    expect(html).toContain("2.0 kB");
  });

  it("flattens path matches without needing expansion and shows folder chips", () => {
    const html = render([], "  DEEP/NOTES  ");
    expect(html).toContain("1 document");
    expect(html).toContain("notes.md");
    expect(html).toContain(">runbooks/deep</span>");
    expect(html).not.toContain('aria-expanded="true"');
    expect(html).not.toContain('aria-label="Actions for runbooks"');
    expect(html).not.toContain("image.png");
  });

  it("shows timestamps with exact-time tooltip triggers instead of titles", () => {
    const html = render(["runbooks", "runbooks/deep"]);
    expect(html).toMatch(/<time[^>]*datetime="2026-01-01T00:00:00Z"/i);
    expect(html).not.toContain("title=");
    expect(html).toContain("Modified");
  });

  it("offers both creation actions only for the empty Library", () => {
    const html = render([], "", []);
    expect(html).toContain("The Library is empty");
    expect(html).toContain("New document");
    expect(html).toContain("<span>New folder</span>");
    const filtered = render([], "nothing-matches");
    expect(filtered).toContain("No documents match");
    expect(filtered).not.toContain("New document");
    expect(filtered).toContain("<span>Clear search</span>");
  });

  it("suppresses empty results when Library loading has failed", () => {
    const empty = render([], "", [], true);
    expect(empty).not.toContain("The Library is empty");
    expect(empty).not.toContain("New document");
    const filtered = render([], "nothing-matches", entries, true);
    expect(filtered).not.toContain("No documents match");
    expect(filtered).not.toContain("Clear search");
  });

  it("does not treat a Library containing only empty folders as empty", () => {
    const html = render([], "", [entries[4]]);
    expect(html).toContain("0 documents");
    expect(html).toContain('aria-label="Library documents"');
    expect(html).not.toContain("The Library is empty");
  });
});

describe("Library row actions", () => {
  it("offers files Rename and Delete and dispatches the selected file", () => {
    const callbacks = handlers();
    const actions = libraryActions(entries[2], callbacks);
    expect(actions.map((action) => action.label)).toEqual(["Rename", "Delete"]);
    actions[0].onSelect();
    actions[1].onSelect();
    expect(callbacks.rename).toHaveBeenCalledWith(entries[2]);
    expect(callbacks.remove).toHaveBeenCalledWith(entries[2]);
  });

  it("keeps the full parent path separate from the entered name", () => {
    const callbacks = handlers();
    const actions = libraryActions(entries[1], callbacks);
    expect(actions.map((action) => action.label)).toEqual([
      "Rename",
      "New document inside",
      "New folder inside",
      "Delete",
    ]);
    actions[1].onSelect();
    actions[2].onSelect();
    expect(callbacks.create.mock.calls).toEqual([
      [{ kind: "file", parent: "runbooks/deep" }],
      [{ kind: "directory", parent: "runbooks/deep" }],
    ]);
  });

  it("names all files removed by a folder delete, including unreadable files", () => {
    expect(deleteConsequence(entries[0], entries)).toBe(
      "Removes 2 files for every Member.",
    );
    expect(deleteConsequence(entries[1], entries)).toBe(
      "Removes 1 file for every Member.",
    );
    expect(deleteConsequence(entries[4], entries)).toBe(
      "Removes 0 files for every Member.",
    );
    expect(deleteConsequence(entries[2], entries)).toBe(
      "The document is removed for every Member.",
    );
    expect(
      deleteConsequence(entries[0], [
        ...entries,
        { ...entries[2], path: "runbooks-other/notes.md" },
      ]),
    ).toBe("Removes 2 files for every Member.");
  });
});

describe("Library creation", () => {
  afterEach(() => vi.restoreAllMocks());

  it.each([
    "",
    "   ",
    ".",
    "..",
    "temp/name",
    "temp\\name",
    "/root",
    "../name",
  ])("rejects invalid name %j without constructing another path", (name) => {
    expect(creationNameError(name)).toBeTruthy();
    expect(() =>
      creationDestination({ kind: "file", parent: "temp" }, name),
    ).toThrow();
  });

  it.each(["ai-chat-input.tsx", "notes.txt", "notes.md", ".hidden", " name "])(
    "preserves the single name %j and the fixed parent",
    (name) => {
      expect(creationNameError(name)).toBeUndefined();
      expect(creationDestination({ kind: "file", parent: "temp" }, name)).toBe(
        `temp/${name}`,
      );
      expect(creationDestination({ kind: "file", parent: "" }, name)).toBe(
        name,
      );
    },
  );

  it("creates inside the selected directory even when the root has the same file", async () => {
    const write = vi
      .spyOn(backend, "writeLibrary")
      .mockResolvedValue(
        {} as Awaited<ReturnType<typeof backend.writeLibrary>>,
      );
    const destination = await createLibraryEntry(
      { kind: "file", parent: "temp" },
      "ai-chat-input.tsx",
      [{ ...entries[2], path: "ai-chat-input.tsx" }],
    );
    expect(destination).toBe("temp/ai-chat-input.tsx");
    expect(write.mock.calls).toEqual([["temp/ai-chat-input.tsx", ""]]);
  });

  it.each(["file", "directory"] as const)(
    "rejects a known same-location %s without any write",
    async (kind) => {
      const write = vi.spyOn(backend, "writeLibrary");
      const mkdir = vi.spyOn(backend, "mkdirLibrary");
      for (const creating of ["file", "directory"] as const) {
        await expect(
          createLibraryEntry({ kind: creating, parent: "temp" }, "same", [
            { ...entries[0], kind, path: "temp/same" },
          ]),
        ).rejects.toThrow("already exists at this location");
      }
      expect(write).not.toHaveBeenCalled();
      expect(mkdir).not.toHaveBeenCalled();
    },
  );

  it("translates a backend duplicate only in creation and never retries with a hash", async () => {
    const write = vi
      .spyOn(backend, "writeLibrary")
      .mockRejectedValue(
        new BackendError("expected_hash_required", "internal detail"),
      );
    const read = vi.spyOn(backend, "readLibrary");
    await expect(
      createLibraryEntry({ kind: "file", parent: "temp" }, "race.txt", []),
    ).rejects.toThrow(
      "A file with this name already exists at this location. Choose another name.",
    );
    expect(write.mock.calls).toEqual([["temp/race.txt", ""]]);
    expect(read).not.toHaveBeenCalled();
  });

  it("preserves unrelated backend failures", async () => {
    const failure = new BackendError("invalid_path", "Path is a directory");
    vi.spyOn(backend, "writeLibrary").mockRejectedValue(failure);
    await expect(
      createLibraryEntry({ kind: "file", parent: "" }, "name", []),
    ).rejects.toBe(failure);
  });

  it.each(["", "temp"])(
    "creates folders under fixed parent %j",
    async (parent) => {
      const mkdir = vi
        .spyOn(backend, "mkdirLibrary")
        .mockResolvedValue(entries[0]);
      const expected = parent ? "temp/new-folder" : "new-folder";
      await expect(
        createLibraryEntry({ kind: "directory", parent }, "new-folder", []),
      ).resolves.toBe(expected);
      expect(mkdir).toHaveBeenCalledWith(expected);
    },
  );
});
