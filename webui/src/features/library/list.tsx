import {
  File,
  FileText,
  FolderPlus,
  Plus,
  Search,
  SquarePen,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "@/app/router";
import { Page, PageBody, PageHeader, Toolbar } from "@/components/layout/shell";
import { ConfirmDialog, PromptDialog } from "@/components/ui/dialog";
import {
  Button,
  CountPill,
  dismissToast,
  IconButton,
  SearchField,
} from "@/components/ui/index";
import type { MenuAction } from "@/components/ui/menu";
import { StatePanel } from "@/components/ui/state-panel";
import { expandedFolders } from "@/features/library/tree";
import { TreeView, type TreeViewProps } from "@/features/library/tree-view";
import { reportLoadFailure } from "@/features/settings/saver";
import { BackendError, backend, type LibraryEntry } from "@/lib/backend";
import { plural } from "@/lib/format";

type Creation = { kind: LibraryEntry["kind"]; parent: string };

export function creationNameError(name: string): string | undefined {
  if (!name.trim()) return "Enter a name.";
  if (name === "." || name === ".." || /[/\\]/.test(name))
    return "Enter a single name, not a path (no / or \\).";
  return undefined;
}

export function creationDestination(creation: Creation, name: string): string {
  const error = creationNameError(name);
  if (error) throw new Error(error);
  return creation.parent ? `${creation.parent}/${name}` : name;
}

export async function createLibraryEntry(
  creation: Creation,
  name: string,
  entries: LibraryEntry[],
): Promise<string> {
  const destination = creationDestination(creation, name);
  const existing = entries.find((entry) => entry.path === destination);
  if (existing)
    throw new Error(
      `A ${existing.kind === "directory" ? "folder" : "file"} with this name already exists at this location. Choose another name.`,
    );
  if (creation.kind === "directory") {
    await backend.mkdirLibrary(destination);
  } else {
    try {
      await backend.writeLibrary(destination, "");
    } catch (failure) {
      if (
        failure instanceof BackendError &&
        failure.code === "expected_hash_required"
      )
        throw new Error(
          "A file with this name already exists at this location. Choose another name.",
        );
      throw failure;
    }
  }
  return destination;
}

export function libraryActions(
  entry: LibraryEntry,
  handlers: {
    rename: (entry: LibraryEntry) => void;
    create: (creation: Creation) => void;
    remove: (entry: LibraryEntry) => void;
  },
): MenuAction[] {
  return [
    {
      id: "rename",
      label: "Rename",
      icon: <SquarePen size={15} />,
      onSelect: () => handlers.rename(entry),
    },
    ...(entry.kind === "directory"
      ? [
          {
            id: "document",
            label: "New document inside",
            icon: <Plus size={15} />,
            onSelect: () =>
              handlers.create({ kind: "file", parent: entry.path }),
          },
          {
            id: "folder",
            label: "New folder inside",
            icon: <FolderPlus size={15} />,
            onSelect: () =>
              handlers.create({ kind: "directory", parent: entry.path }),
          },
        ]
      : []),
    {
      id: "delete",
      label: "Delete",
      icon: <Trash2 size={15} />,
      tone: "danger",
      onSelect: () => handlers.remove(entry),
    },
  ];
}

export function deleteConsequence(
  entry: LibraryEntry | null,
  entries: LibraryEntry[],
): string {
  if (entry?.kind !== "directory")
    return "The document is removed for every Member.";
  const count = entries.filter(
    (item) => item.kind === "file" && item.path.startsWith(`${entry.path}/`),
  ).length;
  return `Removes ${plural(count, "file")} for every Member.`;
}

function CreateActions({
  onCreate,
  disabled = false,
}: {
  onCreate: (creation: Creation) => void;
  disabled?: boolean;
}) {
  return (
    <>
      <Button
        variant="primary"
        disabled={disabled}
        onClick={() => onCreate({ kind: "file", parent: "" })}
      >
        <Plus size={16} />
        New document
      </Button>
      <IconButton
        label="New folder"
        disabled={disabled}
        onClick={() => onCreate({ kind: "directory", parent: "" })}
      >
        <FolderPlus size={16} />
      </IconButton>
    </>
  );
}

export function LibraryContents({
  onCreate,
  onClearQuery,
  disabled = false,
  ...tree
}: TreeViewProps & {
  onCreate: (creation: Creation) => void;
  onClearQuery?: () => void;
  disabled?: boolean;
}) {
  const needle = tree.query?.trim().toLowerCase() ?? "";
  const count = tree.entries.filter(
    (entry) =>
      entry.kind === "file" && entry.path.toLowerCase().includes(needle),
  ).length;
  const empty = tree.entries.length === 0;
  return (
    <>
      <CountPill>{plural(count, "document")}</CountPill>
      {empty || (needle && count === 0) ? (
        disabled ? null : (
          <StatePanel
            icons={
              needle ? [FileText, Search, File] : [File, FileText, FolderPlus]
            }
            title={needle ? "No documents match" : "The Library is empty"}
            description={
              needle
                ? "Try a different name or clear the search to see the Library."
                : "Create a document or folder to share information with Members."
            }
            action={
              needle
                ? onClearQuery
                  ? {
                      label: "Clear search",
                      icon: <X size={16} />,
                      onClick: onClearQuery,
                    }
                  : undefined
                : {
                    label: "New document",
                    icon: <Plus size={16} />,
                    onClick: () => onCreate({ kind: "file", parent: "" }),
                  }
            }
            secondaryAction={
              needle
                ? undefined
                : {
                    label: "New folder",
                    icon: <FolderPlus size={16} />,
                    onClick: () => onCreate({ kind: "directory", parent: "" }),
                  }
            }
          />
        )
      ) : (
        <TreeView {...tree} />
      )}
    </>
  );
}

export function LibraryPage({ path }: { path?: string }) {
  const navigate = useNavigate();
  const [entries, setEntries] = useState<LibraryEntry[] | null>(null);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(() => expandedFolders(path));
  const [creating, setCreating] = useState<Creation | null>(null);
  const [renaming, setRenaming] = useState<LibraryEntry | null>(null);
  const [doomed, setDoomed] = useState<LibraryEntry | null>(null);
  const [failed, setFailed] = useState(false);

  const load = useCallback(async () => {
    try {
      setEntries(await backend.library());
      setFailed(false);
      dismissToast("library-load");
    } catch (failure) {
      setFailed(true);
      reportLoadFailure("library-load", failure, () => void load());
    }
  }, []);

  useEffect(() => {
    void load();
    return () => dismissToast("library-load");
  }, [load]);

  useEffect(
    () =>
      backend.onEvent((event) => {
        if (
          event.type === "library.updated" ||
          event.type === "connection.restored"
        )
          void load();
      }),
    [load],
  );

  return (
    <Page>
      <PageHeader
        title="Library"
        actions={
          <CreateActions
            onCreate={setCreating}
            disabled={entries === null || failed}
          />
        }
      />
      <Toolbar>
        <SearchField
          icon={<Search size={15} />}
          value={query}
          placeholder="Search documents"
          aria-label="Search documents"
          onChange={(event) => setQuery(event.target.value)}
        />
      </Toolbar>
      <PageBody>
        {entries === null ? null : (
          <LibraryContents
            entries={entries}
            disabled={failed}
            query={query}
            expanded={expanded}
            revealPath={path}
            onToggle={(folder) =>
              setExpanded((current) => {
                const next = new Set(current);
                if (next.has(folder)) next.delete(folder);
                else next.add(folder);
                return next;
              })
            }
            onOpen={(file) => navigate({ name: "document", path: file })}
            onCreate={setCreating}
            onClearQuery={() => setQuery("")}
            rowActions={(entry) =>
              libraryActions(entry, {
                rename: setRenaming,
                create: setCreating,
                remove: setDoomed,
              }).map((action) => ({ ...action, disabled: failed }))
            }
          />
        )}
      </PageBody>
      <PromptDialog
        open={creating !== null}
        onOpenChange={(next) => !next && setCreating(null)}
        title={creating?.kind === "directory" ? "New folder" : "New document"}
        label="Name"
        description={
          creating?.parent
            ? `Location: Library / ${creating.parent}`
            : "Location: Library"
        }
        trim={false}
        validate={creationNameError}
        submitLabel={
          creating?.kind === "directory" ? "Create folder" : "Create document"
        }
        onSubmit={async (name) => {
          if (!creating || !entries)
            throw new Error("Library creation is not ready.");
          const destination = await createLibraryEntry(creating, name, entries);
          if (creating.kind === "directory") {
            setExpanded(
              (current) =>
                new Set([...current, ...expandedFolders(destination)]),
            );
          } else {
            navigate({ name: "document", path: destination });
          }
          await load();
        }}
      />
      <PromptDialog
        open={renaming !== null}
        onOpenChange={(next) => !next && setRenaming(null)}
        title={
          renaming?.kind === "directory" ? "Rename folder" : "Rename document"
        }
        label="New path"
        initial={renaming?.path ?? ""}
        submitLabel="Rename"
        onSubmit={async (destination) => {
          if (renaming) {
            await backend.moveLibrary(renaming.path, destination);
            setExpanded(
              (current) =>
                new Set(
                  [...current].map((folder) =>
                    folder === renaming.path ||
                    folder.startsWith(`${renaming.path}/`)
                      ? destination + folder.slice(renaming.path.length)
                      : folder,
                  ),
                ),
            );
          }
          await load();
        }}
      />
      <ConfirmDialog
        open={doomed !== null}
        onOpenChange={(next) => !next && setDoomed(null)}
        title={`Delete ${doomed?.path ?? ""}?`}
        description={deleteConsequence(doomed, entries ?? [])}
        confirmLabel={
          doomed?.kind === "directory" ? "Delete folder" : "Delete document"
        }
        onConfirm={async () => {
          if (doomed) await backend.deleteLibrary(doomed.path);
          await load();
        }}
      />
    </Page>
  );
}
