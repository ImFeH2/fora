import { Save, SquarePen, Trash2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { type Route, useNavigate } from "@/app/router";
import {
  type Crumb,
  Page,
  PageBody,
  PageHeader,
  Toolbar,
} from "@/components/layout/shell";
import { ConfirmDialog, PromptDialog } from "@/components/ui/dialog";
import {
  Button,
  Chip,
  dismissToast,
  EmptyState,
  Spinner,
  Textarea,
  toast,
} from "@/components/ui/index";
import { OverflowMenu } from "@/components/ui/menu";
import { reportLoadFailure } from "@/features/settings/saver";
import { BackendError, backend } from "@/lib/backend";
import { formatBytes } from "@/lib/format";

type Loaded = { content: string; hash: string };

function conflictToastId(path: string): string {
  return `document-conflict:${path}`;
}

const conflictDescription =
  "Your unsaved changes are preserved. Reopen replaces them with the current document.";

export function documentCrumbs(
  path: string,
  navigate: (route: Route) => void,
): Crumb[] {
  const parts = path.split("/");
  return [
    { label: "Library", onSelect: () => navigate({ name: "library" }) },
    ...parts.map((label, index) => ({
      label,
      onSelect: () =>
        navigate({
          name: "library",
          path: parts.slice(0, index + 1).join("/"),
        }),
    })),
  ];
}

export function DocumentUnavailable({
  code,
}: {
  code: "not_found" | "not_readable";
}) {
  return (
    <EmptyState
      title={
        code === "not_readable"
          ? "Cannot read this file as UTF-8 text"
          : "Document not found"
      }
    />
  );
}

export function DocumentPage({ path }: { path: string }) {
  const navigate = useNavigate();
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [draft, setDraft] = useState("");
  const [unavailable, setUnavailable] = useState<
    "not_found" | "not_readable" | null
  >(null);
  const [failed, setFailed] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [doomed, setDoomed] = useState(false);

  const editing = useRef({ loaded, draft });
  editing.current = { loaded, draft };
  const loadRevision = useRef(0);

  const load = useCallback(
    async (preserve = false) => {
      const revision = ++loadRevision.current;
      try {
        const document = await backend.readLibrary(path);
        if (revision !== loadRevision.current) return;
        const current = editing.current;
        if (
          preserve &&
          current.loaded &&
          current.draft !== current.loaded.content
        ) {
          if (document.hash !== current.loaded.hash)
            toast({
              id: conflictToastId(path),
              tone: "danger",
              title: "Saved elsewhere",
              description: conflictDescription,
              duration: null,
              action: { label: "Reopen", onClick: () => void load() },
            });
        } else {
          setLoaded({ content: document.content, hash: document.hash });
          setDraft(document.content);
          dismissToast(conflictToastId(path));
        }
        setUnavailable(null);
        setFailed(false);
        dismissToast(`document-load:${path}`);
      } catch (failure) {
        if (revision !== loadRevision.current) return;
        setFailed(true);
        if (
          failure instanceof BackendError &&
          (failure.code === "not_found" || failure.code === "not_readable")
        ) {
          setUnavailable(failure.code);
        } else {
          reportLoadFailure(
            `document-load:${path}`,
            failure,
            () => void load(preserve),
          );
        }
      }
    },
    [path],
  );

  useEffect(() => {
    void load();
    const off = backend.onEvent((event) => {
      if (event.type === "connection.restored") void load(true);
    });
    return () => {
      loadRevision.current += 1;
      off();
      dismissToast(`document-load:${path}`);
    };
  }, [load, path]);

  useEffect(() => () => dismissToast(conflictToastId(path)), [path]);

  const dirty = loaded !== null && draft !== loaded.content;
  const crumb = documentCrumbs(path, navigate);

  const save = async () => {
    if (!dirty || loaded === null) return;
    setBusy(true);
    setSaveError(null);
    try {
      const result = await backend.writeLibrary(path, draft, loaded.hash);
      if (result.conflict) {
        toast({
          id: conflictToastId(path),
          tone: "danger",
          title: "Saved elsewhere",
          description: conflictDescription,
          duration: null,
          action: { label: "Reopen", onClick: () => void load() },
        });
        return;
      }
      setLoaded({ content: draft, hash: result.hash });
    } catch (failure) {
      if (failure instanceof BackendError && failure.transport)
        backend.reportFailure(failure);
      else
        setSaveError(
          failure instanceof Error ? failure.message : String(failure),
        );
    } finally {
      setBusy(false);
    }
  };

  if (unavailable && loaded === null) {
    return (
      <Page>
        <PageHeader title={path} crumb={crumb} />
        <PageBody>
          <DocumentUnavailable code={unavailable} />
        </PageBody>
      </Page>
    );
  }

  return (
    <Page>
      <PageHeader
        title={path}
        crumb={crumb}
        actions={
          <>
            <Button
              variant="primary"
              disabled={!dirty || busy || failed}
              onClick={save}
            >
              {busy ? <Spinner label="Saving document" /> : <Save size={16} />}
              Save
            </Button>
            <OverflowMenu
              label={`Actions for ${path}`}
              actions={[
                {
                  id: "rename",
                  label: "Rename",
                  icon: <SquarePen size={15} />,
                  onSelect: () => setRenaming(true),
                },
                {
                  id: "delete",
                  label: "Delete",
                  icon: <Trash2 size={15} />,
                  tone: "danger",
                  onSelect: () => setDoomed(true),
                },
              ]}
            />
          </>
        }
      />
      <Toolbar>
        <Chip>{formatBytes(new TextEncoder().encode(draft).length)}</Chip>
        {dirty ? <Chip tone="blue">Unsaved changes</Chip> : null}
      </Toolbar>
      <PageBody variant="flush">
        <div className="flex flex-1 min-h-0 flex-col gap-4 px-8 pb-6">
          {saveError ? (
            <p role="alert" className="text-sm text-danger">
              {saveError}
            </p>
          ) : null}
          {unavailable ? <DocumentUnavailable code={unavailable} /> : null}
          <div className="grid flex-1 min-h-0 grid-rows-[minmax(0,1fr)] items-stretch font-mono tracking-[0]">
            <Textarea
              aria-label="Document"
              disabled={loaded === null}
              readOnly={failed}
              value={draft}
              spellCheck={false}
              onChange={(event) => {
                setDraft(event.target.value);
              }}
            />
          </div>
        </div>
      </PageBody>

      <PromptDialog
        open={renaming}
        onOpenChange={setRenaming}
        title="Rename document"
        label="New path"
        initial={path}
        submitLabel="Rename"
        onSubmit={async (destination) => {
          const moved = await backend.moveLibrary(path, destination);
          setRenaming(false);
          navigate({ name: "document", path: moved.path });
        }}
      />
      <ConfirmDialog
        open={doomed}
        onOpenChange={setDoomed}
        title={`Delete ${path}?`}
        description="The document is removed for every Member."
        confirmLabel="Delete document"
        onConfirm={async () => {
          await backend.deleteLibrary(path);
          setDoomed(false);
          navigate({ name: "library" });
        }}
      />
    </Page>
  );
}
