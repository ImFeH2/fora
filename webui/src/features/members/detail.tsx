import { clsx } from "clsx";
import {
  Check,
  ChevronDown,
  CircleAlert,
  FileText,
  Gauge,
  Maximize2,
  Pause,
  PencilLine,
  Play,
  RefreshCw,
  Send,
  Terminal,
  Trash2,
} from "lucide-react";
import { type ReactNode, useCallback, useEffect, useState } from "react";
import { useOrganization } from "@/app/organization";
import { useNavigate } from "@/app/router";
import { Page, PageBody, PageHeader, Section } from "@/components/layout/shell";
import { ConfirmDialog, Modal } from "@/components/ui/dialog";
import {
  Avatar,
  Button,
  Chip,
  Dot,
  dismissToast,
  Meter,
  StateDot,
  StatusText,
} from "@/components/ui/index";
import { OverflowMenu } from "@/components/ui/menu";
import { StatePanel } from "@/components/ui/state-panel";
import { Tooltip } from "@/components/ui/tooltip";
import { TreeView } from "@/features/library/tree-view";
import { agentStateLabel, canResume } from "@/features/members/state";
import { AgentModelPanel } from "@/features/settings/model";
import { reportLoadFailure } from "@/features/settings/saver";
import {
  type AgentDetail,
  type AgentHistoryRead,
  type AgentHistoryRun,
  type AgentRun,
  backend,
  type HistoryBinary,
  type HistoryMessages,
  type HistoryText,
  type HistoryValue,
  type LibraryDocument,
  type LibraryEntry,
  type Member,
} from "@/lib/backend";
import { formatBytes, formatTime, plural, relativeTime } from "@/lib/format";

const TOOL_ICONS: Record<string, ReactNode> = {
  send: <Send size={13} />,
  run: <Terminal size={13} />,
  edit: <PencilLine size={13} />,
  ack: <Check size={13} />,
};

function toolIcon(tool: string): ReactNode {
  if (TOOL_ICONS[tool]) return TOOL_ICONS[tool];
  if (tool.startsWith("library")) return <FileText size={13} />;
  return <Gauge size={13} />;
}

function runTone(status: string): "green" | "red" | "blue" | "yellow" | "grey" {
  if (status === "completed") return "green";
  if (status === "failed") return "red";
  if (status === "running") return "blue";
  if (status === "interrupted") return "yellow";
  return "grey";
}

export function MemberPage({ id }: { id: number }) {
  const { members, refresh } = useOrganization();
  const navigate = useNavigate();
  const member = members.find((item) => item.id === id);
  const agent = member?.type === "agent" ? member : null;

  useEffect(() => {
    if (!agent) navigate({ name: "members" });
  }, [agent, navigate]);

  if (!agent) return null;
  return <AgentPage member={agent} refresh={refresh} />;
}

function AgentPage({
  member,
  refresh,
}: {
  member: Member;
  refresh: () => Promise<void>;
}) {
  const navigate = useNavigate();
  const [detail, setDetail] = useState<AgentDetail | null>(null);
  const [doomed, setDoomed] = useState(false);
  const paused = canResume(member);

  const load = useCallback(async () => {
    try {
      setDetail(await backend.agentDetail(member.id));
      dismissToast(`agent-load:${member.id}`);
    } catch (failure) {
      reportLoadFailure(`agent-load:${member.id}`, failure, () => void load());
    }
  }, [member.id]);

  useEffect(() => {
    void load();
    return () => dismissToast(`agent-load:${member.id}`);
  }, [load, member.id]);

  useEffect(() => {
    return backend.onEvent((event) => {
      if (
        event.type.startsWith("turn.") ||
        event.type === "organization.changed" ||
        event.type === "settings.updated" ||
        event.type === "connection.restored"
      )
        void load();
    });
  }, [load]);

  return (
    <Page>
      <PageHeader
        title={member.name}
        status={
          <>
            <AgentState member={member} />
            {detail ? <AgentDetailStatus detail={detail} /> : null}
          </>
        }
        crumb={{
          label: "Members",
          onSelect: () => navigate({ name: "members" }),
        }}
        leading={<Avatar memberId={member.id} size="lg" />}
        actions={
          <>
            <Button
              variant={paused ? "primary" : "default"}
              onClick={async () => {
                await (paused
                  ? backend.resumeAgent(member.id)
                  : backend.pauseAgent(member.id));
                await refresh();
              }}
            >
              {paused ? <Play size={16} /> : <Pause size={16} />}
              {paused ? "Resume" : "Pause"}
            </Button>
            <OverflowMenu
              label={`Actions for ${member.name}`}
              actions={[
                {
                  id: "delete",
                  label: "Delete",
                  icon: <Trash2 size={15} />,
                  tone: "danger",
                  disabled: member.state === "running",
                  onSelect: () => setDoomed(true),
                },
              ]}
            />
          </>
        }
      />
      <PageBody>
        {detail ? (
          <>
            <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-px overflow-hidden rounded-md border border-line bg-line flex-none">
              <Stat
                label="Token spend"
                value={detail.usage.total_tokens.toLocaleString("en-US")}
                detail={
                  detail.token_limit === null ? (
                    <span className="text-fg-muted">
                      Token ceiling unavailable
                    </span>
                  ) : detail.token_limit > 0 ? (
                    <>
                      <Meter
                        value={detail.usage.total_tokens}
                        max={detail.token_limit}
                        label={`Token spend for ${member.name}`}
                      />
                      <span className="text-fg-muted">
                        of {detail.token_limit.toLocaleString("en-US")}
                      </span>
                    </>
                  ) : (
                    <span className="text-fg-muted">No ceiling</span>
                  )
                }
              />
              <Stat
                label="Model requests"
                value={detail.usage.requests.toLocaleString("en-US")}
                detail={
                  <span className="text-fg-muted">
                    {detail.usage.input_tokens.toLocaleString("en-US")} in ·{" "}
                    {detail.usage.output_tokens.toLocaleString("en-US")} out
                  </span>
                }
              />
              <Stat label="State" value={<AgentState member={member} />} />
            </div>

            {detail.reasons?.map((reason) => (
              <p key={reason.code} className="text-warning">
                {reason.message}. {reason.recovery}
              </p>
            ))}
            {detail.error ? (
              <p className="text-danger">Last Turn: {detail.error}</p>
            ) : null}
            {detail.statistics_unavailable ? (
              <p className="text-warning">
                Statistics unavailable: {detail.statistics_unavailable}
              </p>
            ) : null}

            <AgentModelPanel key={member.id} agentId={member.id} />

            <WorkspaceSection agentId={member.id} entries={detail.workspace} />

            <HistorySection agentId={member.id} initialRuns={detail.runs} />
          </>
        ) : null}
      </PageBody>

      <ConfirmDialog
        open={doomed}
        onOpenChange={setDoomed}
        title={`Delete ${member.name}?`}
        description="It leaves every Discussion and stops running."
        confirmLabel="Delete Agent"
        onConfirm={async () => {
          await backend.deleteAgent(member.id);
          await refresh();
        }}
      />
    </Page>
  );
}

export function AgentDetailStatus({ detail }: { detail: AgentDetail }) {
  return (
    <>
      {detail.over_token_limit ? (
        <Chip tone="danger">
          <CircleAlert size={12} />
          Token ceiling
        </Chip>
      ) : null}
      {detail.pause_requested && detail.state === "running" ? (
        <Chip tone="warning">Pause requested · Current Turn will finish</Chip>
      ) : null}
      {detail.idle ? (
        <Chip tone="warning">{plural(detail.idle_streak, "idle Turn")}</Chip>
      ) : null}
      {detail.window.number > 1 ? (
        <Tooltip
          focusable
          label={[
            detail.window.reason,
            detail.window.reset_at ? formatTime(detail.window.reset_at) : null,
          ]
            .filter(Boolean)
            .join(" · ")}
        >
          <Chip>Window {detail.window.number.toLocaleString("en-US")}</Chip>
        </Tooltip>
      ) : null}
    </>
  );
}

export function WorkspaceContent({ file }: { file: LibraryDocument }) {
  return (
    <>
      <Chip>{formatBytes(new TextEncoder().encode(file.content).length)}</Chip>
      <pre className="max-h-[50vh] overflow-auto p-3 rounded-sm border border-line bg-surface text-fg font-mono text-xs leading-body tracking-[0]">
        {file.content}
      </pre>
    </>
  );
}

export function WorkspaceSection({
  agentId,
  entries,
}: {
  agentId: number;
  entries: LibraryEntry[];
}) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [path, setPath] = useState<string | null>(null);
  const [file, setFile] = useState<LibraryDocument | null>(null);

  useEffect(() => {
    if (path === null) return;
    let active = true;
    const toastId = `workspace-read:${agentId}:${path}`;
    const load = async () => {
      try {
        const result = await backend.workspaceRead(agentId, path);
        if (!active) return;
        setFile(result);
        dismissToast(toastId);
      } catch (failure) {
        if (active) reportLoadFailure(toastId, failure, () => void load());
      }
    };
    void load();
    const off = backend.onEvent((event) => {
      if (event.type === "connection.restored") void load();
    });
    return () => {
      active = false;
      off();
      dismissToast(toastId);
    };
  }, [agentId, path]);

  return (
    <Section title="Workspace">
      {entries.length === 0 ? (
        <StatePanel
          compact
          icons={[PencilLine, FileText, Terminal]}
          title="No Workspace files"
          description="Files created by this Agent will appear here."
        />
      ) : (
        <TreeView
          entries={entries}
          expanded={expanded}
          onToggle={(folder) =>
            setExpanded((current) => {
              const next = new Set(current);
              if (next.has(folder)) next.delete(folder);
              else next.add(folder);
              return next;
            })
          }
          onOpen={(next) => {
            setFile(null);
            setPath(next);
          }}
          readOnly
        />
      )}
      <Modal
        open={path !== null}
        onOpenChange={(open) => !open && setPath(null)}
        title={path ?? ""}
        footer={null}
      >
        {file ? <WorkspaceContent file={file} /> : null}
      </Modal>
    </Section>
  );
}

function AgentState({ member }: { member: Member }) {
  return (
    <StatusText
      dot={<StateDot state={member.state} ping={member.state === "running"} />}
    >
      {agentStateLabel(member)}
    </StatusText>
  );
}

function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1 p-4 bg-surface">
      <span className="text-xs font-bold tracking-caps uppercase text-fg-muted">
        {label}
      </span>
      <span className="text-md font-semibold tabular-nums tracking-title">
        {value}
      </span>
      {detail ? (
        <span className="flex flex-col gap-1 text-xs">{detail}</span>
      ) : null}
    </div>
  );
}

function historyRun(run: AgentRun): AgentHistoryRun {
  const requestCount = run.request_count ?? 0;
  return {
    sequence: run.sequence,
    run_id: run.run_id ?? "",
    status: run.status,
    started_at: run.started_at,
    completed_at: run.completed_at,
    last_saved_at: run.last_saved_at ?? null,
    window_number: run.window_number ?? null,
    window_reset_at: run.window_reset_at ?? null,
    window_reason: run.window_reason ?? null,
    request_count: requestCount,
    usage: run.usage,
    error: run.error,
    legacy: requestCount === 0,
  };
}

function mergeHistoryRuns(
  current: AgentHistoryRun[],
  incoming: AgentHistoryRun[],
): AgentHistoryRun[] {
  const runs = new Map(current.map((run) => [run.sequence, run]));
  for (const run of incoming) runs.set(run.sequence, run);
  return [...runs.values()].sort(
    (left, right) => right.sequence - left.sequence,
  );
}

export function HistorySection({
  agentId,
  initialRuns,
}: {
  agentId: number;
  initialRuns: AgentRun[];
}) {
  const [runs, setRuns] = useState(() => initialRuns.map(historyRun));
  const [hasOlder, setHasOlder] = useState(initialRuns.length >= 30);
  const [before, setBefore] = useState<number | undefined>(() => {
    const last = initialRuns[initialRuns.length - 1];
    return last?.sequence;
  });
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);

  useEffect(() => {
    setRuns((current) =>
      mergeHistoryRuns(current, initialRuns.map(historyRun)),
    );
    if (initialRuns.length < 30) setHasOlder(false);
  }, [initialRuns]);

  const loadOlder = async () => {
    if (loadingOlder || !hasOlder || before === undefined) return;
    setLoadingOlder(true);
    try {
      const page = await backend.agentHistory(agentId, before, 30);
      setRuns((current) => mergeHistoryRuns(current, page.runs));
      setHasOlder(page.has_before);
      setBefore(page.next_before ?? undefined);
    } finally {
      setLoadingOlder(false);
    }
  };

  return (
    <Section title="History">
      {runs.length === 0 ? (
        <StatePanel
          compact
          icons={[Terminal, Play, Gauge]}
          title="No Turns yet"
          description="This Agent's Turns will appear here."
        />
      ) : (
        <>
          <ul className="flex flex-col gap-2">
            {runs.map((run) => (
              <TurnCard
                key={run.sequence}
                run={run}
                onOpen={() => setSelected(run.sequence)}
              />
            ))}
          </ul>
          {hasOlder ? (
            <Button disabled={loadingOlder} onClick={() => void loadOlder()}>
              {loadingOlder ? (
                <RefreshCw className="animate-spin" size={14} />
              ) : null}
              Load earlier Turns
            </Button>
          ) : null}
        </>
      )}
      {selected !== null ? (
        <TurnHistoryModal
          agentId={agentId}
          sequence={selected}
          open
          onOpenChange={(open) => !open && setSelected(null)}
        />
      ) : null}
    </Section>
  );
}

function TurnHistoryModal({
  agentId,
  sequence,
  open,
  onOpenChange,
}: {
  agentId: number;
  sequence: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [read, setRead] = useState<AgentHistoryRead | null>(null);
  const [ordinal, setOrdinal] = useState<number | undefined>();
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (
      nextOrdinal: number | undefined,
      options: {
        target?: "run" | "input" | "response" | "related";
        offset?: number;
      } = {},
    ) => {
      const target = options.target;
      const offset = options.offset ?? 0;
      try {
        const result = await backend.agentHistoryRead(
          agentId,
          sequence,
          nextOrdinal,
          target === "run" ? offset : 0,
          target === "input" ? offset : 0,
          target === "response" ? offset : 0,
          target === "related" ? offset : 0,
        );
        if (!target && nextOrdinal === undefined && result.request) {
          setOrdinal(result.request.summary.ordinal);
        }
        setRead((current) => {
          if (!current || !target) return result;
          if (target === "run") {
            return {
              ...result,
              messages: {
                ...result.messages,
                offset: current.messages.offset,
                messages: [
                  ...current.messages.messages,
                  ...result.messages.messages,
                ],
              },
            };
          }
          if (!current.request || !result.request) return result;
          return {
            ...result,
            request: {
              ...result.request,
              input:
                target === "input"
                  ? {
                      ...result.request.input,
                      offset: current.request.input.offset,
                      messages: [
                        ...current.request.input.messages,
                        ...result.request.input.messages,
                      ],
                    }
                  : result.request.input,
              response:
                target === "response" &&
                current.request.response &&
                result.request.response
                  ? {
                      ...result.request.response,
                      offset: current.request.response.offset,
                      messages: [
                        ...current.request.response.messages,
                        ...result.request.response.messages,
                      ],
                    }
                  : result.request.response,
              related:
                target === "related"
                  ? {
                      ...result.request.related,
                      offset: current.request.related.offset,
                      messages: [
                        ...current.request.related.messages,
                        ...result.request.related.messages,
                      ],
                    }
                  : result.request.related,
            },
          };
        });
        setError(null);
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
    },
    [agentId, sequence],
  );

  useEffect(() => {
    if (!open) return;
    setRead(null);
    setOrdinal(undefined);
    void load(undefined);
  }, [load, open, sequence]);

  useEffect(() => {
    if (!open) return;
    return backend.onEvent((event) => {
      if (
        event.type.startsWith("turn.") &&
        event.agent_id === agentId &&
        event.sequence === sequence
      )
        void load(ordinal);
    });
  }, [agentId, load, open, ordinal, sequence]);

  const selectRequest = (nextOrdinal: number) => {
    setOrdinal(nextOrdinal);
    void load(nextOrdinal);
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title={`Turn #${sequence}`}
      wide
      footer={<Button onClick={() => onOpenChange(false)}>Close</Button>}
    >
      <div className="min-h-0 overflow-auto flex flex-col gap-4 pr-1">
        {error ? (
          <div className="flex items-center justify-between gap-3 py-2 px-3 border border-red-300/40 rounded-xs bg-red-500/15 text-red-100 text-xs">
            <span className="flex items-center gap-2 min-w-0 wrap-anywhere">
              <CircleAlert size={14} aria-hidden="true" />
              {error}
            </span>
            <Button size="sm" onClick={() => void load(ordinal)}>
              <RefreshCw size={13} />
              Retry
            </Button>
          </div>
        ) : null}
        {read ? (
          <HistoryReadPanel
            read={read}
            ordinal={ordinal}
            onSelectRequest={selectRequest}
            onLoadMoreRun={() =>
              void load(ordinal, {
                target: "run",
                offset:
                  read.messages.next_offset ??
                  read.messages.offset + read.messages.messages.length,
              })
            }
            onLoadMoreInput={() =>
              read.request
                ? void load(ordinal, {
                    target: "input",
                    offset:
                      read.request.input.offset +
                      read.request.input.messages.length,
                  })
                : undefined
            }
            onLoadMoreResponse={() =>
              read.request?.response
                ? void load(ordinal, {
                    target: "response",
                    offset:
                      read.request.response.offset +
                      read.request.response.messages.length,
                  })
                : undefined
            }
            onLoadMoreRelated={() =>
              read.request
                ? void load(ordinal, {
                    target: "related",
                    offset:
                      read.request.related.offset +
                      read.request.related.messages.length,
                  })
                : undefined
            }
          />
        ) : error ? null : (
          <StatePanel
            compact
            icons={[Terminal, Gauge, FileText]}
            title="Loading Turn history"
          />
        )}
      </div>
    </Modal>
  );
}

function HistoryReadPanel({
  read,
  ordinal,
  onSelectRequest,
  onLoadMoreRun,
  onLoadMoreInput,
  onLoadMoreResponse,
  onLoadMoreRelated,
}: {
  read: AgentHistoryRead;
  ordinal: number | undefined;
  onSelectRequest: (ordinal: number) => void;
  onLoadMoreRun: () => void;
  onLoadMoreInput: () => void;
  onLoadMoreResponse: () => void;
  onLoadMoreRelated: () => void;
}) {
  return (
    <>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-muted">
        <span>{read.run.status}</span>
        <span>{formatTime(read.run.started_at)}</span>
        {read.run.last_saved_at ? (
          <span>Saved {formatTime(read.run.last_saved_at)}</span>
        ) : null}
        {read.run.window_number ? (
          <span>
            Window {read.run.window_number}
            {read.run.window_reason ? ` · ${read.run.window_reason}` : ""}
          </span>
        ) : null}
        {read.run.window_reset_at ? (
          <span>Reset {formatTime(read.run.window_reset_at)}</span>
        ) : null}
        {read.run.run_id ? (
          <span className="font-mono">{read.run.run_id}</span>
        ) : null}
      </div>
      {read.missing.length > 0 ? (
        <p className="py-2 px-3 border border-yellow-200/40 rounded-xs bg-yellow-500/10 text-warning text-xs">
          Missing historical data: {read.missing.join(", ")}
        </p>
      ) : null}
      {read.requests.length > 0 ? (
        <div
          className="flex flex-wrap gap-1 border-b border-line pb-3"
          role="tablist"
          aria-label="Model requests"
        >
          {read.requests.map((request) => (
            <button
              key={request.ordinal}
              type="button"
              role="tab"
              aria-selected={ordinal === request.ordinal}
              className={clsx(
                "inline-flex items-center gap-2 px-3 py-1 border rounded-sm text-xs cursor-pointer",
                ordinal === request.ordinal
                  ? "border-blue-300/60 bg-blue-500/18 text-blue-50"
                  : "border-line bg-transparent text-fg-muted hover:bg-surface-hover",
              )}
              onClick={() => onSelectRequest(request.ordinal)}
            >
              Request {request.ordinal}
              <span className="capitalize">{request.status}</span>
            </button>
          ))}
        </div>
      ) : null}
      {read.request ? (
        <RequestHistoryContent
          agentId={read.agent_id}
          sequence={read.run.sequence}
          request={read.request}
          onLoadMoreInput={onLoadMoreInput}
          onLoadMoreResponse={onLoadMoreResponse}
          onLoadMoreRelated={onLoadMoreRelated}
        />
      ) : null}
      {read.messages.messages.length > 0 ? (
        <HistoryMessagesPanel
          agentId={read.agent_id}
          sequence={read.run.sequence}
          title="Saved Turn messages"
          content={read.messages}
          onLoadMore={read.messages.has_more ? onLoadMoreRun : undefined}
        />
      ) : null}
    </>
  );
}

function RequestHistoryContent({
  agentId,
  sequence,
  request,
  onLoadMoreInput,
  onLoadMoreResponse,
  onLoadMoreRelated,
}: {
  agentId: number;
  sequence: number;
  request: NonNullable<AgentHistoryRead["request"]>;
  onLoadMoreInput: () => void;
  onLoadMoreResponse: () => void;
  onLoadMoreRelated: () => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
        <span>Request {request.summary.ordinal}</span>
        <span>
          {request.summary.streaming ? "Streaming" : "Single response"}
        </span>
        <span>{formatTime(request.summary.started_at)}</span>
        {request.summary.error ? (
          <span className="text-danger">{request.summary.error}</span>
        ) : null}
      </div>
      <HistoryMessagesPanel
        agentId={agentId}
        sequence={sequence}
        title="Model input"
        content={request.input}
        onLoadMore={request.input.has_more ? onLoadMoreInput : undefined}
      />
      <HistoryValuePanel
        title="Model request parameters"
        value={request.parameters}
      />
      <HistoryValuePanel title="Model settings" value={request.settings} />
      <HistoryValuePanel title="Model" value={request.model} />
      {request.response ? (
        <HistoryMessagesPanel
          agentId={agentId}
          sequence={sequence}
          title="Model response"
          content={request.response}
          onLoadMore={
            request.response.has_more ? onLoadMoreResponse : undefined
          }
        />
      ) : (
        <p className="text-warning text-sm">
          The model response is waiting for a saved result.
        </p>
      )}
      {request.related.messages.length > 0 ? (
        <HistoryMessagesPanel
          agentId={agentId}
          sequence={sequence}
          title="Associated tool results"
          content={request.related}
          onLoadMore={request.related.has_more ? onLoadMoreRelated : undefined}
        />
      ) : null}
    </div>
  );
}

function HistoryMessagesPanel({
  agentId,
  sequence,
  title,
  content,
  onLoadMore,
}: {
  agentId: number;
  sequence: number;
  title: string;
  content: HistoryMessages;
  onLoadMore?: () => void;
}) {
  return (
    <section className="flex flex-col gap-2">
      <h3 className="m-0 text-sm font-semibold tracking-title">{title}</h3>
      <div className="flex flex-col gap-2">
        {content.messages.map((message, index) => (
          <HistoryValueView
            key={`${content.offset + index}`}
            value={message}
            label={`Message ${content.offset + index + 1}`}
            agentId={agentId}
            sequence={sequence}
          />
        ))}
      </div>
      {onLoadMore ? (
        <Button size="sm" onClick={onLoadMore}>
          Load more messages
        </Button>
      ) : null}
    </section>
  );
}

function HistoryValuePanel({
  title,
  value,
}: {
  title: string;
  value: HistoryValue;
}) {
  return (
    <section className="flex flex-col gap-2">
      <h3 className="m-0 text-sm font-semibold tracking-title">{title}</h3>
      <HistoryValueView value={value} label={title} />
    </section>
  );
}

function isRecord(
  value: HistoryValue,
): value is { [key: string]: HistoryValue } {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function binaryValue(value: HistoryValue): HistoryBinary | null {
  if (!isRecord(value) || value.kind !== "binary") return null;
  if (
    typeof value.source !== "string" ||
    typeof value.message_index !== "number" ||
    !Array.isArray(value.path) ||
    typeof value.media_type !== "string" ||
    typeof value.size !== "number"
  )
    return null;
  return value as unknown as HistoryBinary;
}

function textValue(value: HistoryValue): HistoryText | null {
  if (!isRecord(value) || value.kind !== "text") return null;
  if (
    typeof value.source !== "string" ||
    typeof value.message_index !== "number" ||
    !Array.isArray(value.path) ||
    typeof value.offset !== "number" ||
    typeof value.next_offset !== "number" ||
    typeof value.total_bytes !== "number" ||
    typeof value.value !== "string" ||
    typeof value.has_more !== "boolean"
  )
    return null;
  return value as unknown as HistoryText;
}

function HistoryValueView({
  value,
  label,
  agentId,
  sequence,
}: {
  value: HistoryValue;
  label: string;
  agentId?: number;
  sequence?: number;
}) {
  const binary = binaryValue(value);
  const text = textValue(value);
  if (text && agentId !== undefined && sequence !== undefined) {
    return (
      <HistoryTextPreview
        agentId={agentId}
        sequence={sequence}
        reference={text}
        label={label}
      />
    );
  }
  if (binary && agentId !== undefined && sequence !== undefined) {
    return (
      <HistoryImagePreview
        agentId={agentId}
        sequence={sequence}
        reference={binary}
      />
    );
  }
  if (Array.isArray(value)) {
    return (
      <details open className="border border-line rounded-sm bg-surface">
        <summary className="cursor-pointer px-3 py-2 text-xs font-semibold">
          {label}
        </summary>
        <div className="flex flex-col gap-2 p-3 border-t border-line">
          {value.map((child, index) => (
            <HistoryValueView
              key={`${label}-${index}`}
              value={child}
              label={`${label} ${index + 1}`}
              agentId={agentId}
              sequence={sequence}
            />
          ))}
        </div>
      </details>
    );
  }
  if (isRecord(value)) {
    return (
      <details open className="border border-line rounded-sm bg-surface">
        <summary className="cursor-pointer px-3 py-2 text-xs font-semibold">
          {label}
        </summary>
        <div className="flex flex-col gap-2 p-3 border-t border-line">
          {Object.entries(value).map(([key, child]) => (
            <HistoryValueView
              key={`${label}-${key}`}
              value={child}
              label={key}
              agentId={agentId}
              sequence={sequence}
            />
          ))}
        </div>
      </details>
    );
  }
  return (
    <div className="flex flex-col gap-1 py-2 px-3 border border-line rounded-sm bg-surface">
      <span className="text-xs text-fg-muted">{label}</span>
      <pre className="m-0 whitespace-pre-wrap wrap-anywhere text-xs leading-body font-mono tracking-[0]">
        {value === null ? "null" : String(value)}
      </pre>
    </div>
  );
}

function HistoryTextPreview({
  agentId,
  sequence,
  reference,
  label,
}: {
  agentId: number;
  sequence: number;
  reference: HistoryText;
  label: string;
}) {
  const [text, setText] = useState(reference.value);
  const [nextOffset, setNextOffset] = useState(reference.next_offset);
  const [hasMore, setHasMore] = useState(reference.has_more);
  const [error, setError] = useState<string | null>(null);

  const loadMore = async () => {
    try {
      const result = await backend.agentHistoryText(agentId, sequence, {
        ...reference,
        offset: nextOffset,
      });
      setText((current) => current + result.value);
      setNextOffset(result.next_offset);
      setHasMore(result.has_more);
      setError(null);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  };

  return (
    <div className="flex flex-col gap-2 py-2 px-3 border border-line rounded-sm bg-surface">
      <span className="text-xs text-fg-muted">
        {label} · {formatBytes(reference.total_bytes)}
      </span>
      <pre className="m-0 max-h-64 overflow-auto whitespace-pre-wrap wrap-anywhere text-xs leading-body font-mono tracking-[0]">
        {text}
      </pre>
      {error ? <p className="m-0 text-danger text-xs">{error}</p> : null}
      {hasMore ? (
        <Button size="sm" onClick={() => void loadMore()}>
          Load more text
        </Button>
      ) : null}
    </div>
  );
}

function HistoryImagePreview({
  agentId,
  sequence,
  reference,
}: {
  agentId: number;
  sequence: number;
  reference: HistoryBinary;
}) {
  const [image, setImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState(false);
  const pathKey = reference.path.map(String).join("/");

  useEffect(() => {
    let active = true;
    void backend
      .agentHistoryImage(agentId, sequence, reference)
      .then((result) => {
        if (active) setImage(`data:${result.media_type};base64,${result.data}`);
      })
      .catch((failure: unknown) => {
        if (active)
          setError(
            failure instanceof Error ? failure.message : String(failure),
          );
      });
    return () => {
      active = false;
    };
  }, [agentId, pathKey, reference, sequence]);

  if (error) {
    return <p className="text-danger text-xs">Could not load image: {error}</p>;
  }
  if (!image) {
    return (
      <span className="text-fg-muted text-xs">
        Loading image ({formatBytes(reference.size)})
      </span>
    );
  }
  return (
    <>
      <button
        type="button"
        className="relative self-start max-w-full overflow-hidden border border-line rounded-sm bg-surface cursor-zoom-in"
        aria-label="Open history image"
        onClick={() => setZoom(true)}
      >
        <img
          src={image}
          alt="Saved model content"
          className="block max-h-64 max-w-full object-contain"
        />
        <span className="absolute right-2 bottom-2 inline-flex items-center justify-center p-1 rounded-xs bg-gray-1100/70 text-white">
          <Maximize2 size={14} aria-hidden="true" />
        </span>
      </button>
      <Modal
        open={zoom}
        onOpenChange={setZoom}
        title="History image"
        wide
        footer={<Button onClick={() => setZoom(false)}>Close</Button>}
      >
        <img
          src={image}
          alt="Saved model content"
          className="max-h-[70vh] max-w-full self-center object-contain"
        />
      </Modal>
    </>
  );
}

function TurnCard({
  run,
  onOpen,
}: {
  run: AgentRun | AgentHistoryRun;
  onOpen?: () => void;
}) {
  const [open, setOpen] = useState(run.status === "failed");
  const effects = "effects" in run ? run.effects : [];
  const tools = [...new Set(effects.map((effect) => effect.tool))];
  const requestCount = run.request_count ?? 0;

  return (
    <li className="border border-line rounded-sm bg-gray-800/50 overflow-hidden transition-[border-color] duration-(--duration-fast) ease-linear hover:border-line-interactive">
      <button
        type="button"
        className="flex items-center gap-3 w-full py-3 px-4 border-0 bg-transparent text-inherit text-left cursor-pointer"
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        <ChevronDown
          className={clsx(
            "flex-none text-fg-muted transition-[rotate] duration-(--duration-base) ease-transform",
            !open && "-rotate-90",
          )}
          size={14}
          aria-hidden="true"
        />
        <span className="flex-none text-fg-muted font-mono text-xs tracking-[0]">
          #{run.sequence}
        </span>
        <StatusText dot={<Dot tone={runTone(run.status)} />}>
          <span className="capitalize">{run.status}</span>
        </StatusText>
        <span className="flex-1 min-w-0 truncate text-xs text-fg-muted">
          {requestCount > 0 ? (
            `${plural(requestCount, "model request")} · ${effects.length} ${plural(effects.length, "effect")}`
          ) : effects.length === 0 ? (
            <span className="text-warning">
              {run.status === "running"
                ? "No saved progress yet"
                : "Saved messages only"}
            </span>
          ) : (
            `${plural(effects.length, "effect")} · ${tools.join(", ")}`
          )}
        </span>
        <Tooltip label={formatTime(run.started_at)}>
          <time
            className="flex-none text-xs text-fg-muted"
            dateTime={run.started_at}
          >
            {relativeTime(run.started_at)}
          </time>
        </Tooltip>
      </button>
      {open ? (
        <div className="flex flex-col gap-3 pt-0 pr-4 pb-4 pl-[46px] animate-rise-in [animation-duration:var(--duration-slow)]">
          {run.error ? (
            <p className="flex items-center gap-2 py-2 px-3 border border-red-300/40 rounded-xs bg-red-500/15 text-red-100 text-xs font-mono tracking-[0]">
              <CircleAlert size={14} aria-hidden="true" />
              {run.error}
            </p>
          ) : null}
          {effects.length === 0 ? (
            <StatePanel
              compact
              icons={[Terminal, Gauge, FileText]}
              title={
                run.status === "running"
                  ? "No saved progress yet"
                  : "No effects recorded"
              }
            />
          ) : (
            <ul className="flex flex-col gap-1 -ml-1 border-l border-line pl-4">
              {effects.map((effect) => (
                <li
                  className="flex items-baseline gap-3 min-w-0"
                  key={`${run.sequence}-${effect.ordinal}`}
                >
                  <span className="inline-flex items-center gap-[5px] flex-none min-w-29 text-blue-100 font-mono text-xs tracking-[0]">
                    {toolIcon(effect.tool)}
                    {effect.tool}
                  </span>
                  <span className="flex-1 min-w-0 text-xs truncate text-fg-muted">
                    {effect.summary}
                  </span>
                </li>
              ))}
            </ul>
          )}
          {run.last_saved_at ? (
            <p className="text-fg-muted text-xs">
              Saved {formatTime(run.last_saved_at)}
            </p>
          ) : null}
          {run.completed_at ? (
            <p className="text-fg-muted text-xs">
              Finished {formatTime(run.completed_at)}
            </p>
          ) : null}
          {onOpen ? (
            <Button size="sm" onClick={onOpen}>
              <FileText size={13} />
              View full context
            </Button>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}
