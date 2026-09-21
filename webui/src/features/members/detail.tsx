import { clsx } from "clsx";
import {
  Check,
  ChevronDown,
  CircleAlert,
  FileText,
  Gauge,
  Pause,
  PencilLine,
  Play,
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
  type AgentRun,
  backend,
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

            <Section title="Recent Turns">
              {detail.runs.length === 0 ? (
                <StatePanel
                  compact
                  icons={[Terminal, Play, Gauge]}
                  title="No Turns yet"
                  description="This Agent's Turns will appear here."
                />
              ) : (
                <ul className="flex flex-col gap-2">
                  {detail.runs.slice(0, 10).map((run) => (
                    <TurnCard key={run.sequence} run={run} />
                  ))}
                </ul>
              )}
            </Section>
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

function TurnCard({ run }: { run: AgentRun }) {
  const [open, setOpen] = useState(run.status === "failed");
  const tools = [...new Set(run.effects.map((effect) => effect.tool))];

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
          {run.effects.length === 0 ? (
            <span className="text-warning">
              {run.status === "running" ? "No effects yet" : "Produced nothing"}
            </span>
          ) : (
            `${plural(run.effects.length, "effect")} · ${tools.join(", ")}`
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
          {run.effects.length === 0 ? (
            <StatePanel
              compact
              icons={[Terminal, Gauge, FileText]}
              title={
                run.status === "running"
                  ? "No effects yet"
                  : "Nothing produced."
              }
            />
          ) : (
            <ul className="flex flex-col gap-1 -ml-1 border-l border-line pl-4">
              {run.effects.map((effect) => (
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
          {run.completed_at ? (
            <p className="text-fg-muted text-xs">
              Finished {formatTime(run.completed_at)}
            </p>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}
