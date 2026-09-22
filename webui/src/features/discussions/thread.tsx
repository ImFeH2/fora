import { useVirtualizer } from "@tanstack/react-virtual";
import { clsx } from "clsx";
import {
  Archive,
  ArchiveRestore,
  Check,
  MessageCircle,
  MessagesSquare,
  Send,
  Users,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useOrganization } from "@/app/organization";
import { useNavigate } from "@/app/router";
import { Page, PageBody, PageHeader } from "@/components/layout/shell";
import {
  Avatar,
  AvatarStack,
  Button,
  Chip,
  toast,
} from "@/components/ui/index";
import { OverflowMenu } from "@/components/ui/menu";
import { StatePanel } from "@/components/ui/state-panel";
import { Tooltip } from "@/components/ui/tooltip";
import { MessageAttachment } from "@/features/discussions/attachments";
import { Composer } from "@/features/discussions/composer";
import { DiscussionMembersDialog } from "@/features/discussions/members";
import { useThreadData } from "@/features/discussions/thread-data";
import {
  centeredScrollTop,
  shouldStopPositioningForKey,
} from "@/features/discussions/thread-position";
import { renderMentions } from "@/features/mentions";
import { BackendError, backend, type Message } from "@/lib/backend";
import { formatTime, relativeTime } from "@/lib/format";

export function ThreadPage({ id }: { id: number }) {
  return <ThreadSession key={id} id={id} />;
}

function ThreadSession({ id }: { id: number }) {
  const { members, humanId, discussions, refresh } = useOrganization();
  const navigate = useNavigate();
  const thread = useThreadData(id, humanId);
  const { data: detail, missing, failed: loadFailed } = thread;
  const [busy, setBusy] = useState(false);
  const [ackBusy, setAckBusy] = useState(false);
  const [editingMembers, setEditingMembers] = useState(false);
  const [composerSize, setComposerSize] = useState(48);
  const [lastVisible, setLastVisible] = useState(0);
  const scroll = useRef<HTMLDivElement>(null);
  const ready = useRef(false);
  const preserveBottom = useRef(false);
  const positioning = useRef(false);
  const positioningInterrupted = useRef(false);
  const positionStableFrames = useRef(0);
  const frame = useRef(0);
  const messages = detail?.messages ?? [];
  const virtual = useVirtualizer({
    count: messages.length,
    getScrollElement: () => scroll.current,
    estimateSize: () => 96,
    getItemKey: useCallback((index: number) => messages[index].id, [messages]),
    overscan: 5,
    anchorTo: "end",
    followOnAppend: detail !== null && !detail.hasAfter,
    paddingStart: 16,
    paddingEnd: composerSize + 40,
    scrollEndThreshold: 1,
  });
  const resizeComposer = useCallback(
    (height: number) => {
      preserveBottom.current = ready.current && virtual.isAtEnd();
      setComposerSize(height);
    },
    [virtual],
  );
  const items = virtual.getVirtualItems();
  const memberIds = useMemo(
    () => new Set((detail?.members ?? []).map((member) => member.id)),
    [detail?.members],
  );

  const sample = useCallback(
    (paginate = false) => {
      const root = scroll.current;
      const current = thread.current.current;
      if (!root || !ready.current || !current) return;
      const bounds = root.getBoundingClientRect();
      const end = bounds.bottom - composerSize - 16;
      const visible: number[] = [];
      for (const element of root.querySelectorAll<HTMLElement>(
        "[data-message-id]",
      )) {
        const rect = element.getBoundingClientRect();
        if (rect.bottom > bounds.top && rect.top < end)
          visible.push(Number(element.dataset.messageId));
      }
      setLastVisible(visible.length ? Math.max(...visible) : 0);
      thread.view(visible, virtual.isAtEnd());
      if (
        document.visibilityState === "visible" &&
        visible.length &&
        !backend.disconnected &&
        !thread.loading
      )
        void thread.markRead(Math.max(...visible)).catch(() => {});
      if (!thread.failed && !thread.loading) {
        if (
          current.hasBefore &&
          root.clientHeight > 0 &&
          (virtual.getTotalSize() <= root.clientHeight ||
            (paginate && root.scrollTop < root.clientHeight))
        )
          void thread.request("before");
        else if (
          paginate &&
          root.scrollHeight - root.scrollTop - root.clientHeight <
            root.clientHeight &&
          current.hasAfter
        )
          void thread.request("after");
      }
    },
    [composerSize, thread, virtual],
  );

  const stopPositioning = useCallback(() => {
    if (!positioning.current) return;
    positioning.current = false;
    positioningInterrupted.current = true;
    cancelAnimationFrame(frame.current);
    ready.current = true;
    thread.positioned();
    const root = scroll.current;
    if (root) virtual.scrollToOffset(root.scrollTop);
    frame.current = requestAnimationFrame(() => {
      const currentRoot = scroll.current;
      if (currentRoot) virtual.scrollToOffset(currentRoot.scrollTop);
      positioningInterrupted.current = false;
    });
  }, [thread.positioned, virtual]);

  useEffect(() => {
    if (!thread.position || !detail || detail.divider === null) return;
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof Element &&
        scroll.current?.contains(target) &&
        shouldStopPositioningForKey(event.key, target)
      )
        stopPositioning();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [thread.position, detail, stopPositioning]);

  useLayoutEffect(() => {
    if (!thread.position || !detail || !scroll.current) return;
    ready.current = false;
    positioning.current = false;
    if (thread.position === "latest" || detail.divider === null) {
      virtual.scrollToEnd();
      frame.current = requestAnimationFrame(() => {
        ready.current = true;
        thread.positioned();
      });
      return () => cancelAnimationFrame(frame.current);
    }

    const index = messages.findIndex(
      (message) => message.id === detail.divider,
    );
    if (index < 0) throw new Error("Entry divider is missing from its page");
    positioning.current = true;
    positionStableFrames.current = 0;
    let missingMarkerFrames = 0;
    virtual.scrollToIndex(index, { align: "center" });

    const positionNewMarker = () => {
      if (!positioning.current) return;
      const root = scroll.current;
      if (!root) return;
      const bounds = root.getBoundingClientRect();
      if (root.clientHeight === 0 || bounds.height === 0) {
        frame.current = requestAnimationFrame(positionNewMarker);
        return;
      }
      const marker = root.querySelector<HTMLElement>("[data-new-divider]");
      if (!marker) {
        missingMarkerFrames += 1;
        if (missingMarkerFrames > 10)
          throw new Error("New divider did not render after entry positioning");
        virtual.scrollToIndex(index, { align: "center" });
        frame.current = requestAnimationFrame(positionNewMarker);
        return;
      }
      missingMarkerFrames = 0;
      const target = centeredScrollTop(
        root.scrollTop,
        root.scrollHeight - root.clientHeight,
        marker.getBoundingClientRect().top,
        bounds.top,
        bounds.bottom - composerSize - 16,
      );
      const needsCorrection = Math.abs(target - root.scrollTop) >= 0.5;
      virtual.scrollToOffset(target);
      if (needsCorrection) positionStableFrames.current = 0;
      else positionStableFrames.current += 1;
      if (positionStableFrames.current >= 2) {
        virtual.scrollToOffset(root.scrollTop);
        positioning.current = false;
        ready.current = true;
        thread.positioned();
        sample();
      } else {
        frame.current = requestAnimationFrame(positionNewMarker);
      }
    };
    frame.current = requestAnimationFrame(positionNewMarker);
    return () => cancelAnimationFrame(frame.current);
  }, [
    thread.position,
    detail?.divider,
    messages,
    virtual,
    thread.positioned,
    composerSize,
  ]);

  useEffect(() => {
    const tick = requestAnimationFrame(() => sample());
    const changed = () => sample();
    const observer = new ResizeObserver(changed);
    if (scroll.current) observer.observe(scroll.current);
    document.addEventListener("visibilitychange", changed);
    return () => {
      cancelAnimationFrame(tick);
      observer.disconnect();
      document.removeEventListener("visibilitychange", changed);
    };
  }, [sample, items]);

  useLayoutEffect(() => {
    if (ready.current && preserveBottom.current) virtual.scrollToEnd();
  }, [composerSize, virtual]);

  const load = thread.invalidate;

  const send = async (
    body: string,
    attachmentIds: string[],
    clientMessageId: string,
  ) => {
    setBusy(true);
    let sent = false;
    try {
      await backend.sendVisible(id, body, attachmentIds, clientMessageId);
      sent = true;
      if (!thread.live.current) return true;
      await thread.request("after");
      return true;
    } catch (failure) {
      if (failure instanceof BackendError && failure.code === "send_cancelled")
        return "cancelled" as const;
      if (sent) {
        if (thread.live.current) backend.reportFailure(failure);
        return true;
      }
      if (
        thread.live.current &&
        !(failure instanceof BackendError && failure.transport)
      ) {
        toast({
          tone: "danger",
          title: "Could not send",
          description:
            failure instanceof Error ? failure.message : String(failure),
        });
      }
      return false;
    } finally {
      if (thread.live.current) setBusy(false);
    }
  };

  const ack = async (messageIds: number[], revoke = false) => {
    if (ackBusy) return;
    setAckBusy(true);
    try {
      if (revoke) await backend.revokeAck(id, messageIds);
      else {
        await thread.markRead(Math.max(...messageIds));
        if (!thread.live.current) return;
        await backend.ack(id, messageIds);
      }
      if (thread.live.current) await load();
    } catch (failure) {
      if (!thread.live.current) return;
      toast({
        tone: "danger",
        title: revoke ? "Could not undo" : "Could not mark handled",
        description:
          failure instanceof Error ? failure.message : String(failure),
      });
    } finally {
      if (thread.live.current) setAckBusy(false);
    }
  };

  const ackAll = async () => {
    if (ackBusy || !detail) return;
    setAckBusy(true);
    try {
      await backend.ackPending(id, detail.latest);
      if (thread.live.current) await load();
    } catch (failure) {
      if (thread.live.current) backend.reportFailure(failure);
    } finally {
      if (thread.live.current) setAckBusy(false);
    }
  };

  const archive = async (archived: boolean) => {
    try {
      await backend.archiveDiscussion(id, archived);
      if (!thread.live.current) return;
      if (archived) navigate({ name: "discussions" });
      else await load();
    } catch (failure) {
      if (thread.live.current) backend.reportFailure(failure);
    }
  };

  const names = (detail?.members ?? []).map((member) => member.name);
  const topic =
    discussions.find((item) => item.id === id)?.topic ?? detail?.topic ?? "";

  return (
    <Page>
      <PageHeader
        title={
          missing
            ? "Discussion not found"
            : topic || (loadFailed ? "Unavailable" : "")
        }
        status={detail?.archived ? <Chip>Archived</Chip> : undefined}
        actions={
          missing ? (
            <Button onClick={() => navigate({ name: "discussions" })}>
              Back to Discussions
            </Button>
          ) : detail ? (
            <>
              <Tooltip label={names.join("\n")}>
                <Button
                  variant="ghost"
                  aria-label="Members"
                  onClick={() => setEditingMembers(true)}
                >
                  <AvatarStack members={detail.members} />
                </Button>
              </Tooltip>
              {detail.pendingCount > 0 ? (
                <Button
                  variant="primary"
                  disabled={ackBusy}
                  onClick={() => void ackAll()}
                >
                  <Check size={16} />
                  Mark all handled
                </Button>
              ) : null}
              <OverflowMenu
                label="Discussion actions"
                actions={[
                  {
                    id: "members",
                    label: "Members",
                    icon: <Users size={15} />,
                    onSelect: () => setEditingMembers(true),
                  },
                  detail.archived
                    ? {
                        id: "unarchive",
                        label: "Unarchive",
                        icon: <ArchiveRestore size={15} />,
                        onSelect: () => void archive(false),
                      }
                    : {
                        id: "archive",
                        label: "Archive",
                        icon: <Archive size={15} />,
                        onSelect: () => void archive(true),
                      },
                ]}
              />
            </>
          ) : undefined
        }
      />

      <PageBody variant="flush">
        <div className="@container relative flex min-h-0 flex-1 flex-col">
          <div
            ref={scroll}
            className="min-h-0 flex-1 overflow-y-auto border-t border-line px-8 max-[940px]:px-6 [overflow-anchor:none]"
            onScroll={() => {
              if (positioningInterrupted.current && scroll.current)
                virtual.scrollToOffset(scroll.current.scrollTop);
              sample(true);
            }}
            onWheel={stopPositioning}
            onTouchStart={stopPositioning}
            onPointerDown={stopPositioning}
            aria-busy={thread.loading}
          >
            {detail &&
            messages.length === 0 &&
            !thread.loading &&
            !loadFailed ? (
              <div
                className="flex min-h-full flex-col"
                style={{ paddingBottom: composerSize + 40 }}
              >
                <StatePanel
                  icons={[MessageCircle, MessagesSquare, Send]}
                  title="No messages yet"
                  description="Write the first message to start this Discussion."
                  action={{
                    label: "Write a message",
                    icon: <Send size={16} />,
                    onClick: () => {
                      const input =
                        scroll.current?.parentElement?.querySelector<HTMLTextAreaElement>(
                          'textarea[aria-label="Message"]',
                        );
                      if (!input) throw new Error("Message input is missing");
                      input.focus();
                    },
                  }}
                />
              </div>
            ) : null}
            <ol
              className="relative"
              hidden={messages.length === 0}
              style={{ height: virtual.getTotalSize() }}
            >
              {items.map((item) => {
                const message = messages[item.index];
                const previousSender =
                  item.index === 0
                    ? detail?.previousSender
                    : messages[item.index - 1].sender_id;
                const divider = message.id === detail?.divider;
                const compact =
                  !divider && previousSender === message.sender_id;
                return (
                  <li
                    key={item.key}
                    ref={virtual.measureElement}
                    data-index={item.index}
                    className="absolute top-0 left-0 w-full"
                    style={{
                      transform: `translateY(${item.start}px)`,
                      paddingTop: compact ? 4 : 16,
                    }}
                  >
                    {divider ? (
                      <div
                        data-new-divider
                        className="mb-4 flex items-center gap-3 text-xs font-medium uppercase tracking-caps text-primary before:content-[''] before:h-px before:flex-1 before:bg-blue-500/40 after:content-[''] after:h-px after:flex-1 after:bg-blue-500/40"
                      >
                        <span>New</span>
                      </div>
                    ) : null}
                    <MessageRow
                      discussionId={id}
                      message={message}
                      compact={compact}
                      fresh={false}
                      pending={message.pending}
                      acknowledged={message.acknowledged}
                      busy={ackBusy}
                      onAck={() => void ack([message.id])}
                      onRevoke={() => void ack([message.id], true)}
                    />
                  </li>
                );
              })}
            </ol>
          </div>
          {detail && detail.latest > lastVisible ? (
            <div
              className="absolute right-8"
              style={{ bottom: composerSize + 24 }}
            >
              <Button
                disabled={thread.loading}
                onClick={() => {
                  void thread.request("latest");
                }}
              >
                New messages
              </Button>
            </div>
          ) : null}

          <Composer
            discussionId={id}
            members={members}
            memberIds={memberIds}
            busy={busy}
            placeholder={topic ? `Message ${topic}` : "Message"}
            onSend={send}
            onHeightChange={resizeComposer}
          />
        </div>
      </PageBody>

      {editingMembers && detail ? (
        <DiscussionMembersDialog
          discussionId={id}
          memberIds={detail.members.map((member) => member.id)}
          onClose={() => setEditingMembers(false)}
          onSaved={async (ids) => {
            await refresh();
            if (!thread.live.current) return;
            if (ids.includes(humanId)) await load();
            else navigate({ name: "discussions" });
          }}
        />
      ) : null}
    </Page>
  );
}

export function MessageRow({
  discussionId,
  message,
  compact,
  fresh,
  pending,
  acknowledged,
  busy,
  onAck,
  onRevoke,
}: {
  discussionId: number;
  message: Message;
  compact: boolean;
  fresh: boolean;
  pending: boolean;
  acknowledged: boolean;
  busy: boolean;
  onAck: () => void;
  onRevoke: () => void;
}) {
  const { members } = useOrganization();
  return (
    <div
      data-message-id={message.id}
      className={clsx(
        "flex scroll-m-6 items-start gap-3",
        fresh && "animate-rise-in [animation-duration:var(--duration-slow)]",
      )}
    >
      <div className="w-[26px] flex-none">
        {compact ? null : <Avatar memberId={message.sender_id} />}
      </div>
      <div
        className={clsx(
          "-ml-3 min-w-0 flex-1 border-l-2 pl-3",
          pending ? "border-l-yellow-300" : "border-l-transparent",
        )}
      >
        {compact ? null : (
          <div className="mb-0.5 flex items-baseline gap-2">
            <span className="font-semibold">{message.sender_name}</span>
            <Tooltip label={formatTime(message.created_at)}>
              <time
                className="text-xs text-fg-muted"
                dateTime={message.created_at}
              >
                {relativeTime(message.created_at)}
              </time>
            </Tooltip>
          </div>
        )}
        <div className="whitespace-pre-wrap wrap-anywhere [&_mark]:bg-blue-500/25 [&_mark]:text-blue-100 [&_mark]:rounded-xs [&_mark]:px-[3px] [&_mark]:font-medium">
          {renderMentions(message.body, message.mentions, members)}
        </div>
        {message.attachments.map((attachment) => (
          <MessageAttachment
            key={attachment.id}
            discussionId={discussionId}
            messageId={message.id}
            attachment={attachment}
          />
        ))}
        {pending ? (
          <div className="mt-2 flex items-center gap-2">
            <Chip tone="warning">Mentions you</Chip>
            <Button size="sm" disabled={busy} onClick={onAck}>
              <Check size={13} />
              Mark handled
            </Button>
          </div>
        ) : acknowledged ? (
          <div className="mt-2 flex items-center gap-2">
            <Chip tone="success">Handled</Chip>
            <Button
              size="sm"
              disabled={busy}
              aria-label="Undo confirmation"
              onClick={onRevoke}
            >
              Undo
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
