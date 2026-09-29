import { useCallback, useEffect, useRef, useState } from "react";
import { dismissToast, toast } from "@/components/ui/toast";
import {
  BackendError,
  type BackendEvent,
  backend,
  type DiscussionPage,
  type Message,
  type PageRequest,
} from "@/lib/backend";

export const PAGE_SIZE = 50;
export type ThreadMessage = Message & {
  pending: boolean;
  acknowledged: boolean;
};
export type ThreadData = {
  id: number;
  topic: string;
  archived: boolean;
  members: { id: number; name: string }[];
  messages: ThreadMessage[];
  divider: number | null;
  latest: number;
  readThrough: number;
  pendingCount: number;
  hasBefore: boolean;
  hasAfter: boolean;
  previousSender: number | null;
};
export type PageMode = "entry" | "latest" | "before" | "after" | "refresh";

export function eventDiscussion(event: BackendEvent): number | undefined {
  const value =
    event.type === "discussion.updated" ? event.id : event.discussion_id;
  return typeof value === "number" ? value : undefined;
}

export function mergePage(
  current: ThreadData | null,
  page: DiscussionPage,
  mode: PageMode,
): ThreadData {
  if (current && current.id !== page.id)
    throw new Error("Discussion page belongs to another thread");
  const pending = new Set(page.awaiting_ack);
  const acknowledged = new Set(page.acknowledged);
  const incoming = page.messages.map((message) => ({
    ...message,
    pending: pending.has(message.id),
    acknowledged: acknowledged.has(message.id),
  }));
  const reset = mode === "entry" || mode === "latest";
  const messages = new Map(
    (reset ? [] : (current?.messages ?? [])).map((message) => [
      message.id,
      message,
    ]),
  );
  for (const message of incoming) {
    const previous = messages.get(message.id);
    if (
      previous &&
      (previous.body !== message.body ||
        previous.sender_id !== message.sender_id ||
        JSON.stringify(previous.attachments) !==
          JSON.stringify(message.attachments))
    )
      throw new Error("Immutable message changed");
    messages.set(
      message.id,
      previous ? { ...message, attachments: previous.attachments } : message,
    );
  }
  const metadata = page.metadata ?? current;
  if (!metadata) throw new Error("Initial discussion page is missing metadata");
  const sorted = [...messages.values()].sort((a, b) => a.id - b.id);
  return {
    id: page.id,
    topic: metadata.topic,
    archived: metadata.archived,
    members: metadata.members,
    messages: sorted,
    divider:
      mode === "entry"
        ? (page.first_unread_id ?? null)
        : (current?.divider ?? null),
    latest: Math.max(current?.latest ?? 0, page.latest_id),
    readThrough: Math.max(current?.readThrough ?? 0, page.read_through),
    pendingCount: page.pending_count,
    hasBefore:
      reset || mode === "before"
        ? page.has_before
        : (current?.hasBefore ?? false),
    hasAfter: (sorted[sorted.length - 1]?.id ?? 0) < page.latest_id,
    previousSender:
      reset || mode === "before"
        ? page.previous_sender_id
        : (current?.previousSender ?? null),
  };
}

export function useThreadData(id: number, memberId: number) {
  const [data, setData] = useState<ThreadData | null>(null);
  const [missing, setMissing] = useState(false);
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(false);
  const [position, setPosition] = useState<"entry" | "latest" | null>(null);
  const current = useRef(data);
  const live = useRef(false);
  const generation = useRef(0);
  const inFlight = useRef(false);
  const dirty = useRef(false);
  const latest = useRef(0);
  const visible = useRef<number[]>([]);
  const fresh = useRef(new Set<number>());
  const following = useRef(false);
  const reading = useRef<Promise<void> | null>(null);
  const readingTarget = useRef(0);
  const readingFailed = useRef(false);
  const blocked = useRef(false);
  const requestRef = useRef<(mode: PageMode) => Promise<void>>(async () => {});
  const commit = useCallback((next: ThreadData) => {
    current.current = next;
    setData(next);
  }, []);

  const request = useCallback(
    async (mode: PageMode) => {
      if (!live.current || inFlight.current) return;
      const snapshot = current.current;
      if (mode !== "entry" && mode !== "latest" && !snapshot) return;
      const page: PageRequest = { limit: PAGE_SIZE };
      if (mode === "entry") page.entry = true;
      if (mode === "latest") page.metadata = true;
      if (mode === "before") {
        if (!snapshot?.hasBefore || !snapshot.messages.length) return;
        page.before = snapshot.messages[0].id;
      }
      if (mode === "after") {
        if (
          !snapshot ||
          (snapshot.messages[snapshot.messages.length - 1]?.id ?? 0) >=
            latest.current
        )
          return;
        page.after = snapshot.messages[snapshot.messages.length - 1]?.id ?? 0;
      }
      if (mode === "refresh") {
        const target =
          visible.current.find((message) => !fresh.current.has(message)) ??
          visible.current[0] ??
          snapshot?.messages[0]?.id;
        page.after = target === undefined ? 0 : target - 1;
        page.metadata = true;
      }
      const epoch = generation.current;
      inFlight.current = true;
      let succeeded = false;
      setLoading(true);
      try {
        const result = await backend.discussionPage(id, page);
        if (!live.current || epoch !== generation.current) return;
        latest.current = Math.max(latest.current, result.latest_id);
        const next = mergePage(current.current, result, mode);
        next.latest = latest.current;
        next.hasAfter =
          (next.messages[next.messages.length - 1]?.id ?? 0) < latest.current;
        commit(next);
        if (!dirty.current)
          for (const message of result.messages) fresh.current.add(message.id);
        if (mode === "entry" || mode === "latest") setPosition(mode);
        setMissing(false);
        setFailed(false);
        blocked.current = false;
        succeeded = true;
        dismissToast(`thread-${id}-page`);
      } catch (error) {
        if (!live.current || epoch !== generation.current) return;
        if (
          error instanceof BackendError &&
          ["not_found", "not_a_member"].includes(error.code)
        ) {
          setMissing(true);
          current.current = null;
          setData(null);
        } else {
          setFailed(true);
          blocked.current = true;
          if (!(error instanceof BackendError && error.transport))
            toast({
              id: `thread-${id}-page`,
              tone: "danger",
              title: "Could not load messages",
              description:
                error instanceof Error ? error.message : String(error),
              duration: null,
              action: {
                label: "Retry",
                onClick: () => void requestRef.current(mode),
              },
            });
        }
      } finally {
        if (live.current && epoch === generation.current) {
          inFlight.current = false;
          setLoading(false);
          const refresh = dirty.current && succeeded;
          dirty.current = false;
          if (refresh)
            void requestRef.current(current.current ? "refresh" : "entry");
        }
      }
    },
    [id, commit],
  );
  requestRef.current = request;

  const markRead = useCallback(
    async (messageId: number, retry = false): Promise<void> => {
      if (!live.current || !current.current) return;
      if (messageId <= current.current.readThrough) {
        if (readingTarget.current <= current.current.readThrough) {
          readingFailed.current = false;
          dismissToast(`thread-${id}-read`);
        }
        return;
      }
      readingTarget.current = Math.max(readingTarget.current, messageId);
      if (readingFailed.current && !retry)
        throw new BackendError(
          "reading_position_blocked",
          "Retry saving the reading position before marking handled",
        );
      if (reading.current) return reading.current;
      const epoch = generation.current;
      readingFailed.current = false;
      const operation = async () => {
        while (
          live.current &&
          epoch === generation.current &&
          current.current &&
          readingTarget.current > current.current.readThrough
        ) {
          const target = readingTarget.current;
          const result = await backend.markRead(id, target);
          if (!live.current || epoch !== generation.current || !current.current)
            return;
          commit({
            ...current.current,
            readThrough: Math.max(
              current.current.readThrough,
              result.read_through,
            ),
          });
        }
        dismissToast(`thread-${id}-read`);
      };
      const promise = operation()
        .catch((error: unknown) => {
          if (!live.current || epoch !== generation.current) return;
          readingFailed.current = true;
          if (!(error instanceof BackendError && error.transport))
            toast({
              id: `thread-${id}-read`,
              tone: "danger",
              title: "Could not save reading position",
              description:
                error instanceof Error ? error.message : String(error),
              duration: null,
              action: {
                label: "Retry",
                onClick: () =>
                  void markRead(readingTarget.current, true).catch(() => {}),
              },
            });
          throw new BackendError(
            "reading_position_failed",
            error instanceof Error ? error.message : String(error),
            error instanceof BackendError && error.transport,
          );
        })
        .finally(() => {
          if (epoch === generation.current) reading.current = null;
        });
      reading.current = promise;
      return promise;
    },
    [id, commit],
  );

  useEffect(() => {
    live.current = true;
    generation.current += 1;
    const off = backend.onEvent((event) => {
      if (event.type === "connection.restored") {
        generation.current += 1;
        inFlight.current = false;
        reading.current = null;
        readingTarget.current = current.current?.readThrough ?? 0;
        readingFailed.current = false;
        blocked.current = false;
        dirty.current = false;
        fresh.current.clear();
        dismissToast(`thread-${id}-read`);
        void requestRef.current(
          current.current
            ? following.current
              ? "latest"
              : "refresh"
            : "entry",
        );
        return;
      }
      if (eventDiscussion(event) !== id) return;
      if (event.type === "message.created" && typeof event.id === "number") {
        latest.current = Math.max(latest.current, event.id);
        if (current.current)
          commit({ ...current.current, latest: latest.current });
        if (!blocked.current) {
          if (inFlight.current) dirty.current = true;
          else
            void requestRef.current(
              following.current
                ? "after"
                : current.current
                  ? "refresh"
                  : "entry",
            );
        }
      } else if (
        ["mention.acked", "mention.revoked", "discussion.updated"].includes(
          event.type,
        )
      ) {
        fresh.current.clear();
        if (inFlight.current) dirty.current = true;
        else if (!blocked.current)
          void requestRef.current(current.current ? "refresh" : "entry");
      } else if (
        event.type === "discussion.read_updated" &&
        event.member_id === memberId &&
        typeof event.read_through === "number" &&
        current.current
      ) {
        commit({
          ...current.current,
          readThrough: Math.max(
            current.current.readThrough,
            event.read_through,
          ),
        });
      }
    });
    void requestRef.current("entry");
    return () => {
      live.current = false;
      generation.current += 1;
      inFlight.current = false;
      off();
      dismissToast(`thread-${id}-page`);
      dismissToast(`thread-${id}-read`);
    };
  }, [id, memberId, commit]);

  const view = useCallback(
    (ids: number[], atEnd: boolean) => {
      visible.current = ids;
      following.current =
        atEnd &&
        (current.current?.messages[current.current.messages.length - 1]?.id ??
          0) >= latest.current;
      if (!live.current || inFlight.current || failed) return;
      if (ids.some((message) => !fresh.current.has(message)))
        void requestRef.current("refresh");
      else if (
        atEnd &&
        (current.current?.messages[current.current.messages.length - 1]?.id ??
          0) < latest.current
      )
        void requestRef.current("after");
    },
    [failed],
  );
  const invalidate = useCallback(async () => {
    fresh.current.clear();
    if (inFlight.current) dirty.current = true;
    else await requestRef.current("refresh");
  }, []);
  const positioned = useCallback(() => setPosition(null), []);
  return {
    data,
    missing,
    failed,
    loading,
    position,
    positioned,
    request,
    markRead,
    invalidate,
    view,
    current,
    live,
  };
}
