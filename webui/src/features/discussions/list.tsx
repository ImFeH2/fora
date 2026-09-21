import {
  Archive,
  ArchiveRestore,
  MessageCircle,
  MessageSquare,
  MessagesSquare,
  Plus,
  Search,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useOrganization } from "@/app/organization";
import { useNavigate } from "@/app/router";
import {
  type Column,
  Page,
  PageBody,
  PageHeader,
  RowLink,
  Table,
  Toolbar,
} from "@/components/layout/shell";
import {
  Badge,
  Button,
  Chip,
  CountPill,
  Dot,
  IconButton,
  SearchField,
  StatusText,
} from "@/components/ui/index";
import { OverflowMenu } from "@/components/ui/menu";
import { StatePanel } from "@/components/ui/state-panel";
import { CreateDiscussionDialog } from "@/features/discussions/create";
import {
  backend,
  type DiscussionSummary,
  type FoundMessage,
  type Member,
} from "@/lib/backend";
import { plural } from "@/lib/format";

type Segment = { id: string; text: string; match: boolean };

function segments(text: string, query: string): Segment[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [{ id: "s0", text, match: false }];
  const found: Segment[] = [];
  const haystack = text.toLowerCase();
  let from = 0;
  let count = 0;
  let at = haystack.indexOf(needle, from);
  while (at >= 0) {
    if (at > from)
      found.push({
        id: `s${count++}`,
        text: text.slice(from, at),
        match: false,
      });
    found.push({
      id: `s${count++}`,
      text: text.slice(at, at + needle.length),
      match: true,
    });
    from = at + needle.length;
    at = haystack.indexOf(needle, from);
  }
  if (from < text.length)
    found.push({ id: `s${count++}`, text: text.slice(from), match: false });
  return found;
}

const LIST_COLUMNS: Column[] = [
  { key: "topic", label: "Discussion" },
  { key: "members", label: "Members", hideBelow: "md" },
  { key: "unread", label: "Unread", align: "end", width: "96px" },
  { key: "status", label: "Status", width: "168px" },
  { key: "actions", label: "", width: "56px" },
];

const RESULT_COLUMNS: Column[] = [
  { key: "message", label: "Message" },
  { key: "sender", label: "Sender", width: "180px", hideBelow: "sm" },
  { key: "discussion", label: "Discussion", width: "220px" },
];

export function DiscussionsPage() {
  const { members } = useOrganization();
  const navigate = useNavigate();
  const [list, setList] = useState<DiscussionSummary[] | null>(null);
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState<{
    query: string;
    items: FoundMessage[];
  } | null>(null);
  const results = search?.query === query.trim() ? search.items : null;
  const [archived, setArchived] = useState(false);
  const [creating, setCreating] = useState(false);
  const [connectionRevision, setConnectionRevision] = useState(0);

  const load = useCallback(async () => {
    try {
      setList(await backend.discussions(archived));
    } catch (failure) {
      backend.reportFailure(failure);
    }
  }, [archived]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    return backend.onEvent((event) => {
      if (event.type === "connection.restored")
        setConnectionRevision((value) => value + 1);
      if (
        event.type === "connection.restored" ||
        event.type === "message.created" ||
        event.type === "mention.acked" ||
        event.type === "mention.revoked" ||
        event.type === "discussion.created" ||
        event.type === "discussion.updated"
      ) {
        void load();
      }
    });
  }, [load]);

  useEffect(() => {
    const text = query.trim();
    if (!text) {
      setSearch(null);
      return;
    }
    let live = true;
    const timer = setTimeout(() => {
      void backend
        .searchMessages(text)
        .then((found) => {
          if (live) setSearch({ query: text, items: found });
        })
        .catch(backend.reportFailure);
    }, 120);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [query, connectionRevision]);

  const byId = useMemo(
    () => new Map(members.map((member) => [member.id, member] as const)),
    [members],
  );

  const topicOf = (id: number) =>
    list?.find((item) => item.id === id)?.topic ?? `Discussion ${id}`;

  const searching = query.trim().length > 0;

  return (
    <Page>
      <PageHeader
        title="Discussions"
        actions={
          <Button variant="primary" onClick={() => setCreating(true)}>
            <Plus size={16} />
            New Discussion
          </Button>
        }
      />
      <Toolbar>
        <SearchField
          icon={<Search size={15} />}
          value={query}
          placeholder="Search messages"
          aria-label="Search messages"
          onChange={(event) => setQuery(event.target.value)}
        />
        <IconButton
          label="Show archived"
          pressed={archived}
          onClick={() => setArchived((current) => !current)}
        >
          <Archive size={15} />
        </IconButton>
      </Toolbar>
      <PageBody>
        {(searching ? results !== null : list !== null) ? (
          <CountPill>
            {results
              ? plural(results.length, "result")
              : plural(list?.length ?? 0, "Discussion")}
          </CountPill>
        ) : null}

        {searching ? (
          results === null ? null : results.length === 0 ? (
            <StatePanel
              icons={[MessageCircle, Search, MessagesSquare]}
              title="No messages match"
              description="Try a different search or clear it to see your Discussions."
              action={{
                label: "Clear search",
                icon: <X size={16} />,
                onClick: () => setQuery(""),
              }}
            />
          ) : (
            <Table columns={RESULT_COLUMNS} label="Search results">
              {results.map((result) => (
                <tr key={`${result.discussion_id}-${result.id}`}>
                  <td>
                    <RowLink
                      primary={
                        <span className="block truncate font-normal text-fg-muted [&_mark]:bg-blue-500/30 [&_mark]:rounded-xs [&_mark]:text-fg [&_mark]:font-medium">
                          {segments(result.body, query).map((part) =>
                            part.match ? (
                              <mark key={part.id}>{part.text}</mark>
                            ) : (
                              <span key={part.id}>{part.text}</span>
                            ),
                          )}
                        </span>
                      }
                      onSelect={() =>
                        navigate({
                          name: "discussion",
                          id: result.discussion_id,
                        })
                      }
                    />
                  </td>
                  <td data-hide-below="sm" className="text-fg-muted">
                    {result.sender_name}
                  </td>
                  <td>
                    <Chip>{topicOf(result.discussion_id)}</Chip>
                  </td>
                </tr>
              ))}
            </Table>
          )
        ) : list === null ? null : list.length === 0 ? (
          <StatePanel
            icons={[MessageCircle, MessagesSquare, MessageSquare]}
            title="No Discussions yet"
            description="Create a Discussion to bring Members together."
            action={{
              label: "New Discussion",
              icon: <Plus size={16} />,
              onClick: () => setCreating(true),
            }}
          />
        ) : (
          <Table columns={LIST_COLUMNS} label="Discussions">
            {list.map((item) => (
              <DiscussionRow
                key={item.id}
                item={item}
                byId={byId}
                onOpen={() => navigate({ name: "discussion", id: item.id })}
                onArchive={async () => {
                  await backend.archiveDiscussion(item.id, !item.archived);
                  await load();
                }}
              />
            ))}
          </Table>
        )}
      </PageBody>

      <CreateDiscussionDialog
        open={creating}
        onOpenChange={setCreating}
        onCreated={(id) => navigate({ name: "discussion", id })}
      />
    </Page>
  );
}

export function DiscussionRow({
  item,
  byId,
  onOpen,
  onArchive,
}: {
  item: DiscussionSummary;
  byId: Map<number, Member>;
  onOpen: () => void;
  onArchive: () => void;
}) {
  const people = item.member_ids
    .map((id) => byId.get(id))
    .filter((member): member is Member => member !== undefined);
  const shown = people.slice(0, 3);
  const rest = people.length - shown.length;

  return (
    <tr data-highlight={item.unread > 0}>
      <td>
        <div className="flex min-w-0 items-center gap-3">
          <span
            className="flex size-7 flex-none items-center justify-center rounded-sm border border-line bg-gray-800 text-fg-muted"
            aria-hidden="true"
          >
            <MessageSquare size={15} />
          </span>
          <RowLink primary={item.topic} onSelect={onOpen} />
        </div>
      </td>
      <td data-hide-below="md">
        <div className="flex flex-wrap gap-1">
          {shown.map((member) => (
            <Chip
              key={member.id}
              tone={member.type === "agent" ? "blue" : "neutral"}
            >
              {member.name}
            </Chip>
          ))}
          {rest > 0 ? <Chip>+{rest}</Chip> : null}
        </div>
      </td>
      <td data-align="end" className="tabular-nums whitespace-nowrap">
        {item.unread > 0 ? (
          <Badge tone="unread">{item.unread}</Badge>
        ) : (
          <span className="text-fg-muted">—</span>
        )}
      </td>
      <td>
        {item.archived ? (
          <StatusText dot={<Dot tone="grey" />}>
            <span className="text-fg-muted">Archived</span>
          </StatusText>
        ) : item.unread > 0 ? (
          <StatusText dot={<Dot tone="blue" />}>
            {plural(item.unread, "new message")}
          </StatusText>
        ) : (
          <StatusText dot={<Dot tone="green" />}>
            <span className="text-fg-muted">Up to date</span>
          </StatusText>
        )}
      </td>
      <td className="relative z-1 w-12 text-right">
        <OverflowMenu
          label={`Actions for ${item.topic}`}
          actions={[
            {
              id: "archive",
              label: item.archived ? "Unarchive" : "Archive",
              icon: item.archived ? (
                <ArchiveRestore size={15} />
              ) : (
                <Archive size={15} />
              ),
              onSelect: onArchive,
            },
          ]}
        />
      </td>
    </tr>
  );
}
