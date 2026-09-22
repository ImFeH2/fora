import { describe, expect, it } from "vitest";
import { eventDiscussion, mergePage } from "@/features/discussions/thread-data";
import type { DiscussionPage } from "@/lib/backend";

function page(
  ids: number[],
  extras: Partial<DiscussionPage> = {},
): DiscussionPage {
  return {
    id: 1,
    messages: ids.map((id) => ({
      id,
      sender_id: 2,
      sender_name: "Helper",
      body: `Message ${id}`,
      mentions: [],
      attachments: [],
      created_at: "2026-09-17T00:00:00Z",
    })),
    read_through: 5,
    latest_id: 100,
    first_unread_id: 6,
    has_before: true,
    has_after: true,
    previous_sender_id: 2,
    awaiting_ack: [],
    acknowledged: [],
    pending_count: 0,
    metadata: {
      topic: "Window",
      archived: false,
      members: [{ id: 1, name: "You" }],
    },
    ...extras,
  };
}

describe("thread pages", () => {
  it("keeps the entering divider even when the first item is unread", () => {
    const data = mergePage(null, page([6, 7]), "entry");
    expect(data.divider).toBe(6);
    expect(data.previousSender).toBe(2);
    const read = mergePage(
      data,
      page([6, 7], { read_through: 20, first_unread_id: 21 }),
      "refresh",
    );
    expect(read.divider).toBe(6);
    expect(read.readThrough).toBe(20);
  });

  it("retains an own-message divider while reading and renews it on entry", () => {
    const own = page([4, 5, 6, 7, 8]);
    own.messages = own.messages.map((message) => ({
      ...message,
      sender_id: 1,
      sender_name: "You",
    }));
    const entered = mergePage(null, own, "entry");
    expect(entered.divider).toBe(6);
    expect(entered.messages.map((message) => message.id)).toEqual([
      4, 5, 6, 7, 8,
    ]);
    const updated = {
      ...own,
      read_through: 7,
      first_unread_id: 8,
    };
    const reading = mergePage(entered, updated, "refresh");
    expect(reading.readThrough).toBe(7);
    expect(reading.divider).toBe(6);
    expect(mergePage(null, updated, "entry").divider).toBe(8);
    expect(
      mergePage(
        null,
        { ...updated, read_through: 8, first_unread_id: null },
        "entry",
      ).divider,
    ).toBeNull();
  });

  it("merges both directions without duplicates and refreshes only returned ack state", () => {
    const initial = mergePage(
      null,
      page([6, 7], { awaiting_ack: [7] }),
      "entry",
    );
    const before = mergePage(
      initial,
      page([4, 5, 6], { has_before: false, previous_sender_id: null }),
      "before",
    );
    const after = mergePage(
      before,
      page([7, 8], { acknowledged: [7] }),
      "after",
    );
    expect(after.messages.map((message) => message.id)).toEqual([
      4, 5, 6, 7, 8,
    ]);
    expect(after.hasBefore).toBe(false);
    expect(after.previousSender).toBeNull();
    expect(after.messages.find((message) => message.id === 7)).toMatchObject({
      pending: false,
      acknowledged: true,
    });
    expect(after.divider).toBe(6);
  });

  it("jumps to a separate tail segment rather than fabricating contiguous history", () => {
    const initial = mergePage(null, page([6, 7]), "entry");
    const tail = mergePage(initial, page([99, 100]), "latest");
    expect(tail.messages.map((message) => message.id)).toEqual([99, 100]);
    expect(tail.divider).toBe(6);
    expect(tail.hasAfter).toBe(false);
    expect(tail.readThrough).toBe(5);
  });

  it("does not regress persisted reading state from older page snapshots", () => {
    const initial = mergePage(null, page([6], { read_through: 70 }), "entry");
    expect(
      mergePage(initial, page([7], { read_through: 4 }), "after").readThrough,
    ).toBe(70);
  });

  it("handles empty discussions and no unread entry without a divider", () => {
    const data = mergePage(
      null,
      page([], {
        first_unread_id: null,
        latest_id: 0,
        has_before: false,
        has_after: false,
      }),
      "entry",
    );
    expect(data.messages).toEqual([]);
    expect(data.divider).toBeNull();
    expect(data.hasAfter).toBe(false);
  });

  it("rejects cross-discussion response merges and changed immutable bodies", () => {
    const data = mergePage(null, page([6]), "entry");
    expect(() => mergePage(data, page([7], { id: 2 }), "after")).toThrow(
      "another thread",
    );
    const changed = page([6]);
    changed.messages[0].body = "changed";
    expect(() => mergePage(data, changed, "refresh")).toThrow(
      "Immutable message changed",
    );
  });

  it("requires initial metadata and retains it on incremental pages", () => {
    expect(() =>
      mergePage(null, page([1], { metadata: undefined }), "entry"),
    ).toThrow("missing metadata");
    const initial = mergePage(null, page([6]), "entry");
    expect(
      mergePage(initial, page([7], { metadata: undefined }), "after").topic,
    ).toBe("Window");
  });
});

describe("thread events", () => {
  it("distinguishes message IDs from discussion IDs", () => {
    expect(
      eventDiscussion({ type: "message.created", discussion_id: 4, id: 99 }),
    ).toBe(4);
    expect(eventDiscussion({ type: "discussion.updated", id: 4 })).toBe(4);
    expect(eventDiscussion({ type: "mention.acked", discussion_id: 4 })).toBe(
      4,
    );
    expect(eventDiscussion({ type: "member.updated", id: 4 })).toBeUndefined();
  });
});
