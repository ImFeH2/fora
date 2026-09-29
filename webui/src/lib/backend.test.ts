import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Backend,
  BackendError,
  type Connection,
  connectionFrom,
  type DiscussionDetail,
  type FoundMessage,
  type Frame,
  resolveConnection,
  type Socket,
  withoutConnectionParams,
} from "@/lib/backend";

type Sent = { id: number; method: string; params: Record<string, unknown> };

class FakeSocket implements Socket {
  static instances: FakeSocket[] = [];
  onopen: Socket["onopen"] = null;
  onmessage: Socket["onmessage"] = null;
  onclose: Socket["onclose"] = null;
  onerror: Socket["onerror"] = null;
  sent: Sent[] = [];
  closed = false;

  constructor(readonly url: string) {
    FakeSocket.instances.push(this);
  }

  send(data: string): void {
    if (this.closed) throw new Error("socket is closed");
    this.sent.push(JSON.parse(data));
  }

  close(): void {
    this.closed = true;
  }

  open(): void {
    this.onopen?.(new Event("open"));
  }

  reply(frame: Frame): void {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(frame) }),
    );
  }

  fail(): void {
    this.onerror?.(new Event("error"));
    this.closed = true;
    this.onclose?.(new CloseEvent("close"));
  }
}

const URL = "ws://127.0.0.1:4321/ws?token=secret";

function harness(connection: Connection = { url: URL }) {
  FakeSocket.instances = [];
  const backend = new Backend(
    () => connection,
    (url) => new FakeSocket(url),
  );
  const socket = () => {
    const last = FakeSocket.instances[FakeSocket.instances.length - 1];
    if (!last) throw new Error("no socket");
    return last;
  };
  const connected = async () => {
    const pending = backend.connect();
    await vi.waitFor(() => socket());
    socket().open();
    await pending;
    return socket();
  };
  return { backend, socket, connected };
}

describe("connectionFrom", () => {
  it("derives the same-origin socket for the full-stack page", () => {
    expect(connectionFrom("?token=abc", "http://127.0.0.1:8000")).toEqual({
      url: "ws://127.0.0.1:8000/ws?token=abc",
    });
  });

  it("uses the ws parameter from the Vite dev page", () => {
    expect(
      connectionFrom(
        "?ws=ws://127.0.0.1:4321/ws&token=abc",
        "http://127.0.0.1:1420",
      ),
    ).toEqual({ url: "ws://127.0.0.1:4321/ws?token=abc" });
  });

  it("uses the ws parameter from the Tauri page", () => {
    expect(
      connectionFrom(
        "?ws=ws://127.0.0.1:4321/ws&token=abc",
        "tauri://localhost",
      ),
    ).toEqual({ url: "ws://127.0.0.1:4321/ws?token=abc" });
    expect(
      connectionFrom(
        "?ws=ws://127.0.0.1:4321/ws&token=abc",
        "http://tauri.localhost",
      ),
    ).toEqual({ url: "ws://127.0.0.1:4321/ws?token=abc" });
  });

  it("reports the error parameter instead of a url", () => {
    expect(
      connectionFrom("?error=kernel%20did%20not%20start", "tauri://localhost"),
    ).toEqual({ error: "kernel did not start" });
  });
});

describe("resolveConnection", () => {
  it("remembers a connection given in the page address and asks to scrub it", () => {
    expect(
      resolveConnection("?token=abc", "http://127.0.0.1:8000", null),
    ).toEqual({
      connection: { url: "ws://127.0.0.1:8000/ws?token=abc" },
      remember: "ws://127.0.0.1:8000/ws?token=abc",
      scrub: true,
    });
    expect(
      resolveConnection(
        "?ws=ws://127.0.0.1:4321/ws&token=abc",
        "http://127.0.0.1:1420",
        "ws://stale/ws?token=old",
      ),
    ).toEqual({
      connection: { url: "ws://127.0.0.1:4321/ws?token=abc" },
      remember: "ws://127.0.0.1:4321/ws?token=abc",
      scrub: true,
    });
  });

  it("reuses the remembered connection after the address was scrubbed", () => {
    expect(
      resolveConnection(
        "",
        "http://127.0.0.1:8000",
        "ws://127.0.0.1:8000/ws?token=abc",
      ),
    ).toEqual({
      connection: { url: "ws://127.0.0.1:8000/ws?token=abc" },
      remember: null,
      scrub: false,
    });
  });

  it("falls back to the same-origin socket without a token when nothing is known", () => {
    expect(resolveConnection("", "http://127.0.0.1:8000", null)).toEqual({
      connection: { url: "ws://127.0.0.1:8000/ws?token=" },
      remember: null,
      scrub: false,
    });
  });

  it("keeps the error page address untouched", () => {
    expect(
      resolveConnection(
        "?error=boom",
        "tauri://localhost",
        "ws://x/ws?token=y",
      ),
    ).toEqual({ connection: { error: "boom" }, remember: null, scrub: false });
  });
});

describe("withoutConnectionParams", () => {
  it("drops the socket and token parameters and keeps the rest", () => {
    expect(
      withoutConnectionParams(
        "http://127.0.0.1:1420/?ws=ws://127.0.0.1:4321/ws&token=abc&tab=x",
      ),
    ).toBe("http://127.0.0.1:1420/?tab=x");
    expect(
      withoutConnectionParams(
        "tauri://localhost/index.html?ws=ws://127.0.0.1:4321/ws&token=abc",
      ),
    ).toBe("tauri://localhost/index.html");
    expect(withoutConnectionParams("http://127.0.0.1:8000/?token=abc")).toBe(
      "http://127.0.0.1:8000/",
    );
  });
});

describe("browser connection preparation", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    ["read", "?token=private-token"],
    ["read", "?token=private-token&error=startup"],
    ["write", "?token=private-token"],
  ])(
    "cleans the URL before a %s storage failure (%s)",
    async (operation, search) => {
      const location = new globalThis.URL(`http://localhost:4321/${search}`);
      const replaceState = vi.fn((_state, _title, url: string) => {
        location.href = url;
      });
      const storage = {
        getItem: vi.fn(() => {
          expect(location.search).not.toContain("token");
          if (operation === "read") throw new Error("private-storage-detail");
          return null;
        }),
        setItem: vi.fn(() => {
          expect(location.search).not.toContain("token");
          throw new Error("private-storage-detail");
        }),
      };
      vi.stubGlobal("location", location);
      vi.stubGlobal("history", { state: null, replaceState });
      vi.stubGlobal("sessionStorage", storage);
      const factory = vi.fn((url: string) => new FakeSocket(url));
      const backend = new Backend(undefined, factory);
      const failure = await backend.connect().catch((error) => error);
      expect(failure).toMatchObject({
        code: `storage_${operation}_failed`,
        transport: true,
      });
      expect(failure.message).not.toContain("private-");
      expect(replaceState).toHaveBeenCalledTimes(1);
      expect(factory).not.toHaveBeenCalled();
    },
  );
});

describe("Backend", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("does not connect when the page carries an error", async () => {
    vi.useFakeTimers();
    const { backend } = harness({ error: "kernel did not start" });
    const failure = vi.fn();
    backend.onFailure(failure);
    await expect(backend.connect()).rejects.toMatchObject({
      code: "startup_failed",
      message: "kernel did not start",
      transport: true,
    });
    expect(FakeSocket.instances).toHaveLength(0);
    expect(failure).toHaveBeenCalledTimes(1);
    expect(backend.disconnected).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("uses the authenticated active connection for voice", async () => {
    let connection = "wss://example.test/ws?token=voice-test";
    let socket!: FakeSocket;
    const backend = new Backend(
      () => ({ url: connection }),
      (url) => {
        socket = new FakeSocket(url);
        return socket;
      },
    );
    const connected = backend.connect();
    socket.open();
    await connected;
    connection = "wss://other.test/ws?token=other-test";
    await expect(backend.voiceUrl()).resolves.toBe(
      "wss://example.test/voice?token=voice-test",
    );
    socket.fail();
    await expect(backend.voiceUrl()).rejects.toMatchObject({
      code: "disconnected",
    });
    const recovery = backend.reconnect();
    socket.open();
    await recovery;
    await expect(backend.voiceUrl()).resolves.toBe(
      "wss://other.test/voice?token=other-test",
    );
  });

  it("opens a socket at the resolved url", async () => {
    const { connected } = harness();
    const socket = await connected();
    expect(socket.url).toBe(URL);
  });

  it.each([
    [401, "authentication_failed"],
    [204, "connection_failed"],
    [503, "connection_failed"],
  ])("classifies a refused socket using HTTP %s", async (status, code) => {
    vi.useFakeTimers();
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status }));
    vi.stubGlobal("fetch", fetch);
    const { backend, socket } = harness();
    const failure = vi.fn();
    backend.onFailure(failure);
    const pending = backend.connect().catch((error) => error);
    await vi.advanceTimersByTimeAsync(0);
    socket().fail();
    expect(await pending).toMatchObject({ code, transport: true });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      new globalThis.URL("http://127.0.0.1:4321/ws?token=secret"),
      expect.objectContaining({
        credentials: "omit",
        cache: "no-store",
        redirect: "error",
        referrerPolicy: "no-referrer",
        signal: expect.any(AbortSignal),
      }),
    );
    expect(failure).toHaveBeenCalledTimes(1);
    expect(backend.disconnected).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("rejects missing credentials without creating a socket", async () => {
    const { backend } = harness({ url: "ws://localhost/ws" });
    await expect(backend.connect()).rejects.toMatchObject({
      code: "authentication_missing",
    });
    expect(FakeSocket.instances).toHaveLength(0);
  });

  it("reports a failed diagnostic request as a connection failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network")));
    const { backend, socket } = harness();
    const pending = backend.connect().catch((error) => error);
    socket().fail();
    expect(await pending).toMatchObject({ code: "connection_failed" });
  });

  it("ignores an old authentication diagnosis after recovery", async () => {
    vi.useFakeTimers();
    let finishDiagnosis!: (response: Response) => void;
    const diagnosis = new Promise<Response>((resolve) => {
      finishDiagnosis = resolve;
    });
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(diagnosis));
    const { backend, socket } = harness();
    const failures = vi.fn();
    backend.onFailure(failures);
    const pending = backend.connect().catch((error) => error);
    socket().fail();
    await vi.advanceTimersByTimeAsync(60_000);
    expect(await pending).toMatchObject({ code: "connection_timeout" });
    const recovered = backend.reconnect();
    socket().open();
    await recovered;
    finishDiagnosis(new Response(null, { status: 401 }));
    await vi.advanceTimersByTimeAsync(0);
    expect(backend.disconnected).toBe(false);
    expect(failures).toHaveBeenCalledTimes(1);
    await expect(backend.connect()).resolves.toBeUndefined();
  });

  it("ends concurrent requests once when the socket closes", async () => {
    vi.useFakeTimers();
    const { backend, connected } = harness();
    const socket = await connected();
    const failures = vi.fn();
    const events = vi.fn();
    backend.onFailure(failures);
    backend.onEvent(events);
    const first = backend.organization().catch((error) => error);
    const second = backend.discussions().catch((error) => error);
    await vi.advanceTimersByTimeAsync(0);
    expect(socket.sent).toHaveLength(2);
    socket.fail();
    socket.onclose?.(new CloseEvent("close"));
    expect(await first).toMatchObject({ code: "disconnected" });
    expect(await second).toMatchObject({ code: "disconnected" });
    backend.reportFailure(new Error("late failure"));
    expect(failures).toHaveBeenCalledTimes(1);
    expect(events).toHaveBeenCalledTimes(1);
    expect(events).toHaveBeenCalledWith({
      type: "connection.closed",
      error: expect.objectContaining({ code: "disconnected" }),
    });
    expect(vi.getTimerCount()).toBe(0);
    await expect(backend.organization()).rejects.toMatchObject({
      code: "disconnected",
    });
    expect(socket.sent).toHaveLength(2);
  });

  it("bounds a silent handshake and ignores a late open", async () => {
    vi.useFakeTimers();
    const { backend, socket } = harness();
    const failure = vi.fn();
    backend.onFailure(failure);
    const result = backend.organization().catch((error) => error);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(await result).toMatchObject({ code: "connection_timeout" });
    socket().open();
    await expect(backend.connect()).rejects.toMatchObject({
      code: "connection_timeout",
    });
    expect(failure).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("cleans up a rejected send immediately", async () => {
    vi.useFakeTimers();
    const { backend, connected } = harness();
    const socket = await connected();
    socket.closed = true;
    await expect(backend.organization()).rejects.toMatchObject({
      code: "send_failed",
    });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("reports a request timeout without declaring the connection dead", async () => {
    vi.useFakeTimers();
    const { backend, connected } = harness();
    const socket = await connected();
    const failures = vi.fn();
    backend.onFailure(failures);
    const result = backend.organization().catch((error) => error);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(await result).toMatchObject({ code: "timeout" });
    expect(failures).toHaveBeenCalledTimes(1);
    const next = backend.organization();
    await vi.advanceTimersByTimeAsync(0);
    expect(socket.sent).toHaveLength(2);
    socket.reply({ type: "response", id: socket.sent[0].id, result: "late" });
    socket.reply({
      type: "response",
      id: socket.sent[1].id,
      result: { members: [] },
    });
    await expect(next).resolves.toEqual({ members: [] });
    expect(vi.getTimerCount()).toBe(0);
  });

  it("opens one socket and reuses it", async () => {
    const { backend, connected } = harness();
    await connected();
    await backend.connect();
    expect(FakeSocket.instances).toHaveLength(1);
  });

  it("correlates a response with its request id", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const promise = backend.createAgent("Main");
    await vi.waitFor(() => expect(socket.sent).toHaveLength(1));
    expect(socket.sent[0].method).toBe("organization.create_agent");
    socket.reply({
      type: "response",
      id: socket.sent[0].id,
      result: { id: 2, name: "Main" },
    });
    await expect(promise).resolves.toMatchObject({ name: "Main" });
  });

  it("rejects with a typed error carrying the backend code", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const promise = backend.createAgent("  ");
    await vi.waitFor(() => expect(socket.sent).toHaveLength(1));
    socket.reply({
      type: "response",
      id: socket.sent[0].id,
      error: { code: "invalid_name", message: "Member name must not be empty" },
    });
    await expect(promise).rejects.toBeInstanceOf(BackendError);
    await expect(promise).rejects.toMatchObject({ code: "invalid_name" });
  });

  it("matches responses by id regardless of order", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const first = backend.discussions();
    const second = backend.organization();
    await vi.waitFor(() => expect(socket.sent).toHaveLength(2));
    expect(socket.sent[0].id).not.toBe(socket.sent[1].id);
    socket.reply({
      type: "response",
      id: socket.sent[1].id,
      result: { members: [] },
    });
    socket.reply({ type: "response", id: socket.sent[0].id, result: [] });
    await expect(first).resolves.toEqual([]);
    await expect(second).resolves.toMatchObject({ members: [] });
  });

  it("delivers non-response frames to event listeners", async () => {
    const { backend, connected } = harness();
    const seen: unknown[] = [];
    backend.onEvent((event) => seen.push(event));
    const socket = await connected();
    socket.reply({ type: "message.created", discussion_id: 1, id: 4 });
    socket.reply({ type: "turn.started", agent_id: 2 });
    expect(seen).toHaveLength(2);
    expect(seen[0]).toMatchObject({
      type: "message.created",
      discussion_id: 1,
    });
  });

  it("does not leak responses into the event stream", async () => {
    const { backend, connected } = harness();
    const seen: unknown[] = [];
    backend.onEvent((event) => seen.push(event));
    const socket = await connected();
    const promise = backend.discussions();
    await vi.waitFor(() => expect(socket.sent).toHaveLength(1));
    socket.reply({ type: "response", id: socket.sent[0].id, result: [] });
    await promise;
    expect(seen).toHaveLength(0);
  });

  it("ignores a response for an unknown id", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    socket.reply({ type: "response", id: 999, result: null });
    expect(backend.disconnected).toBe(false);
  });

  it("ends invalid JSON requests and isolates the recovered connection", async () => {
    const { backend, connected, socket } = harness();
    const old = await connected();
    const failures = vi.fn();
    backend.onFailure(failures);
    const first = backend.organization().catch((error) => error);
    const second = backend.discussions().catch((error) => error);
    const write = backend.createAgent("Example").catch((error) => error);
    await vi.waitFor(() => expect(old.sent).toHaveLength(3));
    const invalid = new MessageEvent("message", { data: "{private-frame" });
    old.onmessage?.(invalid);
    expect(await first).toMatchObject({ code: "protocol_error" });
    expect(await second).toMatchObject({ code: "protocol_error" });
    expect(await write).toMatchObject({
      code: "unconfirmed",
      operation: "organization.create_agent",
    });
    expect(old.closed).toBe(true);
    expect(failures).toHaveBeenCalledTimes(2);
    expect(JSON.stringify(failures.mock.calls)).not.toContain("private-frame");

    const restored = backend.reconnect();
    await vi.waitFor(() => expect(socket()).not.toBe(old));
    socket().open();
    await restored;
    old.onmessage?.(invalid);
    old.onclose?.(new CloseEvent("close"));
    expect(backend.disconnected).toBe(false);
    expect(socket().sent).toHaveLength(0);
    const next = backend.organization();
    await vi.waitFor(() => expect(socket().sent).toHaveLength(1));
    socket().reply({
      type: "response",
      id: socket().sent[0].id,
      result: { members: [] },
    });
    await expect(next).resolves.toEqual({ members: [] });
    expect(failures).toHaveBeenCalledTimes(2);
  });

  it("preserves recorded mentions in Discussion reads", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const promise = backend.readDiscussion(1);
    await vi.waitFor(() => expect(socket.sent).toHaveLength(1));
    expect(socket.sent[0]).toMatchObject({
      method: "discussion.read",
      params: { discussion_id: 1 },
    });
    const result: DiscussionDetail = {
      id: 1,
      topic: "Release",
      members: [],
      total_messages: 1,
      archived: false,
      read_through: 0,
      awaiting_ack: [],
      acknowledged: [],
      messages: [
        {
          id: 4,
          sender_id: 1,
          sender_name: "You",
          body: "hi @Main",
          created_at: "2026-01-01T00:00:00Z",
          mentions: [{ member_id: 2, position: 3, length: 5 }],
          attachments: [],
        },
      ],
    };
    socket.reply({ type: "response", id: socket.sent[0].id, result });
    await expect(promise).resolves.toEqual(result);
  });

  it("keeps archiving and Agent deletion but exposes no Discussion deletion", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    expect(backend).not.toHaveProperty("deleteDiscussion");
    const pending = [
      backend.archiveDiscussion(1, true),
      backend.archiveDiscussion(1, false),
      backend.deleteAgent(2),
    ];
    await vi.waitFor(() => expect(socket.sent).toHaveLength(3));
    expect(
      socket.sent.map(({ method, params }) => ({ method, params })),
    ).toEqual([
      {
        method: "discussion.archive",
        params: { discussion_id: 1, archived: true },
      },
      {
        method: "discussion.archive",
        params: { discussion_id: 1, archived: false },
      },
      { method: "organization.delete_agent", params: { agent_id: 2 } },
    ]);
    for (const request of socket.sent)
      socket.reply({ type: "response", id: request.id, result: { id: 1 } });
    await Promise.all(pending);
  });

  it("sends the search query and preserves recorded mentions in results", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const promise = backend.searchMessages("deadline");
    await vi.waitFor(() => expect(socket.sent).toHaveLength(1));
    expect(socket.sent[0]).toMatchObject({
      method: "discussion.search",
      params: { query: "deadline" },
    });
    const result: FoundMessage[] = [
      {
        discussion_id: 1,
        id: 4,
        sender_name: "You",
        body: "@Main deadline",
        mentions: [{ member_id: 2, position: 0, length: 5 }],
      },
      {
        discussion_id: 1,
        id: 5,
        sender_name: "You",
        body: "@Main deadline",
        mentions: [],
      },
    ];
    socket.reply({ type: "response", id: socket.sent[0].id, result });
    await expect(promise).resolves.toEqual(result);
  });

  it("sends Library and Workspace tree requests with their kernel parameters", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const pending = [
      backend.mkdirLibrary("notes/deep"),
      backend.workspaceList(2, "notes"),
      backend.workspaceList(2),
      backend.workspaceRead(2, "notes/a.md"),
    ];
    await vi.waitFor(() => expect(socket.sent).toHaveLength(4));
    expect(
      socket.sent.map(({ method, params }) => ({ method, params })),
    ).toEqual([
      { method: "library.mkdir", params: { path: "notes/deep" } },
      { method: "workspace.list", params: { agent_id: 2, path: "notes" } },
      { method: "workspace.list", params: { agent_id: 2 } },
      { method: "workspace.read", params: { agent_id: 2, path: "notes/a.md" } },
    ]);
    for (const request of socket.sent)
      socket.reply({ type: "response", id: request.id, result: [] });
    await Promise.all(pending);
  });

  it("moves a library document to its destination path", async () => {
    const { backend, connected } = harness();
    const socket = await connected();
    const promise = backend.moveLibrary("old.md", "new.md");
    await vi.waitFor(() => expect(socket.sent).toHaveLength(1));
    expect(socket.sent[0]).toMatchObject({
      method: "library.move",
      params: { path: "old.md", destination: "new.md" },
    });
    socket.reply({
      type: "response",
      id: socket.sent[0].id,
      result: { path: "new.md" },
    });
    await expect(promise).resolves.toMatchObject({ path: "new.md" });
  });

  it.each(["locate", "socket"])(
    "recovers immediately after synchronous %s failure",
    async (source) => {
      vi.useFakeTimers();
      FakeSocket.instances = [];
      let attempts = 0;
      const failOnce = () => {
        if (attempts++ === 0) throw new Error("construction failed");
      };
      const backend = new Backend(
        () => {
          if (source === "locate") failOnce();
          return { url: URL };
        },
        (url) => {
          if (source === "socket") failOnce();
          return new FakeSocket(url);
        },
      );
      let recovery: Promise<void> | undefined;
      backend.onEvent((event) => {
        if (event.type === "connection.closed")
          recovery = backend.reconnect(true);
      });
      const initial = backend.connect().catch((error) => error);
      expect(await initial).toMatchObject({ code: "connection_failed" });
      expect(FakeSocket.instances).toHaveLength(1);
      FakeSocket.instances[0].open();
      await recovery;
      await expect(backend.connect()).resolves.toBeUndefined();
      expect(backend.disconnected).toBe(false);
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it("limits automatic recovery after a synchronous failure and permits manual recovery", async () => {
    let attempts = 0;
    const backend = new Backend(
      () => {
        attempts += 1;
        if (attempts < 3) throw new Error("locate unavailable");
        return { url: URL };
      },
      (url) => new FakeSocket(url),
    );
    const recoveries: Promise<unknown>[] = [];
    backend.onEvent((event) => {
      if (event.type === "connection.closed")
        recoveries.push(backend.reconnect(true).catch((error) => error));
    });
    await expect(backend.connect()).rejects.toMatchObject({
      code: "connection_failed",
    });
    await Promise.all(recoveries);
    await backend.reconnect(true);
    expect(attempts).toBe(2);
    expect(backend.disconnected).toBe(true);
    const manual = backend.reconnect();
    FakeSocket.instances[FakeSocket.instances.length - 1].open();
    await manual;
    expect(attempts).toBe(3);
    expect(backend.disconnected).toBe(false);
  });

  it("shares one recovery and rejects writes once without replaying them", async () => {
    vi.useFakeTimers();
    const { backend, socket, connected } = harness();
    const old = await connected();
    const failures = vi.fn();
    backend.onFailure(failures);
    const write = backend.send(1, "Only once").catch((error) => error);
    await vi.advanceTimersByTimeAsync(0);
    expect(old.sent).toHaveLength(1);
    old.fail();
    expect(await write).toMatchObject({
      code: "unconfirmed",
      operation: "discussion.send",
    });
    expect(failures).toHaveBeenCalledWith(
      expect.objectContaining({
        code: "unconfirmed",
        operation: "discussion.send",
      }),
    );
    expect(
      failures.mock.calls.filter(
        ([error]) => error.operation === "discussion.send",
      ),
    ).toHaveLength(1);
    const first = backend.reconnect(true);
    expect(backend.reconnect(true)).toBe(first);
    expect(backend.reconnect()).toBe(first);
    expect(FakeSocket.instances).toHaveLength(2);
    socket().open();
    await first;
    old.open();
    old.fail();
    old.reply({ type: "response", id: old.sent[0].id, result: { id: 1 } });
    expect(backend.disconnected).toBe(false);
    expect(socket().sent).toHaveLength(0);
    expect(vi.getTimerCount()).toBe(0);
  });

  it.each([
    "authentication_missing",
    "authentication_failed",
    "protocol_error",
    "storage_read_failed",
    "invalid_connection",
  ])("requires explicit recovery for %s", async (code) => {
    let attempts = 0;
    const backend = new Backend(() => {
      attempts += 1;
      throw new BackendError(code, "Explicit recovery required", true);
    });
    await expect(backend.connect()).rejects.toMatchObject({ code });
    await backend.reconnect(true);
    expect(attempts).toBe(1);
    await expect(backend.reconnect()).rejects.toMatchObject({ code });
    expect(attempts).toBe(2);
  });

  it("does not publish restoration for a connection that closes immediately after opening", async () => {
    const { backend, socket, connected } = harness();
    const old = await connected();
    old.fail();
    const events = vi.fn();
    backend.onEvent(events);
    const recovery = backend.reconnect(true).catch((error) => error);
    socket().open();
    socket().fail();
    await recovery;
    expect(backend.disconnected).toBe(true);
    expect(events.mock.calls.map(([event]) => event.type)).toEqual([
      "connection.reconnecting",
      "connection.closed",
    ]);
    await backend.reconnect(true);
    expect(FakeSocket.instances).toHaveLength(2);
  });

  it("unsubscribing stops delivery", async () => {
    const { backend, connected } = harness();
    const seen: unknown[] = [];
    const off = backend.onEvent((event) => seen.push(event));
    const socket = await connected();
    off();
    socket.reply({ type: "member.created", id: 3 });
    expect(seen).toHaveLength(0);
  });
});
