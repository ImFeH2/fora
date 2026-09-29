export type Thinking =
  | "default"
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max"
  | "budget";

export type AgentModelConfig = {
  model_id: string | null;
  thinking: Thinking | null;
};

export type ProviderConfig = {
  id: string;
  name: string;
  api_type: string;
  base_url: string;
  api_key_set: boolean;
  enabled: boolean;
};

export type RegisteredModel = {
  id: string;
  provider_id: string;
  name: string;
  model: string;
  enabled: boolean;
  thinking_budget_tokens: number | null;
  thinking_options: Thinking[];
};

export type ModelCatalog = {
  version: 2;
  providers: ProviderConfig[];
  models: RegisteredModel[];
  default_model_id: string | null;
  default_thinking: Thinking;
  agent_configs: Record<string, AgentModelConfig>;
};

export type AgentStatus = {
  state: "idle" | "running" | "paused" | "blocked" | "error";
  pause_requested?: boolean;
  reasons?: { code: string; message: string; recovery: string }[];
  error?: string | null;
  scheduler_stopped?: string | null;
};

export type Member = AgentStatus & {
  id: number;
  type: "human" | "agent";
  name: string;
  tokens?: number;
};

export type DiscussionSummary = {
  id: number;
  topic: string;
  member_ids: number[];
  archived: boolean;
  unread: number;
};

export type MessageMention = {
  member_id: number;
  position: number;
  length: number;
};

export type Attachment = {
  id: string;
  name: string;
  size: number;
  media_type: string;
  sha256: string;
  width: number | null;
  height: number | null;
};

export type UploadRecord = {
  id: string;
  state:
    | "reserved"
    | "receiving"
    | "ready"
    | "attached"
    | "deleting"
    | "expired";
  expires_at: number;
};

export type Message = {
  id: number;
  sender_id: number;
  sender_name: string;
  body: string;
  mentions: MessageMention[];
  attachments: Attachment[];
  created_at: string;
};

export type DiscussionDetail = {
  id: number;
  topic: string;
  members: { id: number; name: string }[];
  total_messages: number;
  archived: boolean;
  read_through: number;
  awaiting_ack: number[];
  acknowledged: number[];
  messages: Message[];
};

export type DiscussionPage = {
  id: number;
  messages: Message[];
  read_through: number;
  latest_id: number;
  first_unread_id?: number | null;
  has_before: boolean;
  has_after: boolean;
  previous_sender_id: number | null;
  awaiting_ack: number[];
  acknowledged: number[];
  pending_count: number;
  metadata?: {
    topic: string;
    archived: boolean;
    members: { id: number; name: string }[];
  };
};

export type PageRequest = {
  entry?: boolean;
  before?: number;
  after?: number;
  limit?: number;
  metadata?: boolean;
};

export type TurnEffect = { ordinal: number; tool: string; summary: string };

export type AgentRun = {
  sequence: number;
  status: string;
  started_at: string;
  completed_at: string | null;
  usage: string | null;
  error: string | null;
  effects: TurnEffect[];
};

export type Usage = {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  requests: number;
  total_tokens: number;
};

export type AgentDetail = Partial<AgentStatus> & {
  id: number;
  workspace: LibraryEntry[];
  runs: AgentRun[];
  usage: Usage;
  token_limit: number | null;
  over_token_limit: boolean | null;
  idle: boolean | null;
  statistics_unavailable?: string | null;
  idle_streak: number;
  no_tool_streak: number;
  pause_reason: string | null;
  window: {
    number: number;
    since_sequence: number;
    reset_at: string | null;
    reason: string | null;
  };
};

export type FoundMessage = {
  discussion_id: number;
  id: number;
  sender_name: string;
  body: string;
  mentions: MessageMention[];
};

export type LibraryEntry = {
  path: string;
  kind: "file" | "directory";
  size: number;
  modified_at: string;
};

export type LibraryDocument = { path: string; content: string; hash: string };

type LibraryWriteResult =
  | (LibraryEntry & { hash: string; conflict?: false })
  | {
      conflict: true;
      path: string;
      current_hash: string;
      current_content: string;
    };

export type BackendEvent = { type: string } & Record<string, unknown>;

export class BackendError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly transport = false,
    readonly operation: string | null = null,
  ) {
    super(message);
    this.name = "BackendError";
  }
}

type Pending = {
  method: string;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export type Frame = {
  type: string;
  id?: number;
  result?: unknown;
  error?: { code: string; message: string };
} & Record<string, unknown>;

export type Connection = { url: string } | { error: string };

export type Socket = {
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onclose: ((event: CloseEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  send(data: string): void;
  close(): void;
};

const REQUEST_TIMEOUT = 60_000;
const READ_METHODS = new Set([
  "ping",
  "organization.get",
  "discussion.list",
  "discussion.page",
  "discussion.search",
  "agent.detail",
  "library.list",
  "library.read",
  "workspace.list",
  "workspace.read",
  "settings.get",
  "settings.list_models",
]);

function unconfirmed(method: string): BackendError {
  return new BackendError(
    "unconfirmed",
    method === "discussion.send"
      ? "Your message may have been sent. Check the discussion before sending it again."
      : "The operation may have completed. Check the current state before trying again.",
    true,
    method,
  );
}

const DIAGNOSIS_TIMEOUT = 5_000;

export async function diagnoseConnection(url: string): Promise<BackendError> {
  const failure = new BackendError(
    "connection_failed",
    "Check that Huddol is running, then reconnect.",
    true,
  );
  const endpoint = new URL(url);
  endpoint.protocol = endpoint.protocol === "wss:" ? "https:" : "http:";
  let response: Response;
  try {
    response = await fetch(endpoint, {
      credentials: "omit",
      cache: "no-store",
      redirect: "error",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.timeout(DIAGNOSIS_TIMEOUT),
    });
    await response.body?.cancel();
  } catch {
    return failure;
  }
  return response.status === 401
    ? new BackendError(
        "authentication_failed",
        "The access credentials are no longer valid.",
        true,
      )
    : failure;
}

export function connectionFrom(search: string, origin: string): Connection {
  const params = new URLSearchParams(search);
  const error = params.get("error");
  if (error !== null) return { error };
  const ws =
    params.get("ws") ??
    `${origin.replace(/^https:/, "wss:").replace(/^http:/, "ws:")}/ws`;
  const url = new URL(ws);
  url.searchParams.set("token", params.get("token") ?? "");
  return { url: url.toString() };
}

export type Resolved = {
  connection: Connection;
  remember: string | null;
  scrub: boolean;
};

export function resolveConnection(
  search: string,
  origin: string,
  remembered: string | null,
): Resolved {
  const params = new URLSearchParams(search);
  if (params.has("error")) {
    return {
      connection: connectionFrom(search, origin),
      remember: null,
      scrub: false,
    };
  }
  if (params.has("token") || params.has("ws")) {
    const connection = connectionFrom(search, origin);
    return {
      connection,
      remember: "url" in connection ? connection.url : null,
      scrub: true,
    };
  }
  if (remembered) {
    return { connection: { url: remembered }, remember: null, scrub: false };
  }
  return {
    connection: connectionFrom(search, origin),
    remember: null,
    scrub: false,
  };
}

export function withoutConnectionParams(href: string): string {
  const url = new URL(href);
  url.searchParams.delete("ws");
  url.searchParams.delete("token");
  return url.toString();
}

const STORAGE_KEY = "huddol.connection";

function recall(): string | null {
  try {
    return sessionStorage.getItem(STORAGE_KEY);
  } catch {
    throw new BackendError(
      "storage_read_failed",
      "Could not read browser session storage. Allow session storage, then reopen Huddol from its original launch entry.",
      true,
    );
  }
}

function remember(url: string): void {
  try {
    sessionStorage.setItem(STORAGE_KEY, url);
  } catch {
    throw new BackendError(
      "storage_write_failed",
      "Could not save the connection in browser session storage. Allow session storage, then reopen Huddol from its original launch entry.",
      true,
    );
  }
}

function pageConnection(): Connection {
  const search = location.search;
  const params = new URLSearchParams(search);
  if (params.has("token") || params.has("ws")) {
    try {
      history.replaceState(
        history.state,
        "",
        withoutConnectionParams(location.href),
      );
    } catch {
      throw new BackendError(
        "url_cleanup_failed",
        "Could not clear connection parameters from the address. Close this page and reopen Huddol from its original launch entry.",
        true,
      );
    }
  }
  const remembered = recall();
  let resolved: Resolved;
  try {
    resolved = resolveConnection(search, location.origin, remembered);
  } catch {
    throw new BackendError(
      "invalid_connection",
      "The connection address is invalid. Reopen Huddol from its original launch entry.",
      true,
    );
  }
  if (resolved.remember) remember(resolved.remember);
  return resolved.connection;
}

function browserSocket(url: string): Socket {
  return new WebSocket(url);
}

export class Backend {
  #locate: () => Connection;
  #socket: (url: string) => Socket;
  #nextId = 1;
  #connectionUrl: URL | null = null;
  #pending = new Map<number, Pending>();
  #listeners = new Set<(event: BackendEvent) => void>();
  #ready: Promise<Socket> | null = null;
  #activeSocket: Socket | null = null;
  #generation = 0;
  #reconnecting: Promise<void> | null = null;
  #automaticAttempted = false;
  #closed: BackendError | null = null;
  #rejectConnection: ((error: BackendError) => void) | null = null;
  #failures = new Set<(error: BackendError) => void>();

  constructor(
    locate: () => Connection = pageConnection,
    socket: (url: string) => Socket = browserSocket,
  ) {
    this.#locate = locate;
    this.#socket = socket;
  }

  async connect(): Promise<void> {
    await this.#open();
  }

  async voiceUrl(): Promise<string> {
    const socket = await this.#open();
    if (this.#closed) throw this.#closed;
    if (socket !== this.#activeSocket || this.#connectionUrl === null)
      throw new BackendError("connection_changed", "Connection changed.", true);
    const url = new URL(this.#connectionUrl);
    url.pathname = "/voice";
    return url.toString();
  }

  reconnect(automatic = false): Promise<void> {
    if (this.#reconnecting) return this.#reconnecting;
    if (
      !this.#closed ||
      (automatic &&
        (this.#automaticAttempted ||
          !["disconnected", "connection_failed", "connection_timeout"].includes(
            this.#closed.code,
          )))
    )
      return Promise.resolve();
    this.#automaticAttempted = true;
    this.#closed = null;
    const generation = this.#generation;
    let resolve!: () => void;
    let reject!: (error: unknown) => void;
    const operation = new Promise<void>((yes, no) => {
      resolve = yes;
      reject = no;
    });
    this.#reconnecting = operation;
    this.#emit({ type: "connection.reconnecting" });
    this.#open()
      .then(() => {
        if (generation !== this.#generation)
          throw new BackendError(
            "connection_changed",
            "Connection changed while reconnecting.",
            true,
          );
        this.#reconnecting = null;
        this.#automaticAttempted = false;
        this.#emit({ type: "connection.restored" });
      })
      .finally(() => {
        if (this.#reconnecting === operation) this.#reconnecting = null;
      })
      .then(resolve, reject);
    return operation;
  }

  #emit(event: BackendEvent): void {
    for (const listener of this.#listeners) listener(event);
  }

  #attach(): Promise<Socket> {
    const connection = this.#locate();
    if ("error" in connection)
      throw new BackendError("startup_failed", connection.error, true);
    if (!new URL(connection.url).searchParams.get("token"))
      throw new BackendError(
        "authentication_missing",
        "This page was opened without access credentials.",
        true,
      );
    this.#connectionUrl = new URL(connection.url);
    return new Promise<Socket>((resolve, reject) => {
      const socket = this.#socket(connection.url);
      const generation = this.#generation;
      this.#activeSocket = socket;
      let opened = false;
      socket.onopen = () => {
        if (generation !== this.#generation) return;
        opened = true;
        resolve(socket);
      };
      socket.onmessage = (event) => {
        if (generation !== this.#generation) return;
        let frame: Frame;
        try {
          frame = JSON.parse(String(event.data));
        } catch {
          this.#disconnect(
            new BackendError(
              "protocol_error",
              "Huddol received an invalid WebSocket message. Reopen Huddol to reconnect. Check pending operations before trying them again.",
              true,
            ),
          );
          return;
        }
        this.#receive(frame);
      };
      socket.onerror = () => {};
      socket.onclose = (event) => {
        if (generation !== this.#generation) return;
        console.info("WebSocket closed", {
          code: event.code,
          reason:
            event.reason === "keepalive ping timeout" || event.reason === ""
              ? event.reason
              : "[redacted]",
          wasClean: event.wasClean,
        });
        if (opened) {
          this.#disconnect(
            new BackendError(
              "disconnected",
              "Connection lost. Reconnect to continue.",
              true,
            ),
          );
          return;
        }
        void diagnoseConnection(connection.url).then((error) => {
          if (generation === this.#generation) reject(error);
        });
      };
    });
  }

  #open(): Promise<Socket> {
    if (this.#closed) return Promise.reject(this.#closed);
    if (this.#ready) return this.#ready;
    const generation = this.#generation;
    let resolve!: (socket: Socket) => void;
    let reject!: (error: BackendError) => void;
    const ready = new Promise<Socket>((yes, no) => {
      resolve = yes;
      reject = no;
    });
    this.#ready = ready;
    const timer = setTimeout(() => {
      this.#disconnect(
        new BackendError(
          "connection_timeout",
          "Connection timed out. Reconnect to try again.",
          true,
        ),
      );
    }, REQUEST_TIMEOUT);
    this.#rejectConnection = (error) => {
      clearTimeout(timer);
      this.#rejectConnection = null;
      reject(error);
    };
    const fail = (error: unknown) => {
      if (generation !== this.#generation) return;
      this.#disconnect(
        error instanceof BackendError
          ? error
          : new BackendError(
              "connection_failed",
              "Could not open the connection. Reopen the Huddol link.",
              true,
            ),
      );
    };
    let attached: Promise<Socket>;
    try {
      attached = this.#attach();
    } catch (error) {
      fail(error);
      return ready;
    }
    attached.then((socket) => {
      if (generation !== this.#generation) return;
      clearTimeout(timer);
      this.#rejectConnection = null;
      resolve(socket);
    }, fail);
    return ready;
  }

  get disconnected(): boolean {
    return this.#closed !== null;
  }

  onFailure(listener: (error: BackendError) => void): () => void {
    this.#failures.add(listener);
    if (this.#closed) listener(this.#closed);
    return () => this.#failures.delete(listener);
  }

  reportFailure = (error: unknown): void => {
    if (this.#closed || (error instanceof BackendError && error.transport))
      return;
    this.#notify(
      error instanceof BackendError
        ? error
        : new BackendError(
            "request_failed",
            error instanceof Error ? error.message : String(error),
          ),
    );
  };

  #notify(error: BackendError): void {
    for (const listener of this.#failures) listener(error);
  }

  #disconnect(error: BackendError): void {
    if (this.#closed) return;
    this.#closed = error;
    this.#generation += 1;
    this.#ready = null;
    this.#reconnecting = null;
    this.#rejectConnection?.(error);
    const socket = this.#activeSocket;
    this.#activeSocket = null;
    socket?.close();
    for (const pending of this.#pending.values()) {
      const failure = READ_METHODS.has(pending.method)
        ? error
        : unconfirmed(pending.method);
      pending.reject(failure);
      if (failure !== error) this.#notify(failure);
    }
    this.#pending.clear();
    this.#notify(error);
    this.#emit({ type: "connection.closed", error });
  }

  onEvent(listener: (event: BackendEvent) => void): () => void {
    this.#listeners.add(listener);
    return () => this.#listeners.delete(listener);
  }

  #receive(frame: Frame): void {
    if (this.#closed) return;
    if (frame.type === "response" && typeof frame.id === "number") {
      const pending = this.#pending.get(frame.id);
      if (!pending) return;
      if (frame.error) {
        pending.reject(new BackendError(frame.error.code, frame.error.message));
      } else {
        pending.resolve(frame.result);
      }
      return;
    }
    for (const listener of this.#listeners) listener(frame as BackendEvent);
  }

  async call<T>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T> {
    const socket = await this.#open();
    if (this.#closed) throw this.#closed;
    if (socket !== this.#activeSocket)
      throw new BackendError(
        "connection_changed",
        "Connection changed. Request was not sent.",
        true,
      );
    const id = this.#nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        const error = READ_METHODS.has(method)
          ? new BackendError(
              "timeout",
              "Request timed out. Try reading again.",
              true,
            )
          : unconfirmed(method);
        const pending = this.#pending.get(id);
        if (!pending) return;
        pending.reject(error);
        this.#notify(error);
      }, REQUEST_TIMEOUT);
      const finish = () => {
        clearTimeout(timer);
        this.#pending.delete(id);
      };
      this.#pending.set(id, {
        method,
        resolve: (value) => {
          finish();
          resolve(value as T);
        },
        reject: (error) => {
          finish();
          reject(error);
        },
      });
      try {
        socket.send(JSON.stringify({ id, method, params }));
      } catch {
        const error = new BackendError(
          "send_failed",
          "Could not send request. Check before retrying.",
          true,
        );
        this.#pending.get(id)?.reject(error);
        this.#notify(error);
      }
    });
  }

  organization() {
    return this.call<{
      id: number;
      uuid: string;
      members: Member[];
      human_id: number;
      token_limit: number | null;
    }>("organization.get");
  }

  modelCatalog() {
    return this.call<ModelCatalog>("settings.get", { section: "model" });
  }

  configureModel(action: string, id: string | number | null, values = {}) {
    return this.call<ModelCatalog>("settings.update", {
      section: "model",
      values: { action, id, values },
    });
  }

  createAgent(name: string, model_config?: AgentModelConfig) {
    return this.call<Member>("organization.create_agent", {
      name,
      model_config,
    });
  }

  renameMember(member_id: number, name: string) {
    return this.call<Member>("organization.rename_member", { member_id, name });
  }

  pauseAgent(agent_id: number) {
    return this.call<Member>("organization.pause_agent", { agent_id });
  }

  resumeAgent(agent_id: number) {
    return this.call<Member>("organization.resume_agent", { agent_id });
  }

  deleteAgent(agent_id: number) {
    return this.call<{ id: number }>("organization.delete_agent", { agent_id });
  }

  discussions(include_archived = false) {
    return this.call<DiscussionSummary[]>("discussion.list", {
      include_archived,
    });
  }

  createDiscussion(topic: string, member_ids: number[]) {
    return this.call<DiscussionSummary>("discussion.create", {
      topic,
      member_ids,
    });
  }

  readDiscussion(discussion_id: number, message_id?: number) {
    return this.call<DiscussionDetail>("discussion.read", {
      discussion_id,
      message_id,
    });
  }

  discussionPage(discussion_id: number, page: PageRequest) {
    return this.call<DiscussionPage>("discussion.page", {
      discussion_id,
      ...page,
    });
  }

  markRead(discussion_id: number, message_id: number) {
    return this.call<{ read_through: number }>("discussion.mark_read", {
      discussion_id,
      message_id,
    });
  }

  ackPending(discussion_id: number, through_message_id: number) {
    return this.call<{
      acked: number;
      read_through: number;
      pending_count: number;
    }>("discussion.ack_pending", { discussion_id, through_message_id });
  }

  sendVisible(
    discussion_id: number,
    body: string,
    attachment_ids: string[] = [],
    client_message_id?: string,
  ) {
    return this.call<{ id: number }>("discussion.send", {
      discussion_id,
      body,
      attachment_ids,
      client_message_id,
      mark_read: false,
    });
  }

  sendStatus(discussion_id: number, client_message_id: string) {
    return this.call<{
      state: "sent" | "cancelled" | "unknown";
      message: Message | null;
    }>("discussion.send_status", { discussion_id, client_message_id });
  }

  cancelSend(discussion_id: number, client_message_id: string) {
    return this.call<{
      state: "sent" | "cancelled";
      message: Message | null;
    }>("discussion.cancel_send", { discussion_id, client_message_id });
  }

  createUpload(discussion_id: number, client_upload_id: string, file: File) {
    return this.call<UploadRecord>("upload.create", {
      discussion_id,
      client_upload_id,
      name: file.name,
      size: file.size,
      media_type: file.type,
    });
  }

  uploadStatus(upload_ids: string[]) {
    return this.call<UploadRecord[]>("upload.status", { upload_ids });
  }

  cancelUploads(upload_ids: string[]) {
    return this.call<{ cancelled: number }>("upload.cancel", { upload_ids });
  }

  async #fileConnection(path: string) {
    await this.#open();
    if (!this.#connectionUrl) throw new Error("Connection address is missing");
    const url = new URL(this.#connectionUrl);
    const token = url.searchParams.get("token");
    if (!token)
      throw new BackendError(
        "unauthorized",
        "Connection authentication is missing",
      );
    url.protocol = url.protocol === "wss:" ? "https:" : "http:";
    url.pathname = path;
    url.search = "";
    return { url: url.toString(), authorization: `Bearer ${token}` };
  }

  async uploadFile(
    id: string,
    file: File,
    signal: AbortSignal,
    progress: (sent: number) => void,
  ) {
    const connection = await this.#fileConnection(
      `/uploads/${encodeURIComponent(id)}/content`,
    );
    signal.throwIfAborted();
    return new Promise<Attachment>((resolve, reject) => {
      const request = new XMLHttpRequest();
      const abort = () => request.abort();
      const finish = () => signal.removeEventListener("abort", abort);
      request.open("PUT", connection.url);
      request.setRequestHeader("Authorization", connection.authorization);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.timeout = 125_000;
      request.responseType = "json";
      request.upload.onprogress = (event) => progress(event.loaded);
      request.onload = () => {
        finish();
        if (request.status >= 200 && request.status < 300)
          resolve(request.response as Attachment);
        else
          reject(
            new BackendError(
              request.response?.error?.code ?? "upload_failed",
              request.response?.error?.message ??
                `Upload failed (${request.status})`,
            ),
          );
      };
      request.onerror = () => {
        finish();
        reject(new Error("Upload interrupted. Your draft is preserved."));
      };
      request.ontimeout = () => {
        finish();
        reject(new Error("Upload timed out. Your draft is preserved."));
      };
      request.onabort = () => {
        finish();
        reject(new DOMException("Upload cancelled", "AbortError"));
      };
      signal.addEventListener("abort", abort, { once: true });
      request.send(file);
    });
  }

  async attachmentFile(
    discussion: number,
    message: number,
    attachment: Attachment,
    signal: AbortSignal,
  ) {
    const connection = await this.#fileConnection(
      `/discussions/${discussion}/messages/${message}/attachments/${encodeURIComponent(attachment.id)}/content`,
    );
    const response = await fetch(connection.url, {
      headers: { Authorization: connection.authorization },
      signal,
      cache: "no-store",
    });
    if (!response.ok) {
      const error = response.headers
        .get("content-type")
        ?.includes("application/json")
        ? await response.json()
        : null;
      throw new BackendError(
        error?.error?.code ?? "download_failed",
        error?.error?.message ??
          `Could not read attachment (${response.status})`,
      );
    }
    if (response.headers.get("content-length") !== String(attachment.size))
      throw new Error("Attachment size does not match");
    const blob = await response.blob();
    if (blob.size !== attachment.size)
      throw new Error("Attachment download is incomplete");
    return blob;
  }

  send(discussion_id: number, body: string) {
    return this.call<{ id: number }>("discussion.send", {
      discussion_id,
      body,
    });
  }

  ack(discussion_id: number, message_ids: number[]) {
    return this.call<{ acked: number }>("discussion.ack", {
      discussion_id,
      message_ids,
    });
  }

  revokeAck(discussion_id: number, message_ids: number[]) {
    return this.call<{ revoked: number }>("discussion.revoke_ack", {
      discussion_id,
      message_ids,
    });
  }

  setDiscussionMembers(discussion_id: number, member_ids: number[]) {
    return this.call<DiscussionSummary>("discussion.set_members", {
      discussion_id,
      member_ids,
    });
  }

  archiveDiscussion(discussion_id: number, archived: boolean) {
    return this.call<{ id: number }>("discussion.archive", {
      discussion_id,
      archived,
    });
  }

  searchMessages(query: string) {
    return this.call<FoundMessage[]>("discussion.search", { query });
  }

  agentDetail(agent_id: number) {
    return this.call<AgentDetail>("agent.detail", { agent_id });
  }

  library(path?: string) {
    return this.call<LibraryEntry[]>("library.list", { path });
  }

  mkdirLibrary(path: string) {
    return this.call<LibraryEntry>("library.mkdir", { path });
  }

  workspaceList(agent_id: number, path?: string) {
    return this.call<LibraryEntry[]>("workspace.list", { agent_id, path });
  }

  workspaceRead(agent_id: number, path: string) {
    return this.call<LibraryDocument>("workspace.read", { agent_id, path });
  }

  readLibrary(path: string) {
    return this.call<LibraryDocument>("library.read", { path });
  }

  writeLibrary(path: string, content: string, expected_hash?: string) {
    return this.call<LibraryWriteResult>("library.write", {
      path,
      content,
      expected_hash,
    });
  }

  deleteLibrary(path: string) {
    return this.call<{ path: string }>("library.delete", { path });
  }

  moveLibrary(path: string, destination: string) {
    return this.call<LibraryEntry>("library.move", { path, destination });
  }

  settings(section: string) {
    return this.call<Record<string, unknown>>("settings.get", { section });
  }

  updateSettings(section: string, values: Record<string, unknown>) {
    return this.call<Record<string, unknown>>("settings.update", {
      section,
      values,
    });
  }
}

export const backend = new Backend();
