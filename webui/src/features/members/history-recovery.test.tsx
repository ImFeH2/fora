import {
  afterAll,
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import {
  type AgentHistoryRead,
  type AgentHistoryRequestSummary,
  type AgentHistoryRun,
  type AgentRun,
  Backend,
  backend,
  type Frame,
  type HistoryMessages,
  type HistoryText,
  type Socket,
} from "@/lib/backend";

type Sent = { id: number; method: string; params: Record<string, unknown> };

class TestSocket implements Socket {
  onopen: Socket["onopen"] = null;
  onmessage: Socket["onmessage"] = null;
  onclose: Socket["onclose"] = null;
  onerror: Socket["onerror"] = null;
  sent: Sent[] = [];

  constructor() {
    queueMicrotask(() => this.onopen?.(new Event("open")));
  }

  send(data: string) {
    const request: Sent = JSON.parse(data);
    this.sent.push(request);
    if (request.method === "organization.get")
      queueMicrotask(() =>
        this.reply({
          type: "response",
          id: request.id,
          result: {
            id: 1,
            uuid: "test",
            human_id: 1,
            members: [],
            token_limit: null,
          },
        }),
      );
    if (request.method === "discussion.list")
      queueMicrotask(() =>
        this.reply({ type: "response", id: request.id, result: [] }),
      );
  }

  close() {}

  disconnect() {
    this.onclose?.(new CloseEvent("close"));
  }

  reply(frame: Frame) {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(frame) }),
    );
  }

  requests(method: string) {
    return this.sent.filter((request) => request.method === method);
  }

  latest(method: string) {
    const requests = this.requests(method);
    const request = requests[requests.length - 1];
    if (!request) throw new Error(`No ${method} request`);
    return request;
  }

  respond(method: string, result: unknown) {
    this.reply({ type: "response", id: this.latest(method).id, result });
  }
}

const run: AgentRun & AgentHistoryRun = {
  sequence: 4,
  run_id: "run-4",
  legacy: false,
  window_reset_at: null,
  window_reason: null,
  status: "running",
  started_at: "2026-01-01T00:00:00Z",
  completed_at: null,
  last_saved_at: "2026-01-01T00:00:01Z",
  window_number: 1,
  request_count: 2,
  usage: null,
  error: null,
  effects: [],
};

function messages(values: string[], total = values.length): HistoryMessages {
  return {
    messages: values,
    offset: 0,
    total,
    has_more: values.length < total,
    next_offset: values.length < total ? values.length : undefined,
  };
}

function summary(ordinal: number): AgentHistoryRequestSummary {
  return {
    ordinal,
    request_id: `request-${ordinal}`,
    run_id: "run-4",
    window_number: 1,
    status: "pending",
    started_at: run.started_at,
    completed_at: null,
    streaming: false,
    input_length: 2,
    parameters_length: 20,
    settings_length: 2,
    model_length: 2,
    response_length: 0,
    input_count: 1,
    response_count: 0,
    related_count: 0,
    error: null,
  };
}

function field(source: HistoryText["source"], ordinal: number): HistoryText {
  return {
    kind: "text",
    source,
    request_ordinal: ordinal,
    message_index: 0,
    path: [],
    offset: 0,
    next_offset: 2,
    total_bytes: 2,
    value: "{}",
    has_more: false,
  };
}

function history(ordinal = 1): AgentHistoryRead {
  return {
    agent_id: 2,
    run: {
      ...run,
      legacy: false,
      window_reset_at: null,
      window_reason: null,
    },
    windows: [],
    windows_has_more: false,
    windows_next_after: null,
    requests: [summary(1), summary(2)],
    requests_has_more: false,
    requests_next_after: null,
    messages: messages(["saved-first"], 2),
    missing: [],
    request: {
      summary: summary(ordinal),
      input: messages([`input-${ordinal}`]),
      parameters: {
        kind: "text",
        source: "parameters",
        request_ordinal: ordinal,
        message_index: 0,
        path: [],
        offset: 0,
        next_offset: 10,
        total_bytes: 20,
        value: `prefix-${ordinal}`,
        has_more: true,
      },
      settings: field("settings", ordinal),
      model: field("model", ordinal),
      response: null,
      related: messages([]),
    },
  };
}

function paginatedHistory(
  windowCount: number,
  requestCount: number,
  windowsAfter = 0,
  requestAfter = 0,
): AgentHistoryRead {
  const value = history(2);
  value.run.request_count = requestCount;
  value.windows = Array.from(
    { length: Math.min(30, windowCount - windowsAfter) },
    (_, index) => ({
      number: windowsAfter + index + 1,
      since_sequence: windowsAfter + index + 1,
      reset_at: null,
      reason: null,
    }),
  );
  value.windows_has_more = windowsAfter + value.windows.length < windowCount;
  value.windows_next_after = value.windows_has_more
    ? (value.windows[value.windows.length - 1]?.number ?? null)
    : null;
  value.requests = Array.from(
    { length: Math.min(30, requestCount - requestAfter) },
    (_, index) => summary(requestAfter + index + 1),
  );
  value.requests_has_more = requestAfter + value.requests.length < requestCount;
  value.requests_next_after = value.requests_has_more
    ? (value.requests[value.requests.length - 1]?.ordinal ?? null)
    : null;
  if (!value.request) throw new Error("Missing request");
  value.request.input = messages(["input-2"], 2);
  value.request.response = messages(["response-2"], 2);
  value.request.related = messages(["related-2"], 2);
  return value;
}

let testing: typeof import("@testing-library/react");
let userEvent: typeof import("@testing-library/user-event").default;
let useApplication: typeof import("@/App").useApplication;
let HistorySection: typeof import("@/features/members/detail").HistorySection;
let Modal: typeof import("@/components/ui/dialog").Modal;
let toastModule: typeof import("@/components/ui/toast");
let TooltipProvider: typeof import("@/components/ui/tooltip").TooltipProvider;
let closeDom: () => void;
let sockets: TestSocket[];
let connection: Backend;

function Harness() {
  const application = useApplication();
  return (
    <TooltipProvider>
      {application.loaded ? (
        <HistorySection agentId={2} initialRuns={[run]} />
      ) : null}
      <toastModule.Toaster />
    </TooltipProvider>
  );
}

beforeAll(async () => {
  const packageName = "jsdom";
  const { JSDOM } = await import(packageName);
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://localhost",
    pretendToBeVisual: true,
  });
  closeDom = () => dom.window.close();
  vi.stubGlobal("window", dom.window);
  vi.stubGlobal("document", dom.window.document);
  vi.stubGlobal("navigator", dom.window.navigator);
  for (const name of [
    "HTMLElement",
    "HTMLInputElement",
    "Node",
    "NodeFilter",
    "MutationObserver",
    "Event",
    "EventTarget",
    "CustomEvent",
    "MouseEvent",
    "KeyboardEvent",
    "MessageEvent",
    "CloseEvent",
    "AbortController",
    "AbortSignal",
    "getComputedStyle",
  ] as const)
    vi.stubGlobal(name, dom.window[name]);
  Object.defineProperty(document, "visibilityState", { value: "hidden" });
  dom.window.requestAnimationFrame = (callback: FrameRequestCallback) =>
    setTimeout(() => callback(0), 0);
  dom.window.cancelAnimationFrame = clearTimeout;
  vi.stubGlobal("requestAnimationFrame", dom.window.requestAnimationFrame);
  vi.stubGlobal("cancelAnimationFrame", dom.window.cancelAnimationFrame);
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  testing = await import("@testing-library/react");
  userEvent = (await import("@testing-library/user-event")).default;
  useApplication = (await import("@/App")).useApplication;
  HistorySection = (await import("@/features/members/detail")).HistorySection;
  Modal = (await import("@/components/ui/dialog")).Modal;
  toastModule = await import("@/components/ui/toast");
  TooltipProvider = (await import("@/components/ui/tooltip")).TooltipProvider;
}, 30000);

beforeEach(() => {
  sockets = [];
  connection = new Backend(
    () => ({ url: "ws://localhost/ws?token=test" }),
    () => {
      const socket = new TestSocket();
      sockets.push(socket);
      return socket;
    },
  );
  vi.spyOn(backend, "call").mockImplementation(
    connection.call.bind(connection),
  );
  vi.spyOn(backend, "connect").mockImplementation(
    connection.connect.bind(connection),
  );
  vi.spyOn(backend, "reconnect").mockImplementation(
    connection.reconnect.bind(connection),
  );
  vi.spyOn(backend, "onEvent").mockImplementation(
    connection.onEvent.bind(connection),
  );
  vi.spyOn(backend, "onFailure").mockImplementation(
    connection.onFailure.bind(connection),
  );
  vi.spyOn(backend, "disconnected", "get").mockImplementation(
    () => connection.disconnected,
  );
  vi.spyOn(backend, "reconnecting", "get").mockImplementation(
    () => connection.reconnecting,
  );
});

afterEach(async () => {
  await testing.act(async () => {
    testing.cleanup();
    sockets[sockets.length - 1]?.disconnect();
    toastModule.clearToasts();
    if (vi.isFakeTimers()) await vi.runOnlyPendingTimersAsync();
  });
  vi.restoreAllMocks();
  vi.useRealTimers();
  await new Promise((resolve) => setTimeout(resolve, 0));
});

afterAll(() => {
  closeDom();
  vi.unstubAllGlobals();
});

async function openHistory() {
  await testing.act(async () => testing.render(<Harness />));
  const turn = await testing.screen.findByRole("button", { name: /#4/ });
  await testing.act(async () => testing.fireEvent.click(turn));
  await testing.act(async () =>
    testing.fireEvent.click(
      testing.screen.getByRole("button", { name: "View full context" }),
    ),
  );
  await vi.waitFor(() =>
    expect(sockets[0].requests("agent.history.read")).toHaveLength(1),
  );
  return testing.screen.getByRole("dialog");
}

async function respondHistory(value = history()) {
  await testing.act(async () => {
    await Promise.resolve();
    sockets[sockets.length - 1].respond("agent.history.read", value);
  });
}

async function disconnect() {
  await testing.act(async () => sockets[sockets.length - 1].disconnect());
}

async function reconnect() {
  const button = testing.screen.getByRole("button", { name: "Reconnect" });
  expect(button.closest('[role="dialog"]')).toBeTruthy();
  expect(button.closest('[aria-hidden="true"]')).toBeNull();
  await testing.act(async () => button.focus());
  expect(document.activeElement).toBe(button);
  const asyncWrapper = testing.getConfig().asyncWrapper;
  testing.configure({ asyncWrapper: (callback) => callback() });
  await testing.act(async () => {
    await userEvent.setup({ document }).keyboard("{Enter}");
  });
  testing.configure({ asyncWrapper });
  await vi.waitFor(() => expect(sockets).toHaveLength(2));
  await vi.waitFor(() =>
    expect(sockets[1].requests("agent.history.read")).toHaveLength(1),
  );
}

describe("History connection recovery", () => {
  it("reconnects a pending read with one accessible notification inside the real Modal", async () => {
    const dialog = await openHistory();
    const previous = sockets[0].latest("agent.history.read");
    await disconnect();
    expect(testing.within(dialog).getAllByText("Connection lost")).toHaveLength(
      1,
    );
    expect(
      testing.screen.getAllByRole("button", { name: "Reconnect" }),
    ).toHaveLength(1);
    expect(testing.screen.queryByRole("button", { name: "Retry" })).toBeNull();
    await reconnect();
    await testing.act(async () =>
      sockets[0].reply({
        type: "response",
        id: previous.id,
        result: history(1),
      }),
    );
    expect(testing.screen.queryByText("input-1")).toBeNull();
    await respondHistory(history(2));
    expect(
      testing.screen
        .getByRole("tab", { name: /Request 2/ })
        .getAttribute("aria-selected"),
    ).toBe("true");
    expect(testing.screen.getByText("input-2")).toBeTruthy();
    expect(
      toastModule
        .readToasts()
        .filter((item) => item.open && item.id === "connection"),
    ).toHaveLength(0);
    expect(
      testing.screen.queryByText("Connection lost. Reconnect to continue."),
    ).toBeNull();
  });

  it("keeps the selected request, loaded messages and reading position after reconnecting", async () => {
    const dialog = await openHistory();
    await respondHistory();
    const turnPanel = testing.screen
      .getByText("Saved Turn messages")
      .closest("section");
    if (!turnPanel) throw new Error("Missing Turn messages");
    await testing.act(async () =>
      testing.fireEvent.click(
        testing
          .within(turnPanel)
          .getByRole("button", { name: "Load more messages" }),
      ),
    );
    await vi.waitFor(() =>
      expect(sockets[0].requests("agent.history.read")).toHaveLength(2),
    );
    const more = history();
    more.messages = {
      ...messages(["saved-tail"], 2),
      offset: 1,
      has_more: false,
    };
    await respondHistory(more);
    await testing.act(async () =>
      testing.fireEvent.click(
        testing.screen.getByRole("tab", { name: /Request 2/ }),
      ),
    );
    await respondHistory(history(2));
    const scroll = dialog.querySelector('[class~="overflow-y-auto"]');
    if (!scroll) throw new Error("Missing scroll region");
    scroll.scrollTop = 300;
    await disconnect();
    await reconnect();
    expect(sockets[1].latest("agent.history.read").params.ordinal).toBe(2);
    await respondHistory(history(2));
    expect(scroll.scrollTop).toBe(300);
    expect(testing.screen.getByText("saved-tail")).toBeTruthy();
    expect(testing.screen.getByText("input-2")).toBeTruthy();
  });

  it("keeps application read errors during disconnection and replaces them after a successful recovery", async () => {
    await openHistory();
    await testing.act(async () =>
      sockets[0].reply({
        type: "response",
        id: sockets[0].latest("agent.history.read").id,
        error: { code: "invalid_history", message: "History data unavailable" },
      }),
    );
    expect(testing.screen.getByText("History data unavailable")).toBeTruthy();
    await disconnect();
    expect(testing.screen.getByText("History data unavailable")).toBeTruthy();
    await reconnect();
    expect(testing.screen.getByText("History data unavailable")).toBeTruthy();
    await respondHistory();
    expect(testing.screen.queryByText("History data unavailable")).toBeNull();
  });

  it("retries a timed out read on the open socket with one local error", async () => {
    await openHistory();
    await respondHistory();
    vi.useFakeTimers();
    await testing.act(async () =>
      testing.fireEvent.click(
        testing.screen.getByRole("tab", { name: /Request 2/ }),
      ),
    );
    await testing.act(async () => vi.advanceTimersByTimeAsync(60_000));
    expect(connection.disconnected).toBe(false);
    expect(
      testing.screen.getAllByText("Request timed out. Try reading again."),
    ).toHaveLength(1);
    expect(toastModule.readToasts().filter((item) => item.open)).toHaveLength(
      0,
    );
    await testing.act(async () =>
      testing.fireEvent.click(
        testing.screen.getByRole("button", { name: "Retry" }),
      ),
    );
    await testing.act(async () => vi.advanceTimersByTimeAsync(0));
    await respondHistory();
    expect(testing.screen.getByText("input-1")).toBeTruthy();
    expect(sockets).toHaveLength(1);
  });

  it("retries an initial read failure and opens the saved request", async () => {
    await openHistory();
    await testing.act(async () =>
      sockets[0].reply({
        type: "response",
        id: sockets[0].latest("agent.history.read").id,
        error: { code: "invalid_history", message: "History data unavailable" },
      }),
    );
    await testing.act(async () =>
      testing.fireEvent.click(
        testing.screen.getByRole("button", { name: "Retry" }),
      ),
    );
    await respondHistory(history(2));
    expect(testing.screen.getByText("input-2")).toBeTruthy();
    expect(
      testing.screen
        .getByRole("tab", { name: /Request 2/ })
        .getAttribute("aria-selected"),
    ).toBe("true");
    expect(testing.screen.queryByText("History data unavailable")).toBeNull();
  });

  it.each([
    { recovery: "Retry", windowCount: 34, requestCount: 35 },
    { recovery: "Retry", windowCount: 64, requestCount: 65 },
    { recovery: "Reconnect", windowCount: 34, requestCount: 35 },
    { recovery: "Reconnect", windowCount: 64, requestCount: 65 },
  ])(
    "preserves pagination and content after $recovery with $windowCount windows and $requestCount requests",
    async ({ recovery, windowCount, requestCount }) => {
      const dialog = await openHistory();
      await respondHistory(paginatedHistory(windowCount, requestCount));
      await testing.act(async () =>
        testing.fireEvent.click(
          testing.screen.getByRole("button", { name: "Load more windows" }),
        ),
      );
      await respondHistory(paginatedHistory(windowCount, requestCount, 30));
      await testing.act(async () =>
        testing.fireEvent.click(
          testing.screen.getByRole("button", { name: "Load more requests" }),
        ),
      );
      await respondHistory(paginatedHistory(windowCount, requestCount, 0, 30));
      const savedPanel = testing.screen
        .getByText("Saved Turn messages")
        .closest("section");
      if (!savedPanel) throw new Error("Missing saved messages");
      await testing.act(async () =>
        testing.fireEvent.click(
          testing.within(savedPanel).getByRole("button", {
            name: "Load more messages",
          }),
        ),
      );
      const moreSaved = paginatedHistory(windowCount, requestCount);
      moreSaved.messages = {
        ...messages(["saved-tail"], 2),
        offset: 1,
        has_more: false,
      };
      await respondHistory(moreSaved);
      for (const [title, source] of [
        ["Model input", "input"],
        ["Model response", "response"],
        ["Associated tool results", "related"],
      ] as const) {
        const panel = testing.screen.getByText(title).closest("section");
        if (!panel) throw new Error(`Missing ${title}`);
        await testing.act(async () =>
          testing.fireEvent.click(
            testing.within(panel).getByRole("button", {
              name: "Load more messages",
            }),
          ),
        );
        const more = paginatedHistory(windowCount, requestCount);
        if (!more.request) throw new Error("Missing request");
        more.request[source] = {
          ...messages([`${source}-2-tail`], 2),
          offset: 1,
          has_more: false,
        };
        await respondHistory(more);
      }
      const scroll = dialog.querySelector('[class~="overflow-y-auto"]');
      if (!scroll) throw new Error("Missing scroll region");
      scroll.scrollTop = 300;
      let delayed: Sent | undefined;
      if (recovery === "Retry") {
        vi.useFakeTimers();
        await testing.act(async () =>
          testing.fireEvent.click(
            testing.screen.getByRole("tab", { name: /^Request 2pending$/ }),
          ),
        );
        await testing.act(async () => vi.advanceTimersByTimeAsync(0));
        delayed = sockets[0].latest("agent.history.read");
        await testing.act(async () => vi.advanceTimersByTimeAsync(60_000));
        expect(connection.disconnected).toBe(false);
        expect(
          testing.screen.getAllByText("Request timed out. Try reading again."),
        ).toHaveLength(1);
        await testing.act(async () =>
          testing.fireEvent.click(
            testing.screen.getByRole("button", { name: "Retry" }),
          ),
        );
        await testing.act(async () => vi.advanceTimersByTimeAsync(0));
      } else {
        await disconnect();
        await reconnect();
      }
      const socket = sockets[sockets.length - 1];
      if (!socket) throw new Error("Missing socket");
      expect(socket.latest("agent.history.read").params.ordinal).toBe(2);
      await respondHistory(paginatedHistory(windowCount, requestCount));
      expect(scroll.scrollTop).toBe(300);
      expect(testing.screen.getByText("saved-tail")).toBeTruthy();
      const windowPanel = testing.screen
        .getByText("Context windows")
        .closest("section");
      if (!windowPanel) throw new Error("Missing window events");
      expect(testing.within(windowPanel).getAllByRole("listitem")).toHaveLength(
        Math.min(60, windowCount),
      );
      expect(testing.screen.getAllByRole("tab")).toHaveLength(
        Math.min(60, requestCount),
      );
      for (const source of ["input", "response", "related"])
        expect(testing.screen.getByText(`${source}-2-tail`)).toBeTruthy();
      expect(
        testing.screen
          .getByRole("tab", { name: /^Request 2pending$/ })
          .getAttribute("aria-selected"),
      ).toBe("true");
      expect(
        testing.screen.getByText(`Window ${Math.min(60, windowCount)}`),
      ).toBeTruthy();
      expect(
        testing.screen.getByRole("tab", {
          name: `Request ${Math.min(60, requestCount)}pending`,
        }),
      ).toBeTruthy();
      if (delayed) {
        await testing.act(async () =>
          socket.reply({
            type: "response",
            id: delayed.id,
            result: history(1),
          }),
        );
        expect(testing.screen.getByText("input-2-tail")).toBeTruthy();
        expect(
          testing.screen
            .getByRole("tab", { name: /^Request 2pending$/ })
            .getAttribute("aria-selected"),
        ).toBe("true");
      }
      if (windowCount > 60) {
        await testing.act(async () =>
          testing.fireEvent.click(
            testing.screen.getByRole("button", { name: "Load more windows" }),
          ),
        );
        await testing.act(async () => Promise.resolve());
        expect(socket.latest("agent.history.read").params.windows_after).toBe(
          60,
        );
        await respondHistory(paginatedHistory(windowCount, requestCount, 60));
        expect(testing.screen.getByText(`Window ${windowCount}`)).toBeTruthy();
        await testing.act(async () =>
          testing.fireEvent.click(
            testing.screen.getByRole("button", { name: "Load more requests" }),
          ),
        );
        await testing.act(async () => Promise.resolve());
        expect(socket.latest("agent.history.read").params.request_after).toBe(
          60,
        );
        await respondHistory(
          paginatedHistory(windowCount, requestCount, 0, 60),
        );
        expect(
          testing.screen.getByRole("tab", {
            name: `Request ${requestCount}pending`,
          }),
        ).toBeTruthy();
      }
      expect(
        testing.screen.queryByRole("button", { name: "Load more windows" }),
      ).toBeNull();
      expect(
        testing.screen.queryByRole("button", { name: "Load more requests" }),
      ).toBeNull();
      expect(testing.screen.getByText("input-2-tail")).toBeTruthy();
      expect(
        testing.screen.queryByText("Request timed out. Try reading again."),
      ).toBeNull();
    },
  );

  it("resumes interrupted long text and image reads without repeating the text prefix", async () => {
    await openHistory();
    const value = history();
    if (!value.request) throw new Error("Missing request");
    value.request.input.messages = [
      {
        kind: "binary",
        source: "input",
        request_ordinal: 1,
        message_index: 0,
        path: [],
        media_type: "image/png",
        size: 3,
      },
    ];
    await respondHistory(value);
    await testing.act(async () =>
      testing.fireEvent.click(
        testing.screen.getByRole("button", { name: "Load more text" }),
      ),
    );
    await vi.waitFor(() =>
      expect(sockets[0].requests("agent.history.text")).toHaveLength(1),
    );
    await vi.waitFor(() =>
      expect(sockets[0].requests("agent.history.image")).toHaveLength(1),
    );
    await disconnect();
    expect(testing.screen.queryByText(/Could not load image/)).toBeNull();
    await reconnect();
    await respondHistory(value);
    await vi.waitFor(() =>
      expect(sockets[1].requests("agent.history.text")).toHaveLength(1),
    );
    expect(sockets[1].latest("agent.history.text").params.offset).toBe(10);
    await testing.act(async () => {
      sockets[1].respond("agent.history.text", {
        value: "-tail",
        next_offset: 20,
        has_more: false,
      });
      sockets[1].respond("agent.history.image", {
        media_type: "image/png",
        data: "YWJj",
      });
    });
    expect(testing.screen.getByText("prefix-1-tail")).toBeTruthy();
    expect(
      testing.screen
        .getByRole("img", { name: "Saved model content" })
        .getAttribute("src"),
    ).toBe("data:image/png;base64,YWJj");
  });

  it("provides Retry for an image timeout", async () => {
    await openHistory();
    const value = history();
    if (!value.request) throw new Error("Missing request");
    value.request.input.messages = [
      {
        kind: "binary",
        source: "input",
        request_ordinal: 1,
        message_index: 0,
        path: [],
        media_type: "image/png",
        size: 3,
      },
    ];
    vi.useFakeTimers();
    await respondHistory(value);
    await testing.act(async () => vi.advanceTimersByTimeAsync(60_000));
    expect(
      testing.screen.getAllByText(/Could not load image: Request timed out/),
    ).toHaveLength(1);
    await testing.act(async () =>
      testing.fireEvent.click(
        testing.screen.getByRole("button", { name: "Retry image" }),
      ),
    );
    await testing.act(async () => vi.advanceTimersByTimeAsync(0));
    await testing.act(async () =>
      sockets[0].respond("agent.history.image", {
        media_type: "image/png",
        data: "YWJj",
      }),
    );
    expect(
      testing.screen.getByRole("img", { name: "Saved model content" }),
    ).toBeTruthy();
  });

  it("moves the existing notification through nested Modals and back to the page", async () => {
    vi.useFakeTimers();
    const view = (first: boolean, second: boolean) => (
      <TooltipProvider>
        <Modal
          open={first}
          onOpenChange={() => {}}
          title="Create Agent"
          footer={<button type="button">Create</button>}
        >
          <Modal
            open={second}
            onOpenChange={() => {}}
            title="History image"
            footer={<button type="button">Close image</button>}
          />
        </Modal>
        <toastModule.Toaster />
      </TooltipProvider>
    );
    let result!: ReturnType<typeof testing.render>;
    await testing.act(async () => {
      result = testing.render(view(true, false));
    });
    await testing.act(async () =>
      toastModule.toast({
        id: "connection",
        title: "Connection lost",
        tone: "danger",
        duration: null,
        closable: false,
        action: { label: "Reconnect", onClick: vi.fn() },
      }),
    );
    expect(
      testing.screen
        .getByRole("button", { name: "Reconnect" })
        .closest('[role="dialog"]')
        ?.getAttribute("aria-labelledby"),
    ).toBe(testing.screen.getByText("Create Agent").id);
    await testing.act(async () => result.rerender(view(true, true)));
    expect(
      testing.screen
        .getByRole("button", { name: "Reconnect" })
        .closest('[role="dialog"]')
        ?.getAttribute("aria-labelledby"),
    ).toBe(testing.screen.getByText("History image").id);
    await testing.act(async () => result.rerender(view(true, false)));
    expect(
      testing.screen
        .getByRole("button", { name: "Reconnect" })
        .closest('[role="dialog"]')
        ?.getAttribute("aria-labelledby"),
    ).toBe(testing.screen.getByText("Create Agent").id);
    await testing.act(async () => result.rerender(view(false, false)));
    expect(
      testing.screen
        .getByRole("button", { name: "Reconnect" })
        .closest('[role="dialog"]'),
    ).toBeNull();
    expect(testing.screen.getAllByText("Connection lost")).toHaveLength(1);
    await testing.act(async () => vi.runOnlyPendingTimersAsync());
  });
});
