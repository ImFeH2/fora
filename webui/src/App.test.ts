import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useApplication } from "@/App";
import { clearToasts, readToasts } from "@/components/ui/toast";
import { BackendError, type BackendEvent, backend } from "@/lib/backend";

const hooks = vi.hoisted(() => ({
  states: [] as unknown[],
  refs: [] as { current: unknown }[],
  effects: [] as (() => undefined | (() => void))[],
  stateIndex: 0,
  refIndex: 0,
  mounting: true,
}));
vi.mock("react", async (original) => ({
  ...(await original<typeof import("react")>()),
  useState: (initial: unknown) => {
    const index = hooks.stateIndex++;
    if (hooks.mounting) hooks.states[index] = initial;
    return [
      hooks.states[index],
      (value: unknown) => {
        hooks.states[index] = value;
      },
    ];
  },
  useRef: (initial: unknown) => {
    const index = hooks.refIndex++;
    if (hooks.mounting) hooks.refs[index] = { current: initial };
    return hooks.refs[index];
  },
  useCallback: (callback: unknown) => callback,
  useEffect: (effect: () => undefined | (() => void)) => {
    if (hooks.mounting) hooks.effects.push(effect);
  },
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}
const organization = {
  id: 1,
  uuid: "test-organization",
  members: [],
  human_id: 1,
  token_limit: 10,
};
let events: Set<(event: BackendEvent) => void>;
let failures: Set<(error: BackendError) => void>;
let cleanup: (() => void)[];
let closed: boolean;
let page: EventTarget & { visibilityState: string };
let network: { onLine: boolean };

function render() {
  hooks.stateIndex = 0;
  hooks.refIndex = 0;
  return useApplication();
}
function mount() {
  const application = render();
  hooks.mounting = false;
  for (const effect of hooks.effects) {
    const off = effect();
    if (off) cleanup.push(off);
  }
  return application;
}
function emit(event: BackendEvent) {
  for (const listener of events) listener(event);
}
function notifyFailure(
  error = new BackendError("disconnected", "Connection lost", true),
) {
  closed = true;
  for (const listener of failures) listener(error);
  return error;
}
function disconnect(
  error = new BackendError("disconnected", "Connection lost", true),
) {
  notifyFailure(error);
  emit({ type: "connection.closed", error });
}
async function settle() {
  for (let i = 0; i < 30; i++) await Promise.resolve();
}

beforeEach(() => {
  hooks.states = [];
  hooks.refs = [];
  hooks.effects = [];
  hooks.mounting = true;
  cleanup = [];
  events = new Set();
  failures = new Set();
  closed = false;
  clearToasts();
  page = Object.assign(new EventTarget(), { visibilityState: "visible" });
  network = { onLine: true };
  vi.stubGlobal("window", new EventTarget());
  vi.stubGlobal("document", page);
  vi.stubGlobal("navigator", network);
  vi.spyOn(backend, "disconnected", "get").mockImplementation(() => closed);
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    events.add(listener);
    return () => {
      events.delete(listener);
    };
  });
  vi.spyOn(backend, "onFailure").mockImplementation((listener) => {
    failures.add(listener);
    return () => {
      failures.delete(listener);
    };
  });
  vi.spyOn(backend, "connect").mockResolvedValue();
  vi.spyOn(backend, "reconnect").mockImplementation(async () => {
    if (closed) {
      emit({ type: "connection.reconnecting" });
      closed = false;
      emit({ type: "connection.restored" });
    }
  });
  vi.spyOn(backend, "organization").mockResolvedValue(organization);
  vi.spyOn(backend, "discussions").mockResolvedValue([]);
});
afterEach(() => {
  for (const off of cleanup) off();
  clearToasts();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("application recovery with controlled hook effects", () => {
  it.each([
    ["organization.changed", "scheduler_stopped", 10],
    ["settings.updated", "model_unavailable", 10],
    ["settings.updated", "token_limit", 1],
  ] as const)("refreshes %s for %s", async (type, code, tokenLimit) => {
    mount();
    await settle();
    const members = [
      {
        id: 2,
        name: "Main",
        type: "agent" as const,
        state: "blocked" as const,
        pause_requested: false,
        reasons: [{ code, message: code, recovery: "Review settings" }],
      },
    ];
    vi.mocked(backend.organization).mockResolvedValueOnce({
      ...organization,
      members,
      token_limit: tokenLimit,
    });
    emit({ type });
    await settle();
    expect(backend.organization).toHaveBeenCalledTimes(2);
    expect(render().loaded?.members).toEqual(members);
    expect(render().loaded?.tokenLimit).toBe(tokenLimit);
  });

  it("merges organization and settings changes during an active read", async () => {
    const first = deferred<typeof organization>();
    vi.mocked(backend.organization)
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce({ ...organization, token_limit: 20 });
    mount();
    await settle();
    emit({ type: "organization.changed" });
    emit({ type: "settings.updated" });
    emit({ type: "organization.changed" });
    expect(backend.organization).toHaveBeenCalledTimes(1);
    first.resolve(organization);
    await settle();
    expect(backend.organization).toHaveBeenCalledTimes(2);
    expect(render().loaded?.tokenLimit).toBe(20);
  });

  it.each(["internal_error", "timeout"])(
    "retries an initial %s on an open connection",
    async (code) => {
      const error = new BackendError(code, "Read failed", code === "timeout");
      vi.mocked(backend.organization).mockRejectedValueOnce(error);
      mount();
      await settle();
      expect(render().failure).toBe(error);
      await render().retry();
      await settle();
      expect(render().loaded?.humanId).toBe(1);
      expect(render().failure).toBeNull();
      expect(backend.organization).toHaveBeenCalledTimes(2);
      expect(backend.reconnect).not.toHaveBeenCalled();
    },
  );

  it.each(["connection_timeout", "connection_failed"] as const)(
    "waits for %s recovery before showing a connection notice",
    async (code) => {
      mount();
      await settle();
      const error = new BackendError(code, "Connection failed", true);
      notifyFailure(error);
      expect(
        readToasts().find((item) => item.id === "connection"),
      ).toBeUndefined();
      emit({ type: "connection.closed", error });
      await settle();
      expect(closed).toBe(false);
      expect(
        readToasts().find((item) => item.id === "connection"),
      ).toBeUndefined();
    },
  );

  it.each(["connection_timeout", "connection_failed"] as const)(
    "reports one %s notice after automatic recovery fails",
    async (code) => {
      mount();
      await settle();
      const error = new BackendError(code, "Connection failed", true);
      let automaticAttempted = false;
      vi.mocked(backend.reconnect).mockImplementation((automatic = false) => {
        if (!automatic || automaticAttempted) return Promise.resolve();
        automaticAttempted = true;
        emit({ type: "connection.reconnecting" });
        return Promise.resolve().then(() => {
          notifyFailure(error);
          emit({ type: "connection.closed", error });
          return Promise.reject(error);
        });
      });
      notifyFailure(error);
      emit({ type: "connection.closed", error });
      await settle();
      expect(render().reconnecting).toBe(false);
      expect(
        readToasts().filter((item) => item.open && item.id === "connection"),
      ).toHaveLength(1);
    },
  );

  it("keeps an unconfirmed write result visible during disconnect recovery", async () => {
    mount();
    await settle();
    const uncertain = new BackendError(
      "unconfirmed",
      "Check the current operation before retrying.",
      true,
    );
    notifyFailure(uncertain);
    emit({
      type: "connection.closed",
      error: new BackendError("disconnected", "Connection lost", true),
    });
    await settle();
    expect(
      readToasts().find(
        (item) => item.open && item.title === "Check operation result",
      ),
    ).toMatchObject({
      title: "Check operation result",
      description: "Check the current operation before retrying.",
    });
  });

  it.each(["close-first", "visible-first"])(
    "recovers with %s ordering",
    async (order) => {
      page.visibilityState = "hidden";
      mount();
      await settle();
      if (order === "close-first") disconnect();
      page.visibilityState = "visible";
      page.dispatchEvent(new Event("visibilitychange"));
      if (order === "visible-first") disconnect();
      await settle();
      expect(closed).toBe(false);
      expect(backend.organization).toHaveBeenCalledTimes(2);
    },
  );

  it("waits for both online and visible before automatic recovery", async () => {
    mount();
    await settle();
    network.onLine = false;
    disconnect();
    await settle();
    expect(backend.reconnect).not.toHaveBeenCalled();
    page.visibilityState = "hidden";
    network.onLine = true;
    window.dispatchEvent(new Event("online"));
    await settle();
    expect(backend.reconnect).not.toHaveBeenCalled();
    page.visibilityState = "visible";
    page.dispatchEvent(new Event("visibilitychange"));
    await settle();
    expect(closed).toBe(false);
  });

  it("subscribes before a synchronous startup failure", async () => {
    vi.mocked(backend.connect).mockImplementationOnce(() => {
      expect(events.size).toBeGreaterThan(0);
      disconnect();
      return Promise.reject(new BackendError("disconnected", "Closed", true));
    });
    mount();
    await settle();
    expect(render().loaded?.humanId).toBe(1);
    expect(render().failure).toBeNull();
  });

  it("keeps initialization failure visible while a retry is loading", async () => {
    vi.mocked(backend.organization).mockRejectedValueOnce(
      new BackendError("internal_error", "Read failed"),
    );
    mount();
    await settle();
    const next = deferred<typeof organization>();
    vi.mocked(backend.organization).mockReturnValueOnce(next.promise);
    const retry = render().retry();
    await settle();
    expect(render().failure?.code).toBe("internal_error");
    expect(render().loading).toBe(true);
    next.resolve(organization);
    await retry;
    expect(render().failure).toBeNull();
    expect(render().loading).toBe(false);
  });

  it("invalidates old reads without letting their finally clear the new request", async () => {
    const old = deferred<typeof organization>();
    vi.mocked(backend.organization).mockReturnValueOnce(old.promise);
    mount();
    await settle();
    const next = deferred<typeof organization>();
    vi.mocked(backend.organization).mockReturnValueOnce(next.promise);
    disconnect();
    await settle();
    old.resolve({ ...organization, human_id: 99 });
    await settle();
    expect(render().loaded).toBeNull();
    const retry = render().retry();
    await settle();
    expect(backend.organization).toHaveBeenCalledTimes(2);
    next.resolve(organization);
    await retry;
    await settle();
    expect(render().loaded?.humanId).toBe(1);
  });

  it("merges events during a read and retains loaded data on a failed follow-up", async () => {
    const first = deferred<typeof organization>();
    vi.mocked(backend.organization)
      .mockReturnValueOnce(first.promise)
      .mockRejectedValueOnce(
        new BackendError("internal_error", "Follow-up failed"),
      );
    mount();
    await settle();
    emit({ type: "member.updated" });
    emit({ type: "discussion.updated" });
    first.resolve(organization);
    await settle();
    expect(render().loaded?.humanId).toBe(1);
    expect(backend.organization).toHaveBeenCalledTimes(2);
    const failure = readToasts().find(
      (item) => item.open && item.action?.label === "Retry",
    );
    expect(failure?.duration).toBeNull();
    failure?.action?.onClick();
    await settle();
    expect(backend.organization).toHaveBeenCalledTimes(3);
    expect(readToasts().find((item) => item.id === failure?.id)?.open).toBe(
      false,
    );
  });

  it("shows a retryable read error after connection restoration", async () => {
    mount();
    await settle();
    const snapshot = render().loaded;
    vi.mocked(backend.discussions).mockRejectedValueOnce(
      new BackendError("timeout", "Read timed out", true),
    );
    disconnect();
    await settle();
    expect(render().loaded).toBe(snapshot);
    expect(render().failure?.code).toBe("timeout");
    const failure = readToasts().find(
      (item) => item.open && item.id === "application-data",
    );
    expect(failure?.duration).toBeNull();
    expect(failure?.action?.label).toBe("Retry");
    failure?.action?.onClick();
    await settle();
    expect(render().failure).toBeNull();
    expect(backend.discussions).toHaveBeenCalledTimes(3);
  });

  it("coalesces repeated retry and visibility events during a read", async () => {
    mount();
    await settle();
    const pending = deferred<typeof organization>();
    vi.mocked(backend.organization).mockReturnValueOnce(pending.promise);
    const first = render().retry();
    expect(render().retry()).toBe(first);
    page.dispatchEvent(new Event("visibilitychange"));
    window.dispatchEvent(new Event("online"));
    await settle();
    expect(backend.organization).toHaveBeenCalledTimes(2);
    pending.resolve(organization);
    await first;
    expect(render().failure).toBeNull();
  });

  it("ignores results after unmount and removes lifecycle subscriptions", async () => {
    const pending = deferred<typeof organization>();
    vi.mocked(backend.organization).mockReturnValueOnce(pending.promise);
    mount();
    await settle();
    for (const off of cleanup.splice(0)) off();
    pending.resolve(organization);
    await settle();
    expect(render().loaded).toBeNull();
    expect(events.size).toBe(0);
    expect(failures.size).toBe(0);
    closed = true;
    window.dispatchEvent(new Event("online"));
    expect(backend.reconnect).not.toHaveBeenCalled();
  });
});
