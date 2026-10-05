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
import { backend, type ModelCatalog } from "@/lib/backend";

let dom: {
  window: Window & {
    close: () => void;
    HTMLElement: typeof HTMLElement;
    Node: typeof Node;
    MutationObserver: typeof MutationObserver;
    Event: typeof Event;
    EventTarget: typeof EventTarget;
    CustomEvent: typeof CustomEvent;
    MouseEvent: typeof MouseEvent;
    KeyboardEvent: typeof KeyboardEvent;
    AbortController: typeof AbortController;
    AbortSignal: typeof AbortSignal;
  };
};
let act: typeof import("@testing-library/react").act;
let cleanup: typeof import("@testing-library/react").cleanup;
let fireEvent: typeof import("@testing-library/react").fireEvent;
let render: typeof import("@testing-library/react").render;
let screen: typeof import("@testing-library/react").screen;
let waitFor: typeof import("@testing-library/react").waitFor;
let within: typeof import("@testing-library/react").within;
let App: typeof import("@/App").default;
let RouterProvider: typeof import("@/app/router").RouterProvider;
let SettingsPage: typeof import("@/features/settings/page").SettingsPage;
let renderToStaticMarkup: typeof import("react-dom/server").renderToStaticMarkup;
let isSettingsSaving: typeof import("@/features/settings/saver").isSettingsSaving;

beforeAll(async () => {
  const packageName = "jsdom";
  const { JSDOM } = await import(packageName);
  dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://localhost",
  });
  vi.stubGlobal("window", dom.window);
  vi.stubGlobal("document", dom.window.document);
  vi.stubGlobal("navigator", dom.window.navigator);
  vi.stubGlobal("HTMLElement", dom.window.HTMLElement);
  vi.stubGlobal("Node", dom.window.Node);
  vi.stubGlobal("MutationObserver", dom.window.MutationObserver);
  vi.stubGlobal("Event", dom.window.Event);
  vi.stubGlobal("EventTarget", dom.window.EventTarget);
  vi.stubGlobal("CustomEvent", dom.window.CustomEvent);
  vi.stubGlobal("MouseEvent", dom.window.MouseEvent);
  vi.stubGlobal("KeyboardEvent", dom.window.KeyboardEvent);
  vi.stubGlobal("AbortController", dom.window.AbortController);
  vi.stubGlobal("AbortSignal", dom.window.AbortSignal);
  vi.stubGlobal("getComputedStyle", dom.window.getComputedStyle);
  dom.window.requestAnimationFrame = vi.fn();
  dom.window.cancelAnimationFrame = vi.fn();
  await import("react");
  ({ isSettingsSaving } = await import("@/features/settings/saver"));
  const testing = await import("@testing-library/react");
  act = testing.act;
  cleanup = testing.cleanup;
  fireEvent = testing.fireEvent;
  render = testing.render;
  screen = testing.screen;
  waitFor = testing.waitFor;
  within = testing.within;
  ({ default: App } = await import("@/App"));
  ({ RouterProvider } = await import("@/app/router"));
  ({ SettingsPage } = await import("@/features/settings/page"));
  ({ renderToStaticMarkup } = await import("react-dom/server"));
}, 30000);

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const modelCatalog: ModelCatalog = {
  version: 2,
  providers: [
    {
      id: "provider-a",
      name: "Provider A",
      api_type: "openai-chat",
      base_url: "http://127.0.0.1:1",
      api_key_set: false,
      enabled: false,
    },
    {
      id: "provider-b",
      name: "Provider B",
      api_type: "openai-chat",
      base_url: "http://127.0.0.1:1",
      api_key_set: false,
      enabled: false,
    },
  ],
  models: [],
  default_model_id: null,
  default_thinking: "default",
  agent_configs: {},
};

const agentSettings = {
  context_window_tokens: 8192,
  memory_index_bytes: 64000,
  exchange_nudge_after: 24,
  idle_streak_after: 3,
  no_tool_turns_before_pause: 5,
  max_concurrent_turns: 4,
  token_limit: 100000,
  request_limit: 0,
};

class ResizeObserverStub {
  observe() {}

  unobserve() {}

  disconnect() {}
}

function getModelForm(element: HTMLElement): HTMLFormElement {
  const form = element.closest("form");
  if (!form) throw new Error("Model form is missing");
  return form;
}

async function getModelsSection(settings: ReturnType<typeof within>) {
  const heading = await settings.findByRole("heading", { name: "Models" });
  const section = heading.closest("section");
  if (!section) throw new Error("Models section is missing");
  return within(section);
}

function getAddModelToggle(
  models: ReturnType<typeof within>,
  expanded: boolean,
): HTMLElement {
  return models.getByRole("button", { name: "Add model", expanded });
}

async function prepareApplication() {
  vi.spyOn(backend, "connect").mockResolvedValue();
  vi.spyOn(backend, "onEvent").mockReturnValue(vi.fn());
  vi.spyOn(backend, "onFailure").mockReturnValue(vi.fn());
  vi.spyOn(backend, "organization").mockResolvedValue({
    id: 1,
    uuid: "test-organization",
    members: [],
    human_id: 1,
    token_limit: null,
  });
  vi.spyOn(backend, "discussions").mockResolvedValue([]);
  vi.spyOn(backend, "modelCatalog").mockResolvedValue(modelCatalog);
  vi.spyOn(backend, "settings").mockImplementation(async (section) => {
    if (section === "agent") return agentSettings;
    if (section === "observability") {
      return {
        enabled: false,
        base_url: "https://cloud.langfuse.com",
        keys_set: true,
      };
    }
    return {};
  });
  vi.stubGlobal("ResizeObserver", ResizeObserverStub);
  vi.stubGlobal("requestAnimationFrame", vi.fn());
  await act(async () => {
    render(<App />);
  });
}

async function getSettingsNavigation() {
  const navigation = await screen.findByLabelText("Settings", {
    selector: "nav",
  });
  return within(navigation).getByRole("button", { name: "Settings" });
}

async function openSettings(section: "model" | "agent") {
  fireEvent.click(await getSettingsNavigation());
  const tablist = await screen.findByRole("tablist", { name: "Settings" });
  const tabs = within(tablist);
  const tab = await tabs.findByRole("tab", { name: "Model" });
  if (section === "agent") {
    fireEvent.click(tabs.getByRole("tab", { name: "Agent" }));
    await screen.findByLabelText("Model requests per Turn");
  }
  return tab;
}

function flushScheduledWork() {
  return new Promise<void>((resolve) => setImmediate(resolve));
}

afterEach(async () => {
  cleanup();
  await act(async () => {
    await flushScheduledWork();
  });
  vi.restoreAllMocks();
});

afterAll(async () => {
  await act(async () => {
    await flushScheduledWork();
  });
  dom.window.close();
  vi.unstubAllGlobals();
});

describe("Settings save navigation", () => {
  it("stays blocked until every active save finishes", () => {
    expect(isSettingsSaving({})).toBe(false);
    expect(isSettingsSaving({ model: true, agent: false })).toBe(true);
    expect(isSettingsSaving({ model: false, agent: false })).toBe(false);
  });
});

describe("mounted Settings interactions", () => {
  beforeEach(prepareApplication);

  it("resets new Model fields between Providers and retains them on collapse", async () => {
    await openSettings("model");
    const settings = within(
      await screen.findByRole("group", { name: "Model settings" }),
    );
    const provider = await settings.findByLabelText("Provider");
    fireEvent.change(provider, { target: { value: "provider-a" } });

    let models = await getModelsSection(settings);
    fireEvent.click(getAddModelToggle(models, false));
    const name = await models.findByLabelText("Model name");
    const form = getModelForm(name);
    const remote = within(form).getByRole("combobox", {
      name: "Remote model ID",
    });
    const enabled = within(form).getByRole("checkbox", { name: "Enabled" });

    act(() => {
      fireEvent.change(name, { target: { value: "A draft" } });
      fireEvent.change(remote, { target: { value: "remote-a" } });
      fireEvent.click(enabled);
    });

    fireEvent.change(provider, { target: { value: "provider-b" } });
    models = await getModelsSection(settings);
    fireEvent.click(getAddModelToggle(models, false));
    const nameB = await models.findByLabelText("Model name");
    const formB = getModelForm(nameB);
    expect((nameB as HTMLInputElement).value).toBe("");
    expect(
      (
        within(formB).getByRole("combobox", {
          name: "Remote model ID",
        }) as HTMLInputElement
      ).value,
    ).toBe("");
    expect(
      (
        within(formB).getByRole("checkbox", {
          name: "Enabled",
        }) as HTMLInputElement
      ).checked,
    ).toBe(true);

    fireEvent.change(provider, { target: { value: "provider-a" } });
    models = await getModelsSection(settings);
    fireEvent.click(getAddModelToggle(models, false));
    const nameA = await models.findByLabelText("Model name");
    const formA = getModelForm(nameA);
    expect((nameA as HTMLInputElement).value).toBe("");
    const remoteA = within(formA).getByRole("combobox", {
      name: "Remote model ID",
    }) as HTMLInputElement;
    expect(remoteA.value).toBe("");
    const enabledA = within(formA).getByRole("checkbox", {
      name: "Enabled",
    }) as HTMLInputElement;
    expect(enabledA.checked).toBe(true);

    act(() => {
      fireEvent.change(nameA, { target: { value: "Retained draft" } });
      fireEvent.change(remoteA, { target: { value: "retained-remote" } });
      fireEvent.click(enabledA);
    });
    const toggleA = getAddModelToggle(models, true);
    fireEvent.click(toggleA);
    expect(toggleA.getAttribute("aria-expanded")).toBe("false");
    expect(toggleA.isConnected).toBe(true);
    fireEvent.click(toggleA);
    expect(toggleA.getAttribute("aria-expanded")).toBe("true");

    const retainedName = await models.findByLabelText("Model name");
    expect((retainedName as HTMLInputElement).value).toBe("Retained draft");
    expect(getModelForm(retainedName)).toBe(formA);
    expect(remoteA.isConnected).toBe(true);
    expect(enabledA.isConnected).toBe(true);
    expect(remoteA.value).toBe("retained-remote");
    expect(enabledA.checked).toBe(false);
  });

  it("blocks Tabs and Sidebar through delayed write and readback", async () => {
    const readback = deferred<Record<string, unknown>>();
    const update = deferred<Record<string, unknown>>();
    let agentReads = 0;
    vi.spyOn(backend, "settings").mockImplementation(async (section) => {
      if (section === "agent") {
        agentReads += 1;
        return agentReads === 1 ? agentSettings : readback.promise;
      }
      return {};
    });
    vi.spyOn(backend, "updateSettings").mockReturnValueOnce(update.promise);

    await openSettings("agent");
    const requestLimit = screen.getByLabelText("Model requests per Turn");
    fireEvent.change(requestLimit, { target: { value: "7" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    const agentTab = screen.getByRole("tab", { name: "Agent" });
    const settingsNavigation = await getSettingsNavigation();
    await waitFor(() => {
      expect((agentTab as HTMLButtonElement).disabled).toBe(true);
      expect(settingsNavigation.closest("[inert]")).not.toBeNull();
    });

    await act(async () => {
      update.resolve({});
      await update.promise;
    });
    await waitFor(() => expect(agentReads).toBe(2));
    expect((agentTab as HTMLButtonElement).disabled).toBe(true);
    expect(settingsNavigation.closest("[inert]")).not.toBeNull();

    await act(async () => {
      readback.resolve({ ...agentSettings, request_limit: 7 });
      await readback.promise;
    });
    await waitFor(() => {
      expect((agentTab as HTMLButtonElement).disabled).toBe(false);
      expect(settingsNavigation.closest("[inert]")).toBeNull();
    });
    expect((requestLimit as HTMLInputElement).value).toBe("7");
  });

  it("restores navigation and preserves the draft after a failed write", async () => {
    vi.spyOn(backend, "updateSettings").mockRejectedValueOnce(
      new Error("offline"),
    );
    await openSettings("agent");
    const requestLimit = screen.getByLabelText("Model requests per Turn");
    fireEvent.change(requestLimit, { target: { value: "9" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await screen.findByText("Could not save");
    expect(
      (screen.getByRole("tab", { name: "Agent" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
    expect((await getSettingsNavigation()).closest("[inert]")).toBeNull();
    expect((requestLimit as HTMLInputElement).value).toBe("9");
  });
});

describe("Settings page", () => {
  it("shows one tab per section and the requested panel", () => {
    const html = renderToStaticMarkup(
      <RouterProvider>
        <SettingsPage section="agent" />
      </RouterProvider>,
    );
    expect(html).toMatch(/<h1\b[^>]*>Settings<\/h1>/);
    for (const label of ["Model", "Execution", "Agent", "Langfuse"]) {
      expect(html).toContain(`>${label}</button>`);
    }
    expect(html).toContain('aria-selected="true"');
    expect(html).toContain("0 means no ceiling.");
    expect(html).toContain('aria-label="Agent settings"');
    expect(html).toContain(">Context</h3>");
    expect(html).toContain(">Run limits</h3>");
    expect(html).toContain(">Reminders and pausing</h3>");
    expect(html.match(/<form/g)).toHaveLength(1);
    expect(html).not.toContain(">Limits</button>");
    expect(
      Array.from(
        html.matchAll(/<p\b[^>]*>([\s\S]*?)<\/p>/g),
        ([, text]) => text,
      ),
    ).toEqual([
      "0 means no ceiling.",
      "0 means unlimited. Changes apply to new Turns.",
    ]);
  });

  it("shows only remote voice settings and the microphone test", () => {
    const html = renderToStaticMarkup(
      <RouterProvider>
        <SettingsPage section="voice" />
      </RouterProvider>,
    );
    expect(html).toContain(">Service address</label>");
    expect(html).toContain(">Model</label>");
    expect(html).toContain(">API key</label>");
    expect(html).toContain("Start microphone test");
    expect(html).not.toContain("Whisper");
    expect(html).not.toContain("Download model");
    expect(html).not.toContain("<select");
  });
});
