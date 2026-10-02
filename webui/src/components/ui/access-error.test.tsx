import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { BackendError } from "@/lib/backend";

let dom: {
  window: Window & {
    close: () => void;
    HTMLElement: typeof HTMLElement;
    Node: typeof Node;
    MutationObserver: typeof MutationObserver;
  };
};
let react: typeof import("react");
let testing: typeof import("@testing-library/react");
let AccessError: typeof import("@/components/ui/access-error").AccessError;

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
  vi.stubGlobal("getComputedStyle", dom.window.getComputedStyle);
  react = await import("react");
  testing = await import("@testing-library/react");
  ({ AccessError } = await import("@/components/ui/access-error"));
}, 30000);

afterEach(() => testing?.cleanup());

afterAll(() => {
  dom.window.close();
  vi.unstubAllGlobals();
});

describe("AccessError", () => {
  it.each([
    ["authentication_missing", "Access to Fora requires authentication"],
    ["authentication_failed", "Access to Fora requires authentication"],
    ["startup_failed", "Unable to start Fora"],
    ["connection_failed", "Could not connect to Fora"],
    ["protocol_error", "Fora sent an invalid response"],
    ["invalid_connection", "Unable to prepare the connection"],
  ] as const)(
    "renders the %s message with linked alert text",
    (code, title) => {
      const message = "Connection detail with a second line";
      testing.render(
        react.createElement(AccessError, {
          error: new BackendError(code, message, true),
          reconnect: () => {},
        }),
      );
      const alert = testing.screen.getByRole("alert", { name: title });
      const heading = testing
        .within(alert)
        .getByRole("heading", { name: title });
      const description = alert.getAttribute("aria-describedby");
      if (!description) throw new Error("Alert description is not linked");
      expect(document.getElementById(description)?.textContent).toContain(
        message,
      );
      expect(heading.id).toBe(alert.getAttribute("aria-labelledby"));
      expect(testing.within(alert).getByText(message).textContent).toBe(
        message,
      );
      expect(alert.className).toContain("overflow-wrap:anywhere");
    },
  );

  it("disables controls during loading and reconnect waits", () => {
    testing.render(
      react.createElement(AccessError, {
        error: new BackendError("connection_failed", "Could not connect", true),
        reconnect: () => {},
        waiting: true,
        loadingData: true,
      }),
    );
    expect(
      testing.screen
        .getByRole("button", { name: /Loading/ })
        .hasAttribute("disabled"),
    ).toBe(true);
    testing.cleanup();
    testing.render(
      react.createElement(AccessError, {
        error: new BackendError("connection_failed", "Could not connect", true),
        reconnect: () => {},
        waiting: true,
      }),
    );
    expect(
      testing.screen
        .getByRole("button", { name: /Reconnecting/ })
        .hasAttribute("disabled"),
    ).toBe(true);
  });

  it("activates the retry callback and preserves startup recovery guidance", () => {
    const reconnect = vi.fn();
    testing.render(
      react.createElement(AccessError, {
        error: new BackendError("connection_failed", "Could not connect", true),
        reconnect,
        loadingData: true,
      }),
    );
    testing.fireEvent.click(
      testing.screen.getByRole("button", { name: "Retry" }),
    );
    expect(reconnect).toHaveBeenCalledOnce();
    testing.cleanup();
    testing.render(
      react.createElement(AccessError, {
        error: new BackendError("startup_failed", "Kernel failed", true),
        reconnect: () => {},
      }),
    );
    expect(
      testing.screen.getByText("Close this window and start Fora again."),
    ).toBeTruthy();
    expect(testing.screen.queryByRole("button")).toBeNull();
  });
});
