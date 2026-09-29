import { JSDOM } from "jsdom";
import type { ReactNode } from "react";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";

vi.mock("@radix-ui/react-dialog", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@radix-ui/react-dialog")>()),
  Portal: ({ children }: { children?: ReactNode }) => children,
}));

import { Modal } from "@/components/ui/dialog";
import { Button } from "@/components/ui/index";
import { TooltipProvider } from "@/components/ui/tooltip";

describe("Modal", () => {
  let cleanup: typeof import("@testing-library/react").cleanup;
  let render: typeof import("@testing-library/react").render;
  let screen: typeof import("@testing-library/react").screen;
  let closeDom: (() => void) | undefined;

  beforeAll(async () => {
    const dom = new JSDOM("<!doctype html><html><body></body></html>", {
      url: "http://localhost",
    });
    Object.defineProperty(dom.window, "innerWidth", {
      configurable: true,
      value: 1440,
    });
    Object.defineProperty(dom.window, "innerHeight", {
      configurable: true,
      value: 675,
    });
    closeDom = () => dom.window.close();
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
    vi.stubGlobal("requestAnimationFrame", dom.window.requestAnimationFrame);
    vi.stubGlobal("cancelAnimationFrame", dom.window.cancelAnimationFrame);
    const testing = await import("@testing-library/react");
    cleanup = testing.cleanup;
    render = testing.render;
    screen = testing.screen;
  }, 30000);

  afterEach(async () => {
    cleanup();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  afterAll(() => {
    closeDom?.();
    vi.unstubAllGlobals();
  });

  it("bounds long content to a viewport scroll region", async () => {
    render(
      <TooltipProvider>
        <Modal
          open
          onOpenChange={() => {}}
          title="History"
          wide
          footer={<Button>Close</Button>}
        >
          <div style={{ minHeight: "2000px" }}>Long history</div>
        </Modal>
      </TooltipProvider>,
    );

    const dialog = await screen.findByRole("dialog");
    const scrollRegion = dialog.querySelector('[class~="overflow-y-auto"]');
    expect(dialog.className).toContain("max-h-[calc(100vh-32px)]");
    expect(scrollRegion?.className).toContain("min-h-0");
    expect(scrollRegion?.className).toContain("flex-1");
    expect(scrollRegion?.textContent).toContain("Long history");
    expect(dialog.querySelector('button[aria-label="Close"]')).not.toBeNull();
    const footerClose = Array.from(dialog.querySelectorAll("button")).find(
      (button) => button.textContent === "Close",
    );
    expect(footerClose).toBeTruthy();
  });
});
