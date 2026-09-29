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
import type { BackendEvent, LibraryDocument } from "@/lib/backend";

let act: typeof import("@testing-library/react").act;
let cleanup: typeof import("@testing-library/react").cleanup;
let fireEvent: typeof import("@testing-library/react").fireEvent;
let render: typeof import("@testing-library/react").render;
let waitFor: typeof import("@testing-library/react").waitFor;
let DocumentPage: typeof import("@/features/library/document").DocumentPage;
let RouterProvider: typeof import("@/app/router").RouterProvider;
let backend: typeof import("@/lib/backend").backend;
let clearToasts: typeof import("@/components/ui/toast").clearToasts;
let readToasts: typeof import("@/components/ui/toast").readToasts;
let closeDom: () => void;
let event: (value: BackendEvent) => void;

const path = "notes.md";
const initial: LibraryDocument = {
  path,
  content: "server version one",
  hash: "hash-one",
};
const latest: LibraryDocument = {
  path,
  content: "server version two",
  hash: "hash-two",
};

beforeAll(async () => {
  const packageName = "jsdom";
  const { JSDOM } = await import(packageName);
  const dom = new JSDOM("<!doctype html><html><body></body></html>", {
    url: "http://localhost",
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
  vi.stubGlobal(
    "getComputedStyle",
    dom.window.getComputedStyle.bind(dom.window),
  );
  vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", true);
  Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
  vi.stubGlobal("requestAnimationFrame", vi.fn());
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe() {}
      disconnect() {}
    },
  );
  await import("react");
  ({ act, cleanup, fireEvent, render, waitFor } = await import(
    "@testing-library/react"
  ));
  ({ DocumentPage } = await import("@/features/library/document"));
  ({ RouterProvider } = await import("@/app/router"));
  ({ backend } = await import("@/lib/backend"));
  ({ clearToasts, readToasts } = await import("@/components/ui/toast"));
}, 30000);

beforeEach(() => {
  clearToasts();
  vi.spyOn(backend, "onEvent").mockImplementation((listener) => {
    event = listener;
    return vi.fn();
  });
  vi.spyOn(backend, "readLibrary").mockResolvedValue(initial);
});

afterEach(() => {
  cleanup();
  clearToasts();
  vi.restoreAllMocks();
});

afterAll(() => {
  closeDom();
  vi.unstubAllGlobals();
});

function mount() {
  return render(
    <RouterProvider>
      <DocumentPage path={path} />
    </RouterProvider>,
  );
}

function editor(container: HTMLElement): HTMLTextAreaElement {
  const field = container.querySelector<HTMLTextAreaElement>(
    'textarea[aria-label="Document"]',
  );
  if (!field) throw new Error("Document editor is missing");
  return field;
}

function conflict() {
  const item = readToasts().find(
    (value) => value.id === `document-conflict:${path}`,
  );
  if (!item?.action) throw new Error("Document conflict action is missing");
  return item;
}

function saveButton(container: HTMLElement): HTMLButtonElement {
  const button = Array.from(container.querySelectorAll("button")).find(
    (item) => item.textContent?.trim() === "Save",
  );
  if (!button) throw new Error("Document save button is missing");
  return button;
}

async function openInitialDocument(container: HTMLElement) {
  await waitFor(() => expect(editor(container).value).toBe(initial.content));
}

describe("Library document conflict recovery", () => {
  it("keeps edits until Reopen after a reconnect conflict", async () => {
    const view = mount();
    await openInitialDocument(view.container);
    fireEvent.change(editor(view.container), {
      target: { value: "local edit" },
    });

    vi.mocked(backend.readLibrary).mockResolvedValueOnce(latest);
    await act(async () => {
      event({ type: "connection.restored" });
      await waitFor(() => expect(backend.readLibrary).toHaveBeenCalledTimes(2));
    });

    expect(editor(view.container).value).toBe("local edit");
    const notification = conflict();
    expect(notification.description).toBe(
      "Your unsaved changes are preserved. Reopen replaces them with the current document.",
    );

    vi.mocked(backend.readLibrary).mockResolvedValueOnce(latest);
    await act(async () => {
      const readsBeforeReopen = vi.mocked(backend.readLibrary).mock.calls
        .length;
      notification.action?.onClick();
      await waitFor(() =>
        expect(
          vi.mocked(backend.readLibrary).mock.calls.length,
        ).toBeGreaterThan(readsBeforeReopen),
      );
    });
  });

  it("keeps edits until Reopen after a save conflict", async () => {
    const view = mount();
    await openInitialDocument(view.container);
    fireEvent.change(editor(view.container), {
      target: { value: "local edit" },
    });
    vi.spyOn(backend, "writeLibrary").mockResolvedValueOnce({
      conflict: true,
      path,
      current_hash: latest.hash,
      current_content: latest.content,
    });

    fireEvent.click(saveButton(view.container));
    await waitFor(() => expect(backend.writeLibrary).toHaveBeenCalledOnce());

    expect(editor(view.container).value).toBe("local edit");
    const notification = conflict();
    expect(notification.description).toBe(
      "Your unsaved changes are preserved. Reopen replaces them with the current document.",
    );

    vi.mocked(backend.readLibrary).mockResolvedValueOnce(latest);
    await act(async () => {
      const readsBeforeReopen = vi.mocked(backend.readLibrary).mock.calls
        .length;
      notification.action?.onClick();
      await waitFor(() =>
        expect(
          vi.mocked(backend.readLibrary).mock.calls.length,
        ).toBeGreaterThan(readsBeforeReopen),
      );
    });
  });
});
