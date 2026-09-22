import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { autoGrowHeight } from "@/components/ui/index";
import {
  Composer,
  type ComposerKey,
  composerFades,
  composerHeight,
  composerKey,
  composerMultiline,
} from "@/features/discussions/composer";

function press(key: string, modifiers: Partial<ComposerKey> = {}): ComposerKey {
  return { key, shiftKey: false, ctrlKey: false, metaKey: false, ...modifiers };
}

describe("composerKey", () => {
  it("sends on Enter", () => {
    expect(composerKey(press("Enter"), false)).toBe("send");
  });

  it("inserts a newline on Shift+Enter", () => {
    expect(composerKey(press("Enter", { shiftKey: true }), false)).toBe(
      "newline",
    );
    expect(composerKey(press("Enter", { shiftKey: true }), true)).toBe(
      "newline",
    );
  });

  it("still sends on Ctrl+Enter and Cmd+Enter", () => {
    expect(composerKey(press("Enter", { ctrlKey: true }), false)).toBe("send");
    expect(composerKey(press("Enter", { metaKey: true }), false)).toBe("send");
  });

  it("accepts the suggestion on Enter or Tab while the mention menu is open", () => {
    expect(composerKey(press("Enter"), true)).toBe("accept");
    expect(composerKey(press("Tab"), true)).toBe("accept");
  });

  it("sends on Ctrl+Enter even while the mention menu is open", () => {
    expect(composerKey(press("Enter", { ctrlKey: true }), true)).toBe("send");
    expect(composerKey(press("Enter", { metaKey: true }), true)).toBe("send");
  });

  it("dismisses the mention menu on Escape", () => {
    expect(composerKey(press("Escape"), true)).toBe("dismiss");
  });

  it("moves through the suggestions with the arrow keys", () => {
    expect(composerKey(press("ArrowDown"), true)).toBe("down");
    expect(composerKey(press("ArrowUp"), true)).toBe("up");
  });

  it("leaves navigation keys alone when no menu is open", () => {
    for (const key of ["ArrowDown", "ArrowUp", "Escape", "Tab"]) {
      expect(composerKey(press(key), false)).toBeNull();
    }
  });

  it("leaves an Enter that commits an IME composition alone", () => {
    expect(
      composerKey(press("Enter", { isComposing: true }), false),
    ).toBeNull();
    expect(composerKey(press("Enter", { isComposing: true }), true)).toBeNull();
  });

  it("ignores ordinary typing", () => {
    expect(composerKey(press("a"), false)).toBeNull();
    expect(composerKey(press("a"), true)).toBeNull();
  });
});

describe("composer layout", () => {
  it("uses measured wrapping rather than character count", () => {
    expect(composerMultiline("x".repeat(500), 46, 22, 24)).toBe(false);
    expect(composerMultiline("中文😀", 68, 22, 24)).toBe(true);
    expect(composerMultiline("", 68, 22, 24)).toBe(false);
    expect(composerMultiline("text", 47, 22, 24)).toBe(false);
  });

  it("expands explicit newlines and contracts when measured as one line", () => {
    expect(composerMultiline("\n", 46, 22, 24)).toBe(true);
    expect(composerMultiline("first\nsecond", 68, 22, 24)).toBe(true);
    expect(composerMultiline("first", 46, 22, 24)).toBe(false);
  });

  it("keeps reference silhouettes and grows with the capped input", () => {
    expect(composerHeight(false, 46)).toBe(48);
    expect(composerHeight(true, 68)).toBe(116);
    expect(composerHeight(true, 200)).toBe(248);
    expect(autoGrowHeight(248, 22, 28, 0, 8)).toBe(204);
    expect(composerHeight(true, 204)).toBe(252);
  });

  it("renders a real compact input, an inaccessible measuring probe and disabled send", () => {
    const html = renderToStaticMarkup(
      createElement(Composer, {
        discussionId: 1,
        members: [],
        memberIds: new Set<number>(),
        busy: false,
        placeholder: "Message",
        onSend: async () => true,
        onHeightChange: () => {},
      }),
    );
    expect(html).toContain('data-expanded="false"');
    expect(html).toContain('role="combobox"');
    expect(html).toContain('aria-label="Message"');
    expect(html).toContain('aria-label="Voice input · Coming soon"');
    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain('tabindex="-1"');
    expect(html).toContain("motion-reduce:transition-none");
    expect(html).not.toContain("Open prompt input");
    expect(html).toContain('type="file"');
    expect(html).toContain('aria-label="Choose attachments"');
    expect(html).toContain("@[601px]:w-3/4");
    expect(html).not.toContain("max-width");
    expect(html).toContain("--composer-card:oklch(20.5% 0 0)");
    expect(html).toContain("--composer-primary:oklch(92.2% 0 0)");
    expect(html).toContain("M7 12V2M7 2L2.5 6.5M7 2L11.5 6.5");
    expect(html).toContain("M7 2.5V11.5M2.5 7H11.5");
    expect(html).toContain("opacity-0 scale-50 rotate-45 blur-[1px]");
    expect(html).toContain("opacity-0 blur-sm translate-y-2");
    expect(html).toContain("from-(--composer-card)");
    expect(html).not.toContain("lucide");
    expect(html).toContain(
      "shadow-[0_1px_3px_0_rgb(0_0_0/0.1),0_1px_2px_-1px_rgb(0_0_0/0.1)]",
    );
  });
});

describe("composer scroll fades", () => {
  it("fades only where more text remains", () => {
    expect(composerFades(0, 46, 46)).toEqual({ top: 0, bottom: 0 });
    expect(composerFades(0, 400, 200)).toEqual({ top: 0, bottom: 1 });
    expect(composerFades(200, 400, 200)).toEqual({ top: 1, bottom: 0 });
    expect(composerFades(10, 226, 200)).toEqual({ top: 0.5, bottom: 0 });
  });
});
