import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LangfusePanel, langfuseUpdate } from "@/features/settings/langfuse";

const stored = {
  enabled: true,
  base_url: " https://example.invalid/ ",
  keys_set: true,
  environment: "development",
};

describe("Langfuse settings", () => {
  it("omits blank credentials and response-only fields", () => {
    expect(langfuseUpdate(stored, "  ", "")).toEqual({
      enabled: true,
      base_url: "https://example.invalid/",
    });
  });

  it("only replaces credentials that were entered", () => {
    expect(langfuseUpdate(stored, " test-public ", " test-secret ")).toEqual({
      enabled: true,
      base_url: "https://example.invalid/",
      public_key: "test-public",
      secret_key: "test-secret",
    });
    expect(langfuseUpdate(stored, "", "replacement")).not.toHaveProperty(
      "public_key",
    );
  });

  it("renders labelled password fields and disables saving before load", () => {
    const html = renderToStaticMarkup(<LangfusePanel />);
    expect(html).toContain("Public key");
    expect(html).toContain("Secret key");
    expect(html.match(/type="password"/g)).toHaveLength(2);
    expect(html).toMatch(/<fieldset\b[^>]*\bdisabled=""/);
    expect(html).toContain('type="submit"');
    const enabledId = html.match(/<input[^>]*id="([^"]+-enabled)"/)?.[1];
    expect(enabledId).toBeDefined();
    expect(html).toContain(`for="${enabledId}"`);
    expect(html).not.toContain('role="alert"');
    expect(html).not.toContain("Restart Fora");
  });
});
