import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  ExecutionForm,
  type ExecutionSettings,
  executionUpdate,
} from "@/features/settings/execution";

const initial: ExecutionSettings = {
  write_directories: ["/work"],
  unusable_write_directories: [],
  error: null,
};

describe("Execution settings", () => {
  it("saves the directories without App startup fields", () => {
    expect(
      executionUpdate([" /work ", "/work", "/home/you/中文", " "]),
    ).toEqual({
      write_directories: ["/work", "/home/you/中文"],
    });
    expect(executionUpdate([])).toEqual({ write_directories: [] });
  });

  it("edits the file policy without an environment choice", () => {
    const html = renderToStaticMarkup(
      <ExecutionForm initial={initial} onSave={async () => initial} />,
    );
    expect(html).toContain("Writable directories");
    expect(html).toContain("/work</span>");
    expect(html).toContain("Edit /work");
    expect(html).toContain("Delete /work");
    expect(html).toContain("Add directory");
    expect(html).not.toContain("Execution environment");
    expect(html).not.toContain("Next start");
  });

  it("shows an unavailable execution with its diagnostics", () => {
    const failed: ExecutionSettings = {
      ...initial,
      write_directories: ["/missing"],
      error: "Execution environment is unavailable",
      unusable_write_directories: [
        { path: "/missing", reason: "invalid_directory" },
      ],
    };
    const html = renderToStaticMarkup(
      <ExecutionForm initial={failed} onSave={async () => failed} />,
    );
    expect(html).toContain("/missing</span>");
    expect(html).toContain("Execution environment is unavailable");
    expect(html).toContain("/missing");
    expect(html).toContain("invalid_directory");
    expect(html).not.toContain('role="alert"');
    expect(html).not.toContain('role="status"');
    expect(html).toContain(">Unavailable</span>");
    expect(html).toContain(">invalid_directory</span>");
  });

  it("shows the complete path with wrapping", () => {
    const html = renderToStaticMarkup(
      <ExecutionForm initial={initial} onSave={async () => initial} />,
    );
    expect(html).toContain(
      "whitespace-pre-wrap font-mono text-sm wrap-anywhere",
    );
    expect(html).toContain("border-line bg-surface");
    expect(html).not.toContain("Restart Fora");
  });
});
