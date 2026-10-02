import { cpSync, rmSync } from "node:fs";
import { resolve } from "node:path";
import { root, run } from "./process.mjs";

const core = resolve(root, "core");
const bundled = resolve(root, "app", "core");
const dist = resolve(core, "dist");
const work = resolve(core, "build");

const extension = process.platform === "win32" ? ".exe" : "";

run(
  "uv",
  [
    "run",
    "--project",
    core,
    "python",
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--distpath",
    dist,
    "--workpath",
    work,
    resolve(core, "fora.spec"),
  ],
  { cwd: core },
);

rmSync(bundled, { force: true, recursive: true });
cpSync(resolve(dist, "fora"), bundled, { recursive: true });

const executable = resolve(bundled, `fora${extension}`);
process.stdout.write(`${executable}\n`);
