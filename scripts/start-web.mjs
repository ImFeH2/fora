import { readdirSync } from "node:fs";
import { resolve } from "node:path";
import { root, run } from "./process.mjs";

const packages = resolve(root, "dist", "python");
const environment = resolve(root, "dist", "web-environment");
const scripts = process.platform === "win32" ? "Scripts" : "bin";
const extension = process.platform === "win32" ? ".exe" : "";

run("pnpm", ["build:python"]);

const wheels = readdirSync(packages)
  .filter((name) => name.endsWith(".whl"))
  .map((name) => resolve(packages, name));

run("uv", ["venv", "--allow-existing", "--python", "3.13", environment]);
run("uv", [
  "pip",
  "install",
  "--python",
  resolve(environment, scripts, `python${extension}`),
  "--reinstall",
  ...wheels,
]);
run(
  resolve(environment, scripts, `fora-web${extension}`),
  process.argv.slice(2),
);
