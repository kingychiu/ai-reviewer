import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import {
  readRepoFile,
  grepRepo,
  findRepoGuidelines,
  createReviewTools,
} from "../agentic/tools";

let root: string;

function write(rel: string, content: string) {
  const abs = path.join(root, rel);
  fs.mkdirSync(path.dirname(abs), { recursive: true });
  fs.writeFileSync(abs, content);
}

beforeAll(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "airev-tools-"));

  write("CLAUDE.md", "# Claude rules\nUse tabs.");
  write("AGENTS.md", "# Agents\nFollow conventions.");
  write("backend/AGENTS.md", "# Backend agents\nDjango Ninja only.");
  write(".claude/rules/python.md", "# Python\nUse type hints.");
  write(".claude/skills/foo/SKILL.md", "# Foo skill\nDoes foo.");
  write("node_modules/pkg/AGENTS.md", "should be ignored");
  write(
    "src/index.ts",
    "export function doThing() {\n  return 42;\n}\n\nconst x = doThing();\n"
  );
  write("src/big.ts", "x\n".repeat(100000)); // ~200KB
  // binary file with a NUL byte
  fs.writeFileSync(path.join(root, "bin.dat"), Buffer.from([1, 2, 0, 3, 4]));
});

afterAll(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

describe("readRepoFile", () => {
  test("reads a file inside the repo", () => {
    const file = readRepoFile(root, "src/index.ts");
    expect(file.content).toContain("doThing");
    expect(file.totalLines).toBeGreaterThan(1);
    expect(file.truncated).toBe(false);
  });

  test("supports line ranges", () => {
    const file = readRepoFile(root, "src/index.ts", {
      startLine: 1,
      endLine: 1,
    });
    expect(file.content).toBe("export function doThing() {");
  });

  test("rejects path traversal outside the root", () => {
    expect(() => readRepoFile(root, "../etc/passwd")).toThrow(/escapes/);
    expect(() => readRepoFile(root, "/etc/passwd")).toThrow(/escapes/);
  });

  test("throws on missing file", () => {
    expect(() => readRepoFile(root, "does/not/exist.ts")).toThrow(/not found/);
  });

  test("refuses binary files", () => {
    expect(() => readRepoFile(root, "bin.dat")).toThrow(/binary/);
  });

  test("truncates oversized reads", () => {
    const file = readRepoFile(root, "src/big.ts", { maxBytes: 1000 });
    expect(file.truncated).toBe(true);
    expect(Buffer.byteLength(file.content, "utf8")).toBeLessThanOrEqual(1000);
  });
});

describe("grepRepo", () => {
  test("finds substring matches with file:line", () => {
    const matches = grepRepo(root, "doThing");
    expect(matches.length).toBeGreaterThanOrEqual(2);
    expect(matches.every((m) => m.file === "src/index.ts")).toBe(true);
    expect(matches[0]).toHaveProperty("line");
  });

  test("ignores node_modules", () => {
    const matches = grepRepo(root, "should be ignored");
    expect(matches).toHaveLength(0);
  });

  test("supports regex", () => {
    const matches = grepRepo(root, "do.*\\(\\)", { regex: true });
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  test("throws on invalid regex", () => {
    expect(() => grepRepo(root, "(", { regex: true })).toThrow(/Invalid regex/);
  });

  test("respects maxResults", () => {
    const matches = grepRepo(root, "doThing", { maxResults: 1 });
    expect(matches).toHaveLength(1);
  });

  test("returns empty for empty pattern", () => {
    expect(grepRepo(root, "")).toHaveLength(0);
  });
});

describe("findRepoGuidelines", () => {
  test("discovers root, nested, rules and skills guideline files", () => {
    const found = findRepoGuidelines(root);
    const paths = found.map((g) => g.path.split(path.sep).join("/"));

    expect(paths).toContain("CLAUDE.md");
    expect(paths).toContain("AGENTS.md");
    expect(paths).toContain("backend/AGENTS.md");
    expect(paths).toContain(".claude/rules/python.md");
    expect(paths).toContain(".claude/skills/foo/SKILL.md");
  });

  test("ignores guideline files inside node_modules", () => {
    const found = findRepoGuidelines(root);
    const paths = found.map((g) => g.path.split(path.sep).join("/"));
    expect(paths).not.toContain("node_modules/pkg/AGENTS.md");
  });

  test("includes file content", () => {
    const found = findRepoGuidelines(root);
    const claude = found.find((g) => g.path === "CLAUDE.md");
    expect(claude?.content).toContain("Use tabs");
  });
});

describe("createReviewTools", () => {
  const tools = () => createReviewTools(root);

  test("exposes read_file, list_guidelines and grep", () => {
    const t = tools();
    expect(Object.keys(t).sort()).toEqual(["grep", "list_guidelines", "read_file"]);
  });

  test("read_file tool returns formatted content", async () => {
    const result = await (tools().read_file as any).execute({
      path: "src/index.ts",
    });
    expect(result).toContain("File: src/index.ts");
    expect(result).toContain("doThing");
  });

  test("read_file tool returns an error string on bad path", async () => {
    const result = await (tools().read_file as any).execute({
      path: "../escape",
    });
    expect(result).toMatch(/^Error:/);
  });

  test("list_guidelines tool concatenates guideline files", async () => {
    const result = await (tools().list_guidelines as any).execute({});
    expect(result).toContain("CLAUDE.md");
    expect(result).toContain("Use tabs");
  });

  test("grep tool returns match lines", async () => {
    const result = await (tools().grep as any).execute({ pattern: "doThing" });
    expect(result).toContain("src/index.ts:");
  });

  test("grep tool reports no matches", async () => {
    const result = await (tools().grep as any).execute({
      pattern: "zzz_no_such_token_zzz",
    });
    expect(result).toMatch(/No matches/);
  });
});
