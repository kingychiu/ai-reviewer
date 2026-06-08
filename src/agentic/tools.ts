import { tool, type CoreTool } from "ai";
import { z } from "zod";
import * as fs from "fs";
import * as path from "path";

// Directories that never contain useful review context and would slow the walk.
// (Build output, caches, deps, VCS — these crowd out real source on large repos
// and would otherwise consume the walk budget before reaching the code.)
const IGNORED_DIRS = new Set([
  ".git",
  ".svn",
  ".hg",
  "node_modules",
  "dist",
  "build",
  "out",
  ".output",
  ".next",
  ".nuxt",
  ".svelte-kit",
  ".astro",
  "coverage",
  "htmlcov",
  "vendor",
  "target",
  "bin",
  "obj",
  ".gradle",
  ".idea",
  ".venv",
  "venv",
  "__pycache__",
  ".pytest_cache",
  ".mypy_cache",
  ".ruff_cache",
  ".tox",
  ".turbo",
  ".cache",
]);

// Well-known agent/instruction files to surface as review guidelines.
const ROOT_GUIDELINE_FILES = [
  "CLAUDE.md",
  "AGENTS.md",
  "AGENT.md",
  "GEMINI.md",
  ".cursorrules",
  ".windsurfrules",
  path.join(".github", "copilot-instructions.md"),
];

// Names that count as nested guideline files anywhere in the tree.
const NESTED_GUIDELINE_NAMES = new Set([
  "CLAUDE.md",
  "AGENTS.md",
  "AGENT.md",
  "GEMINI.md",
]);

const DEFAULT_MAX_FILE_BYTES = 64 * 1024; // 64KB per file read
const DEFAULT_MAX_GREP_RESULTS = 50;
const MAX_WALK_FILES = 50000; // safety cap on tree traversal (large monorepos)
const MAX_GUIDELINE_FILES = 50;
const MAX_GUIDELINE_BYTES = 16 * 1024; // per guideline file

export type RepoFile = {
  path: string;
  content: string;
  truncated: boolean;
  totalLines: number;
};

export type GrepMatch = {
  file: string;
  line: number;
  text: string;
};

export type Guideline = {
  path: string;
  content: string;
  truncated: boolean;
};

/** Resolve `relPath` strictly inside `root`, or return null if it escapes. */
function resolveInsideRoot(root: string, relPath: string): string | null {
  const normalizedRoot = path.resolve(root);
  const candidate = path.resolve(normalizedRoot, relPath);
  if (
    candidate !== normalizedRoot &&
    !candidate.startsWith(normalizedRoot + path.sep)
  ) {
    return null;
  }
  return candidate;
}

function looksBinary(buffer: Buffer): boolean {
  // Heuristic: a NUL byte in the first chunk means binary.
  const len = Math.min(buffer.length, 8000);
  for (let i = 0; i < len; i++) {
    if (buffer[i] === 0) return true;
  }
  return false;
}

/**
 * Read a file from the checked-out repo, bounded by size and optional line
 * range. Path traversal outside `root` is rejected.
 */
export function readRepoFile(
  root: string,
  relPath: string,
  opts: { maxBytes?: number; startLine?: number; endLine?: number } = {}
): RepoFile {
  const resolved = resolveInsideRoot(root, relPath);
  if (!resolved) {
    throw new Error(`Path escapes repository root: ${relPath}`);
  }
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    throw new Error(`File not found: ${relPath}`);
  }

  const maxBytes = opts.maxBytes ?? DEFAULT_MAX_FILE_BYTES;
  const raw = fs.readFileSync(resolved);
  if (looksBinary(raw)) {
    throw new Error(`Refusing to read binary file: ${relPath}`);
  }

  const fullText = raw.toString("utf8");
  const allLines = fullText.split("\n");
  const totalLines = allLines.length;

  let lines = allLines;
  if (opts.startLine !== undefined || opts.endLine !== undefined) {
    const start = Math.max(1, opts.startLine ?? 1);
    const end = Math.min(totalLines, opts.endLine ?? totalLines);
    lines = allLines.slice(start - 1, end);
  }

  let content = lines.join("\n");
  let truncated = false;
  if (Buffer.byteLength(content, "utf8") > maxBytes) {
    content = content.slice(0, maxBytes);
    truncated = true;
  }

  return { path: relPath, content, truncated, totalLines };
}

/** Recursively collect file paths (relative to root), bounded by MAX_WALK_FILES. */
function walkFiles(root: string): string[] {
  const results: string[] = [];
  const stack: string[] = [root];

  while (stack.length > 0 && results.length < MAX_WALK_FILES) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const abs = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (IGNORED_DIRS.has(entry.name)) continue;
        stack.push(abs);
      } else if (entry.isFile()) {
        results.push(path.relative(root, abs));
        if (results.length >= MAX_WALK_FILES) break;
      }
    }
  }

  return results;
}

/**
 * Search the repo for `pattern` (substring, or regex when `regex` is true).
 * Skips ignored dirs and binary/oversized files. Bounded by `maxResults`.
 */
export function grepRepo(
  root: string,
  pattern: string,
  opts: { maxResults?: number; regex?: boolean } = {}
): GrepMatch[] {
  if (!pattern) return [];
  const maxResults = opts.maxResults ?? DEFAULT_MAX_GREP_RESULTS;

  let matcher: (line: string) => boolean;
  if (opts.regex) {
    let re: RegExp;
    try {
      re = new RegExp(pattern);
    } catch (e) {
      throw new Error(`Invalid regex: ${pattern}`);
    }
    matcher = (line) => re.test(line);
  } else {
    const needle = pattern;
    matcher = (line) => line.includes(needle);
  }

  const matches: GrepMatch[] = [];
  for (const rel of walkFiles(root)) {
    if (matches.length >= maxResults) break;
    const abs = path.join(root, rel);
    let raw: Buffer;
    try {
      const stat = fs.statSync(abs);
      if (stat.size > 1024 * 1024) continue; // skip files > 1MB
      raw = fs.readFileSync(abs);
    } catch {
      continue;
    }
    if (looksBinary(raw)) continue;

    const lines = raw.toString("utf8").split("\n");
    for (let i = 0; i < lines.length; i++) {
      if (matcher(lines[i])) {
        matches.push({ file: rel, line: i + 1, text: lines[i].trim().slice(0, 300) });
        if (matches.length >= maxResults) break;
      }
    }
  }

  return matches;
}

/** Recursively list markdown files (relative to root) under root/subDir. */
function listMarkdownUnder(root: string, subDir: string): string[] {
  const base = path.join(root, subDir);
  if (!fs.existsSync(base) || !fs.statSync(base).isDirectory()) return [];

  const results: string[] = [];
  const stack: string[] = [base];
  while (stack.length > 0 && results.length < MAX_GUIDELINE_FILES) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const abs = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!IGNORED_DIRS.has(entry.name)) stack.push(abs);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
        results.push(path.relative(root, abs));
      }
    }
  }
  return results;
}

/**
 * Discover repo guideline/instruction files, in priority order so the most
 * important rules are never crowded out on large repos:
 *   1. well-known root files (CLAUDE.md, AGENTS.md, .cursorrules, …)
 *   2. `.claude/rules/**.md` (project review rules — highest signal)
 *   3. nested AGENTS.md / CLAUDE.md across the tree
 *   4. `.claude/skills/**` SKILL.md (one per skill, to bound volume)
 */
export function findRepoGuidelines(root: string): Guideline[] {
  const found = new Map<string, Guideline>();

  const addFile = (rel: string) => {
    if (found.size >= MAX_GUIDELINE_FILES) return;
    if (found.has(rel)) return;
    const abs = path.join(root, rel);
    try {
      if (!fs.statSync(abs).isFile()) return;
      const raw = fs.readFileSync(abs);
      if (looksBinary(raw)) return;
      let content = raw.toString("utf8");
      let truncated = false;
      if (Buffer.byteLength(content, "utf8") > MAX_GUIDELINE_BYTES) {
        content = content.slice(0, MAX_GUIDELINE_BYTES);
        truncated = true;
      }
      found.set(rel, { path: rel, content, truncated });
    } catch {
      /* ignore unreadable files */
    }
  };

  // 1. Well-known root files.
  for (const name of ROOT_GUIDELINE_FILES) {
    if (fs.existsSync(path.join(root, name))) addFile(name);
  }

  // 2. Project rules (highest priority).
  for (const rel of listMarkdownUnder(root, ".claude/rules")) addFile(rel);

  // 3. Nested AGENTS.md / CLAUDE.md across the tree.
  for (const rel of walkFiles(root)) {
    if (found.size >= MAX_GUIDELINE_FILES) break;
    if (NESTED_GUIDELINE_NAMES.has(path.basename(rel))) addFile(rel);
  }

  // 4. Skill descriptions (only the top-level SKILL.md per skill).
  for (const rel of listMarkdownUnder(root, ".claude/skills")) {
    if (path.basename(rel).toLowerCase() === "skill.md") addFile(rel);
  }

  return Array.from(found.values());
}

/**
 * Build the AI SDK tool set the agentic reviewer can call. Tools are bound to a
 * single repo `root` and return plain strings (what the model consumes).
 */
export function createReviewTools(root: string): Record<string, CoreTool> {
  return {
    read_file: tool({
      description:
        "Read a file from the repository to understand context around the diff " +
        "(definitions, callers, related code). Optionally restrict to a line range.",
      parameters: z.object({
        path: z
          .string()
          .describe("Repository-relative path, e.g. 'src/utils/auth.ts'"),
        start_line: z
          .number()
          .optional()
          .describe("1-based first line to read (inclusive)"),
        end_line: z
          .number()
          .optional()
          .describe("1-based last line to read (inclusive)"),
      }),
      execute: async ({ path: relPath, start_line, end_line }) => {
        try {
          const file = readRepoFile(root, relPath, {
            startLine: start_line,
            endLine: end_line,
          });
          const header = `File: ${file.path} (${file.totalLines} lines${
            file.truncated ? ", truncated" : ""
          })`;
          return `${header}\n\n${file.content}`;
        } catch (e) {
          return `Error: ${e instanceof Error ? e.message : String(e)}`;
        }
      },
    }),

    list_guidelines: tool({
      description:
        "List and read the repository's review guidelines and agent instruction " +
        "files (CLAUDE.md, AGENTS.md, .claude/rules, .claude/skills, etc.). " +
        "Call this first to learn project-specific conventions to enforce.",
      parameters: z.object({}),
      execute: async () => {
        const guidelines = findRepoGuidelines(root);
        if (guidelines.length === 0) {
          return "No guideline files found in this repository.";
        }
        return guidelines
          .map(
            (g) =>
              `===== ${g.path}${g.truncated ? " (truncated)" : ""} =====\n${g.content}`
          )
          .join("\n\n");
      },
    }),

    grep: tool({
      description:
        "Search the repository for a string or regex to find related code, " +
        "usages, definitions, or similar patterns outside the diff.",
      parameters: z.object({
        pattern: z.string().describe("The string or regex to search for"),
        regex: z
          .boolean()
          .optional()
          .describe("Treat pattern as a regular expression (default false)"),
        max_results: z
          .number()
          .optional()
          .describe(`Max matches to return (default ${DEFAULT_MAX_GREP_RESULTS})`),
      }),
      execute: async ({ pattern, regex, max_results }) => {
        try {
          const matches = grepRepo(root, pattern, {
            regex,
            maxResults: max_results,
          });
          if (matches.length === 0) {
            return `No matches found for: ${pattern}`;
          }
          return matches.map((m) => `${m.file}:${m.line}: ${m.text}`).join("\n");
        } catch (e) {
          return `Error: ${e instanceof Error ? e.message : String(e)}`;
        }
      },
    }),
  };
}
