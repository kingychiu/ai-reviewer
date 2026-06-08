import { info, warning } from "@actions/core";
import config, { type AgentSpec } from "../config";
import { runAgent, runStructured } from "../ai";
import { generateFileCodeDiff, FileDiff } from "../diff";
import { PullRequestReview } from "../prompts";
import { createReviewTools } from "./tools";
import { buildReviewResultSchema } from "./schema";

export type AgenticReviewInput = {
  prTitle: string;
  prDescription: string;
  prSummary: string;
  files: FileDiff[];
  repoRoot: string;
};

type AgentNotes = {
  agentId: string;
  notes: string;
};

const EMPTY_REVIEW: PullRequestReview = {
  review: {
    estimated_effort_to_review: 1,
    score: 0,
    has_relevant_tests: false,
    security_concerns: "No",
  },
  comments: [],
};

/**
 * The base agent built from the top-level `LLM_MODEL` (+ provider/baseUrl/
 * apiKey). Every role falls back to this when not explicitly configured — roles
 * never borrow each other's agents.
 */
function baseAgent(): AgentSpec {
  return {
    id: config.llmModel ?? "default",
    model: config.llmModel ?? "",
    provider: config.llmProvider,
    baseUrl: config.llmBaseUrl,
    apiKey: config.llmApiKey,
  };
}

/**
 * The explorer panel. When `AGENTS` is empty the single `llmModel` is used, so
 * existing single-model configs keep working unchanged.
 */
export function resolveExploreAgents(): AgentSpec[] {
  return config.agents.length > 0 ? config.agents : [baseAgent()];
}

/** The synthesis (judge) agent — explicit `SYNTHESIS_AGENT`, else `LLM_MODEL`. */
export function resolveSynthesisAgent(): AgentSpec {
  return config.synthesisAgent ?? baseAgent();
}

function buildDiffBlock(files: FileDiff[]): string {
  return files.map((file) => generateFileCodeDiff(file)).join("\n\n");
}

function agentFocus(agent: AgentSpec): string {
  return agent.instructions && agent.instructions.trim().length > 0
    ? `\n<Your Focus>\n${agent.instructions.trim()}\n</Your Focus>\n`
    : "";
}

function baseReviewSystemPrompt(agent: AgentSpec): string {
  return `You are an experienced senior software engineer reviewing a Git Pull Request.

You have tools to investigate the repository before commenting:
- list_guidelines: read project conventions (CLAUDE.md, AGENTS.md, .claude/rules, .claude/skills). ALWAYS call this first.
- read_file: read any file to understand context around the diff (definitions, callers, related code).
- grep: search the repo for usages, related patterns, or similar code.

Process:
1. Read the project guidelines and treat violations as important findings.
2. For each changed file/hunk, fetch the surrounding context you need (don't assume code you haven't read).
3. Look for bugs, security issues, regressions, broken contracts with callers, and guideline violations introduced by the new code (lines starting with '+').

Do NOT comment on formatting, code style, or adding code comments. Focus only on actionable issues in the new code.

When done investigating, output your findings as concise markdown notes. For each finding include: the file path, the affected line number(s) from the new code, a severity (critical/minor), a short label (e.g. bug, security, performance), and a clear explanation. If you found no actionable issues, say so explicitly.${agentFocus(agent)}`;
}

function buildReviewUserPrompt(pr: AgenticReviewInput): string {
  const guidelines =
    config.styleGuideRules && config.styleGuideRules.length > 0
      ? `\n<Additional Style Guide Rules>\n${config.styleGuideRules}\n</Additional Style Guide Rules>\n`
      : "";

  return `<PR title>
${pr.prTitle}
</PR title>

<PR Description>
${pr.prDescription}
</PR Description>

<PR Summary>
${pr.prSummary}
</PR Summary>
${guidelines}
<PR File Diffs>
${buildDiffBlock(pr.files)}
</PR File Diffs>

Investigate using your tools, then report your review findings as notes.`;
}

/** Run one explorer agent through the agentic loop, returning its raw notes. */
async function exploreWithAgent(
  agent: AgentSpec,
  pr: AgenticReviewInput
): Promise<AgentNotes> {
  const result = await runAgent({
    systemPrompt: baseReviewSystemPrompt(agent),
    prompt: buildReviewUserPrompt(pr),
    tools: createReviewTools(pr.repoRoot),
    maxSteps: config.agenticMaxSteps,
    model: agent,
  });
  info(`agentic exploration complete for '${agent.id}' (${result.steps} steps)`);
  return { agentId: agent.id, notes: result.text };
}

/**
 * One peer-debate round: every agent sees the union of the OTHER agents'
 * current findings and revises its own position — agreeing, refuting (with
 * reasons, verified via tools), or adding missed issues — while keeping its own
 * model and persona. Agents run in parallel against a fixed snapshot of the
 * previous round's notes. An agent that errors keeps its prior position.
 */
async function debateRound(
  debaters: AgentSpec[],
  pr: AgenticReviewInput,
  currentNotes: AgentNotes[],
  roundNum: number
): Promise<AgentNotes[]> {
  const byId = new Map(currentNotes.map((n) => [n.agentId, n]));

  const results = await Promise.all(
    debaters.map(async (agent): Promise<AgentNotes | null> => {
      const own = byId.get(agent.id) ?? null;
      const others = currentNotes.filter((n) => n.agentId !== agent.id);
      if (others.length === 0) return own; // nothing to debate against

      const othersBlock = others
        .map((o) => `===== Findings from '${o.agentId}' =====\n${o.notes}`)
        .join("\n\n");

      const systemPrompt = `${baseReviewSystemPrompt(agent)}

You are now in a DEBATE with the other reviewers. Critically evaluate their findings — do NOT rubber-stamp them. For each of their findings: AGREE only if you can confirm it (verify with read_file/grep), REFUTE it with a concrete reason if you believe it is wrong or a false positive, and ADD any issues everyone missed. Re-state your own findings, dropping any of yours that were correctly refuted. Output your updated findings in the same notes format.`;

      const userPrompt = `<Your Previous Findings>
${own ? own.notes : "(you produced no findings yet)"}
</Your Previous Findings>

<Other Reviewers' Findings>
${othersBlock}
</Other Reviewers' Findings>

${buildReviewUserPrompt(pr)}`;

      try {
        const result = await runAgent({
          systemPrompt,
          prompt: userPrompt,
          tools: createReviewTools(pr.repoRoot),
          maxSteps: config.agenticMaxSteps,
          model: agent,
        });
        info(
          `debate round ${roundNum}: '${agent.id}' updated (${result.steps} steps)`
        );
        return { agentId: agent.id, notes: result.text };
      } catch (e) {
        warning(`debate round ${roundNum} failed for '${agent.id}': ${e}`);
        return own; // keep prior position on failure
      }
    })
  );

  return results.filter((n): n is AgentNotes => n !== null);
}

/**
 * Synthesize agents' notes into the strict PullRequestReview shape using a
 * structured (generateObject) call with the synthesis agent's model.
 */
async function synthesizeReview(
  agent: AgentSpec,
  pr: AgenticReviewInput,
  allNotes: AgentNotes[]
): Promise<PullRequestReview> {
  const notesBlock = allNotes
    .map((n) => `===== Reviewer: ${n.agentId} =====\n${n.notes}`)
    .join("\n\n");

  const systemPrompt = `You are a senior engineer consolidating one or more reviewers' findings into a final structured PR review.

The PR diff uses '__new hunk__' / '__old hunk__' sections. The '__new hunk__' lines are prefixed with line numbers; use those exact numbers for 'start_line' and 'end_line'. Only comment on new code (lines starting with '+').

Rules:
- Merge duplicate or overlapping findings into a single comment. Do not emit near-duplicate comments.
- Weight by cross-agent agreement: findings multiple reviewers independently raised or confirmed are high-confidence; keep them. Drop findings that another reviewer refuted and no one defended, and anything not actionable.
- 'critical' must be true only for findings that should block merge (bugs, security, regressions).
- Map each comment to the precise file and '__new hunk__' line numbers. If a finding cannot be mapped to changed lines, omit it.
- If there are no actionable findings, return an empty comments array.${agentFocus(agent)}`;

  const userPrompt = `<PR title>
${pr.prTitle}
</PR title>

<PR Summary>
${pr.prSummary}
</PR Summary>

<Reviewer Findings>
${notesBlock}
</Reviewer Findings>

<PR File Diffs>
${buildDiffBlock(pr.files)}
</PR File Diffs>

Produce the final consolidated review.`;

  return (await runStructured({
    prompt: userPrompt,
    systemPrompt,
    schema: buildReviewResultSchema(),
    model: agent,
  })) as PullRequestReview;
}

/**
 * Opt-in agentic review entrypoint. Returns the same PullRequestReview shape as
 * runReviewPrompt so it is a drop-in replacement in pull_request.ts.
 *
 * Pipeline: explore (AGENTS panel, parallel) -> [debate (DEBATE_AGENT)] ->
 * synthesize (SYNTHESIS_AGENT).
 */
export async function runAgenticReview(
  pr: AgenticReviewInput
): Promise<PullRequestReview> {
  const explorers = resolveExploreAgents();
  if (explorers.length === 0 || !explorers[0].model) {
    warning("agentic review: no agent/model configured; skipping");
    return EMPTY_REVIEW;
  }

  info(
    `agentic review: ${explorers.length} agent(s) [${explorers
      .map((a) => a.id)
      .join(", ")}], strategy=${config.reviewStrategy}`
  );

  // Phase 1: explorers investigate the repo and produce notes (in parallel).
  const explorations = await Promise.all(
    explorers.map((agent) =>
      exploreWithAgent(agent, pr).catch((e) => {
        warning(`agentic exploration failed for '${agent.id}': ${e}`);
        return null;
      })
    )
  );
  let notes = explorations.filter((n): n is AgentNotes => n !== null);

  if (notes.length === 0) {
    warning("agentic review: all explorations failed; returning empty review");
    return EMPTY_REVIEW;
  }

  // Phase 2 (optional): peer debate — the panel critiques each other across
  // rounds. Needs at least two surviving agents to debate.
  if (config.reviewStrategy === "debate" && notes.length > 1) {
    const debaters = explorers.filter((a) =>
      notes.some((n) => n.agentId === a.id)
    );
    for (let r = 1; r <= config.agenticDebateRounds; r++) {
      info(
        `agentic review: debate round ${r}/${config.agenticDebateRounds}`
      );
      notes = await debateRound(debaters, pr, notes, r);
      if (notes.length <= 1) break;
    }
  }

  // Phase 3: synthesize into the structured review result.
  return await synthesizeReview(resolveSynthesisAgent(), pr, notes);
}
