# Agentic Review

An opt-in, multi-agent, context-aware review mode for this action. It is
**additive and off by default** — when disabled, the action behaves exactly as
before (a single-shot review). When enabled, a panel of agents reads the repo
with tools (files, rules, grep) before commenting, optionally discusses each
other's findings, and a synthesis agent merges everything into the final inline
comments.

> Requires `LLM_PROVIDER=ai-sdk` (the default). For any other provider the
> action logs a warning and falls back to the standard single-shot review.

---

## Contents

- [Quick start](#quick-start)
- [Pipeline & modes](#pipeline--modes)
- [Configuration reference](#configuration-reference)
- [Agent spec format](#agent-spec-format)
- [What each agent can access](#what-each-agent-can-access)
- [Prompt composition & instruction usage](#prompt-composition--instruction-usage)
- [Cost & latency](#cost--latency)
- [Failure handling](#failure-handling)
- [Full examples](#full-examples)

---

## Quick start

Single agent, agentic (reads repo context before commenting):

```yaml
- uses: actions/checkout@v4        # required — agents read the checkout
- uses: presubmit/ai-reviewer@latest
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
    LLM_PROVIDER: ai-sdk
    LLM_BASE_URL: https://openrouter.ai/api/v1   # optional (OpenAI-compatible)
    LLM_MODEL: anthropic/claude-sonnet-4.5
    AGENTIC_REVIEW: "true"
```

A panel of two agents (2+ agents automatically discuss):

```yaml
    AGENTIC_REVIEW: "true"
    AGENTS: "anthropic/claude-sonnet-4.5, google/gemini-2.5-pro"
```

---

## Pipeline & modes

Read this first — the config below names the parts of this pipeline.

There is **no mode flag** — the mode is implied by how many agents you
configure: **1 agent = single**, **2+ agents = discussion**.

Every agentic run is **explore → [discuss] → synthesize**. During explore and
discussion, each agent runs a **tool loop**: call a tool, read the result,
decide the next action, repeat — until it has enough context to report findings.
Synthesis has no tools.

```
   tools available to explore/discussion agents:
   ┌────────────────┬───────────────────────────────────────────┐
   │ list_guidelines │ CLAUDE.md, AGENTS.md, .claude/rules, skills │
   │ read_file       │ surrounding code / definitions / callers    │
   │ grep            │ usages & related patterns across the repo   │
   └────────────────┴───────────────────────────────────────────┘
   every explore AND discussion call is capped by AGENTIC_MAX_STEPS
```

### Single agent (`AGENTS` empty or one entry)

One agent makes **one explore call**. Inside it, a tool loop runs up to
`AGENTIC_MAX_STEPS` steps; then synthesize (no tool loop).

```
        ┌─ EXPLORE  (1 call) ─────────────────────┐   ┌─ SYNTHESIZE ────────┐
        │  ┌───────────────────────────────────┐  │   │ SYNTHESIS_AGENT     │
 PR ───▶ │  │ tool loop: read_file/grep/        │  │──▶│ (no tools)          │──▶ comments
 diff    │  │ list_guidelines → reason → repeat │  │   │ notes → diff lines  │
        │  │  ⟲ up to AGENTIC_MAX_STEPS steps  │  │   └─────────────────────┘
        │  └───────────────────────────────────┘  │
        │            → findings notes              │
        └─────────────────────────────────────────┘
   AGENTIC_MAX_STEPS caps this single explore loop.
```

### Multiple agents (`AGENTS` has 2+) — panel critiques EACH OTHER, then merge

`AGENTIC_MAX_STEPS` caps the tool loop **inside every box below** — each of the
N explore calls AND each of the N×R discuss calls runs its own loop up to that
many steps.

```
   ┌─ EXPLORE (N calls, concurrent)┐  ┌─ DISCUSS · R rounds (N calls/round) ───────────────┐   ┌─ SYNTHESIZE ─────┐
A: │ "security"  ─notes_A─┐         │  │  A sees {B}  ─▶ agree / refute(verify) / add ─▶ A' │──▶│ SYNTHESIS_AGENT  │
   │ tool loop ≤MAX_STEPS │ + focus │  │            ╲ ╱  tool loop ≤MAX_STEPS · own model   │   │ weight by        │──▶ comments
 PR│                      │         │  │             ╳                                      │   │ cross-agent      │
   │                      │         │  │            ╱ ╲  tool loop ≤MAX_STEPS · own model   │   │ agreement →      │
B: │ "correctness"─notes_B┘         │  │  B sees {A}  ─▶ agree / refute(verify) / add ─▶ B' │──▶│ diff lines       │
   │ tool loop ≤MAX_STEPS │ + focus │  │   (each round runs on the prev round's snapshot)   │   └──────────────────┘
   └────────────────────────────────┘  │   fail → that agent keeps its prior notes          │
                                       └────────────────────────────────────────────────────┘
            ≤MAX_STEPS = AGENTIC_MAX_STEPS · R = AGENTIC_DISCUSSION_ROUNDS · discuss skipped if < 2 agents
```

One discussion round in close-up — the cross is the point (every agent reads
every *other* agent):

```
        round r input = all agents' notes from round r-1
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  agent A     agent B     agent C        ← run concurrently
  reads B,C   reads A,C   reads A,B      ← each sees the OTHERS
     │           │           │
     ▼           ▼           ▼
    A'          B'          C'           ← revised positions → round r+1 (or synthesize)
```

### Mode comparison

| | single agent | multiple agents (discussion) |
|---|---|---|
| selected by | `AGENTS` has 0–1 entries | `AGENTS` has 2+ entries |
| agents see each other's findings | n/a (one agent) | ✅ (the panel critiques each other) |
| who critiques | nobody (just synthesized) | the agents critique **each other** |
| extra tool-using passes | none | `N × AGENTIC_DISCUSSION_ROUNDS` |
| best at | speed, low cost | precision — killing false positives, filling gaps |
| guardrail | — | anti-sycophancy prompt + tool verification |

**Mental model:** *single agent* = one reviewer's report, finalized.
*multiple agents* = the reviewers argue it out (refute weak claims, defend
strong ones, surface what others missed), then the editor finalizes.

---

## Configuration reference

Each setting maps to a part of the pipeline above. Every setting can be passed
as an **environment variable** or as an **action input** (lower-cased name). All
are optional except where noted.

| Env var | Action input | Type | Default | Controls |
|---|---|---|---|---|
| `AGENTIC_REVIEW` | `agentic_review` | bool | `false` | Master switch. `false` = standard single-shot review (no panel/tools). |
| `AGENTS` | `agents` | agent list | _(empty)_ | The **explore** panel — and it sets the mode: 0–1 agents = single (explore→synthesize), 2+ agents = discussion (explore→discuss→synthesize). When set, overrides `LLM_MODEL` and the top-level `LLM_*` may be omitted. |
| `AGENTIC_DISCUSSION_ROUNDS` | `agentic_discussion_rounds` | int > 0 | `2` | How many **discuss** rounds (the `R` in the diagram) when there are 2+ agents. Each round = every agent critiques the others once. Ignored for a single agent. |
| `SYNTHESIS_AGENT` | `synthesis_agent` | single agent | `LLM_MODEL` | The **synthesize** agent (the judge that merges findings → inline comments). |
| `AGENTIC_MAX_STEPS` | `agentic_max_steps` | int > 0 | `12` | Caps each agent's **tool loop** in a single explore/discussion call. One step = one model turn (optionally a tool call + its result). Higher = the agent can read more files / grep more before concluding (more thorough, more cost/latency); lower = faster/cheaper but shallower. It is *not* the number of comments or rounds. |

Built on the existing base settings (used as fallbacks for every agent):

| Env var | Purpose |
|---|---|
| `LLM_PROVIDER` | Must be `ai-sdk` for agentic review (this is the default). |
| `LLM_MODEL` | The base/single model. Required unless a self-describing `AGENTS` panel is used; also the default for `SYNTHESIS_AGENT`. |
| `LLM_API_KEY` | Default key fallback for agents without an `apiKeyEnv`. Optional when every agent (and the synthesis agent) sets `apiKeyEnv`. |
| `LLM_BASE_URL` | Default base URL (e.g. OpenRouter) for agents without their own `baseUrl`. |
| `STYLE_GUIDE_RULES` / `style_guide_rules` | Extra rules injected directly into the explore/discussion prompts. |

> **Model requirement:** because the explore/discussion phases use tool calling,
> each agent's model must support function/tool calling (Claude, GPT-4o/4.1/5,
> Gemini 2.x, DeepSeek v4, Kimi k2, etc.).

---

## Agent spec format

### `AGENTS` (the panel)

Two accepted forms.

**Comma-separated model names** — `id` defaults to the model, everything else
inherits the base `LLM_*`:

```yaml
AGENTS: "anthropic/claude-sonnet-4.5, google/gemini-2.5-pro"
```

**JSON array of objects** — full control per agent:

```yaml
AGENTS: >-
  [
    {"id":"security","model":"anthropic/claude-sonnet-4.5",
     "instructions":"Focus on security: injection, authz, secrets, SSRF."},
    {"id":"correctness","model":"google/gemini-2.5-pro",
     "instructions":"Focus on logic bugs, edge cases, broken contracts.",
     "provider":"ai-sdk","baseUrl":"https://openrouter.ai/api/v1","apiKeyEnv":"OPENROUTER_API_KEY"}
  ]
```

### `SYNTHESIS_AGENT` (single agent)

A bare model name, or a JSON object:

```yaml
SYNTHESIS_AGENT: "openai/gpt-5"
# or
SYNTHESIS_AGENT: '{"id":"judge","model":"openai/gpt-5","instructions":"Be conservative; keep only high-confidence findings."}'
```

### Agent fields

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | ✅ | — | Model id (e.g. `anthropic/claude-sonnet-4.5`). |
| `id` | — | the `model` | Label used in logs, discussion references, and synthesis attribution. |
| `instructions` | — | _(none)_ | Focus/persona appended to that agent's prompt (see [instruction usage](#prompt-composition--instruction-usage)). |
| `provider` | — | `LLM_PROVIDER` (→ `ai-sdk`) | Must resolve to `ai-sdk`. |
| `baseUrl` | — | `LLM_BASE_URL` | Per-agent endpoint (mix providers across the panel). |
| `apiKeyEnv` | — | _(falls back to `LLM_API_KEY`)_ | **Name** of an env var holding this agent's key (e.g. `OPENROUTER_API_KEY`). Raw keys are not accepted — see [Secrets & multi-platform](#secrets--multi-platform). |

Precedence: `AGENTS` non-empty ⇒ each agent self-describes and the top-level
`LLM_MODEL`/`LLM_API_KEY` may be left unset. `AGENTS` empty ⇒ a single agent
built from `LLM_MODEL` is used.

### Secrets & multi-platform

Never put raw API keys in `AGENTS` (it's a plain config string). Instead:

1. Add each platform's key as a **repo secret** (e.g. `OPENROUTER_API_KEY`,
   `ANTHROPIC_API_KEY`).
2. Expose them as **env vars** in the workflow step:
   ```yaml
   env:
     OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
     ANTHROPIC_API_KEY:  ${{ secrets.ANTHROPIC_API_KEY }}
   ```
3. Reference them **by name** in each agent via `apiKeyEnv`:
   ```yaml
   AGENTS: >-
     [{"id":"sec","model":"anthropic/claude-sonnet-4.5","baseUrl":"https://openrouter.ai/api/v1","apiKeyEnv":"OPENROUTER_API_KEY"},
      {"id":"logic","model":"claude-sonnet-4-5","apiKeyEnv":"ANTHROPIC_API_KEY"}]
   ```

Key resolution per agent: `env[apiKeyEnv]` → top-level `LLM_API_KEY`. An agent
with no `apiKeyEnv` uses `LLM_API_KEY` (itself sourced from a secret). In
multi-agent mode where every agent sets `apiKeyEnv`, `LLM_API_KEY` /
`LLM_MODEL` / `LLM_BASE_URL` can all be omitted.

---

## What each agent can access

| | Code changes (diff) | Code **outside** the diff | Repo rules (CLAUDE.md / AGENTS.md / .claude/rules / skills) | `style_guide_rules` |
|---|---|---|---|---|
| **Explore agents** | ✅ in prompt | ✅ via `read_file` + `grep` | ✅ via `list_guidelines` tool ("ALWAYS call first") | ✅ in prompt |
| **Discussion agents** | ✅ in prompt | ✅ via tools | ✅ via `list_guidelines` tool | ✅ in prompt |
| **Synthesis agent** | ✅ in prompt | ❌ no tools | ❌ only via the explorers' notes | ❌ not in its prompt |

```
 EXPLORE / DISCUSS agent gets:                   SYNTHESIS agent gets:
 ┌─────────────────────────────┐                ┌─────────────────────────────┐
 │ • PR diff           (prompt) │                │ • PR diff          (prompt) │
 │ • style_guide_rules (prompt) │                │ • all agents' notes(prompt) │
 │ • TOOLS:                     │                │ • PR title + summary        │
 │    read_file  (context)      │                │ • NO tools                  │
 │    grep       (context)      │                │   → no live code/rules      │
 │    list_guidelines (rules)   │                └─────────────────────────────┘
 └─────────────────────────────┘
```

Notes:
- Repo rules are **tool-gated**: agents are instructed to call `list_guidelines`
  first, but it is not forced. `style_guide_rules` (the action input) is the
  only rule source injected into the prompt every time.
- Synthesis is deliberately "blind" to the repo — it merges the notes and maps
  them to diff line numbers (it has the diff for that), trusting the explorers
  to have applied the rules.

---

## Prompt composition & instruction usage

Each agent's prompt is built in **layers**, so output format and tool usage stay
consistent while each agent can specialize:

```
   [ base review prompt ]   role + tools + output contract   (constant)
 + [ phase instruction  ]   explore | discuss | synthesize   (per phase)
 + [ <Your Focus> block ]   the agent's own `instructions`    (per agent)
```

### The injection point

The base explore system prompt ends with the agent's focus block:

```
…When done investigating, output your findings as concise markdown notes. …{{AGENT_FOCUS}}
```

`{{AGENT_FOCUS}}` is produced by:

```ts
function agentFocus(agent) {
  return agent.instructions?.trim()
    ? `\n<Your Focus>\n${agent.instructions.trim()}\n</Your Focus>\n`
    : "";   // no instructions → nothing added
}
```

### Example

Agent:

```json
{ "id": "security", "model": "anthropic/claude-sonnet-4.5",
  "instructions": "Focus on security: injection, authz, secrets, SSRF. Ignore style." }
```

Appended to its system prompt (in **both** explore and discussion, since
discussion reuses the base prompt):

```
<Your Focus>
Focus on security: injection, authz, secrets, SSRF. Ignore style.
</Your Focus>
```

So with `AGENTS = [security, correctness]` (2 agents → they discuss):

```
security agent    = base + <Focus: security…>   → explores, then discusses correctness (still security-lens)
correctness agent = base + <Focus: logic bugs…> → explores, then discusses security (still logic-lens)
synthesis agent   = synthesis prompt + <Focus: …if SYNTHESIS_AGENT.instructions set>
```

`SYNTHESIS_AGENT.instructions` flows through the same `agentFocus()` hook into
the synthesis prompt, so you can tune merge behavior (e.g. "be conservative;
keep only high-confidence findings").

### The phase instructions (verbatim)

**Discussion** (appended to the base prompt):

```
You are now in a DISCUSSION with the other reviewers. Critically evaluate their
findings — do NOT rubber-stamp them. For each of their findings: AGREE only if
you can confirm it (verify with read_file/grep), REFUTE it with a concrete
reason if you believe it is wrong or a false positive, and ADD any issues
everyone missed. Re-state your own findings, dropping any of yours that were
correctly refuted. Output your updated findings in the same notes format.
```

**Synthesis** (key rules):

```
- Merge duplicate or overlapping findings into a single comment.
- Weight by cross-agent agreement: findings multiple reviewers independently
  raised or confirmed are high-confidence; keep them. Drop findings that another
  reviewer refuted and no one defended, and anything not actionable.
- 'critical' must be true only for findings that should block merge.
- Map each comment to precise '__new hunk__' line numbers; omit unmappable ones.
```

---

## Cost & latency

Roughly, number of LLM calls (N = panel size, R = `AGENTIC_DISCUSSION_ROUNDS`):

| Mode | LLM calls |
|---|---|
| single agent | `1 explore + 1 synthesis` |
| `single` (N agents) | `N explore + 1 synthesis` |
| `discussion` (N agents) | `N explore + (N × R) discussion + 1 synthesis` |

Each explore/discussion call may internally loop up to `AGENTIC_MAX_STEPS` tool
steps. Multiple agents and discussion rounds multiply tokens and wall-clock on
**every PR** — start with one or two agents and 1 discussion round, and use
cheaper models for explorers if needed.

---

## Failure handling

- **An explorer fails** → its findings are dropped; synthesis proceeds with the
  survivors.
- **All explorers fail** → an empty review is returned (no synthesis call).
- **A discussion call fails** → that agent keeps its pre-discussion position for
  the round.
- **Single agent + `discussion`** → discussion is skipped (no peer).
- **Non-`ai-sdk` provider + `AGENTIC_REVIEW=true`** → warning + fallback to the
  standard single-shot review.

---

## Full examples

**Panel of two (2 agents → they discuss), multi-platform via `apiKeyEnv`, custom synthesis model:**

```yaml
- uses: actions/checkout@v4
- uses: presubmit/ai-reviewer@latest
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}   # referenced by apiKeyEnv
    AGENTIC_REVIEW: "true"
    AGENTS: >-
      [
        {"id":"security","model":"anthropic/claude-sonnet-4.5","baseUrl":"https://openrouter.ai/api/v1","apiKeyEnv":"OPENROUTER_API_KEY","instructions":"Focus on security and input validation."},
        {"id":"correctness","model":"google/gemini-2.5-pro","baseUrl":"https://openrouter.ai/api/v1","apiKeyEnv":"OPENROUTER_API_KEY","instructions":"Focus on logic bugs and edge cases."}
      ]
    SYNTHESIS_AGENT: '{"id":"judge","model":"openai/gpt-5","baseUrl":"https://openrouter.ai/api/v1","apiKeyEnv":"OPENROUTER_API_KEY"}'
```

**Three agents, two discussion rounds (single OpenRouter key as the default):**

```yaml
    LLM_PROVIDER: ai-sdk
    LLM_BASE_URL: https://openrouter.ai/api/v1
    LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
    AGENTIC_REVIEW: "true"
    AGENTIC_DISCUSSION_ROUNDS: "2"
    AGENTS: "anthropic/claude-sonnet-4.5, google/gemini-2.5-pro, openai/gpt-5"
```

**Single agentic agent (simplest upgrade from standard review):**

```yaml
    AGENTIC_REVIEW: "true"
    LLM_MODEL: anthropic/claude-sonnet-4.5
```
