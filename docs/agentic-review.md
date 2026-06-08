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
- [Configuration reference](#configuration-reference)
- [Agent spec format](#agent-spec-format)
- [Pipeline & modes](#pipeline--modes)
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

A panel of two agents that discuss:

```yaml
    AGENTIC_REVIEW: "true"
    REVIEW_MODE: discussion
    AGENTS: "anthropic/claude-sonnet-4.5, google/gemini-2.5-pro"
```

---

## Configuration reference

Every setting can be passed as an **environment variable** or as an **action
input** (lower-cased name). All are optional except where noted.

| Env var | Action input | Type | Default | Purpose |
|---|---|---|---|---|
| `AGENTIC_REVIEW` | `agentic_review` | bool | `false` | Master switch. `false` = standard single-shot review. |
| `AGENTS` | `agents` | agent list | _(empty)_ | The explorer panel. When set, **overrides `LLM_MODEL`**. When empty, the single `LLM_MODEL` is used. |
| `SYNTHESIS_AGENT` | `synthesis_agent` | single agent | `LLM_MODEL` | The judge that merges findings into the final structured review. Independent config; does **not** reuse an explorer agent. |
| `REVIEW_MODE` | `review_mode` | `single` \| `discussion` | `single` | `single` = explore → synthesize. `discussion` = the panel critiques each other first (needs 2+ agents). |
| `AGENTIC_DISCUSSION_ROUNDS` | `agentic_discussion_rounds` | int > 0 | `1` | Peer-discussion rounds when `REVIEW_MODE=discussion`. |
| `AGENTIC_MAX_STEPS` | `agentic_max_steps` | int > 0 | `12` | Max tool-use steps per agent per call. |

Built on the existing base settings (used as fallbacks for every agent):

| Env var | Purpose |
|---|---|
| `LLM_PROVIDER` | Must be `ai-sdk` for agentic review. |
| `LLM_MODEL` | Required. The base model — used when `AGENTS` is empty, and as the default for `SYNTHESIS_AGENT`. |
| `LLM_API_KEY` | Default API key for every agent. |
| `LLM_BASE_URL` | Default base URL (e.g. OpenRouter) for every agent. |
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
     "provider":"ai-sdk","baseUrl":"https://openrouter.ai/api/v1"}
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
| `provider` | — | `LLM_PROVIDER` | Must resolve to `ai-sdk`. |
| `baseUrl` | — | `LLM_BASE_URL` | Per-agent endpoint (mix providers across the panel). |
| `apiKey` | — | `LLM_API_KEY` | Per-agent key. |

Precedence: `AGENTS` non-empty ⇒ `LLM_MODEL` is ignored for exploring (still
the default for `SYNTHESIS_AGENT`). `AGENTS` empty ⇒ a single agent built from
`LLM_MODEL` is used.

---

## Pipeline & modes

Every agentic run is **explore → [discuss] → synthesize**. Tools are available
during explore and discussion (not synthesis).

```
   tools available to explore/discussion agents:
   ┌────────────────┬───────────────────────────────────────────┐
   │ list_guidelines │ CLAUDE.md, AGENTS.md, .claude/rules, skills │
   │ read_file       │ surrounding code / definitions / callers    │
   │ grep            │ usages & related patterns across the repo   │
   └────────────────┴───────────────────────────────────────────┘
```

### Single agent (`AGENTS` empty → uses `LLM_MODEL`)

```
        ┌─ EXPLORE ─────────────────┐   ┌─ SYNTHESIZE ────────┐
 PR ───▶ │ agent loops with tools    │──▶│ SYNTHESIS_AGENT     │──▶ comments
 diff    │  → findings notes         │   │ notes → diff lines  │
        └───────────────────────────┘   └─────────────────────┘
```

### `REVIEW_MODE=single` — panel reviews, then merge

```
                 ┌─ EXPLORE (concurrent) ─────────┐
            ┌───▶│  ┌────────────────────────┐   │──┐ notes_A
            │    │  │ "security"   (model A) │   │  │
 PR ───────▶┤    │  │ + tools, own focus     │   │  │      ┌─ SYNTHESIZE ──────┐
 diff       │    │  └────────────────────────┘   │  ├─────▶│ SYNTHESIS_AGENT   │──▶ comments
            │    │  ┌────────────────────────┐   │  │      │ merge + de-dupe   │
            └───▶│  │ "correctness"(model B) │   │──┘ notes_B │ → diff lines    │
                 │  │ + tools, own focus     │   │         └───────────────────┘
                 │  └────────────────────────┘   │
                 └────────────────────────────────┘
   agents never see each other's work · cheapest multi-agent · broad coverage
```

### `REVIEW_MODE=discussion` — panel critiques EACH OTHER, then merge

```
   ┌─ EXPLORE (concurrent)┐   ┌─ DISCUSS · R rounds (concurrent each round) ───────┐   ┌─ SYNTHESIZE ─────┐
A: │ "security"  ─notes_A─┼──▶│  A sees {B}  ─▶ agree / refute(verify) / add ─▶ A' │──▶│ SYNTHESIS_AGENT  │
   │ + tools, focus       │   │            ╲ ╱   keeps A's own model+persona       │   │ weight by        │──▶ comments
 PR│                      │   │             ╳                                      │   │ cross-agent      │
   │                      │   │            ╱ ╲                                     │   │ agreement →      │
B: │ "correctness"─notes_B┼──▶│  B sees {A}  ─▶ agree / refute(verify) / add ─▶ B' │──▶│ diff lines       │
   │ + tools, focus       │   │   (each round runs on the prev round's snapshot)   │   └──────────────────┘
   └──────────────────────┘   │   fail → that agent keeps its prior notes          │
                              └────────────────────────────────────────────────────┘
                                 skipped if < 2 agents · "do NOT rubber-stamp"
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

| | single | discussion |
|---|---|---|
| agents see each other's findings | ❌ | ✅ (the panel, not an external critic) |
| who critiques | nobody (just merged) | the agents critique **each other** |
| extra tool-using passes | none | `N × AGENTIC_DISCUSSION_ROUNDS` |
| needs | 1+ agents | 2+ agents |
| best at | coverage, low cost | precision — killing false positives, filling gaps |
| guardrail | — | anti-sycophancy prompt + tool verification |

**Mental model:** *single* = independent reports stapled together by an editor.
*discussion* = the same reviewers argue it out (refute weak claims, defend
strong ones, surface what others missed), then the editor finalizes.

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

So with `AGENTS = [security, correctness]` and `REVIEW_MODE=discussion`:

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

Roughly, number of LLM calls (N = panel size, R = discussion rounds):

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

**Single mode, panel of two, mixed providers, custom synthesis model:**

```yaml
- uses: actions/checkout@v4
- uses: presubmit/ai-reviewer@latest
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    LLM_API_KEY: ${{ secrets.LLM_API_KEY }}     # OpenRouter key
    LLM_PROVIDER: ai-sdk
    LLM_BASE_URL: https://openrouter.ai/api/v1
    LLM_MODEL: deepseek/deepseek-v4-flash       # base + synthesis default
    AGENTIC_REVIEW: "true"
    REVIEW_MODE: single
    AGENTS: >-
      [
        {"id":"security","model":"anthropic/claude-sonnet-4.5","instructions":"Focus on security and input validation."},
        {"id":"correctness","model":"google/gemini-2.5-pro","instructions":"Focus on logic bugs and edge cases."}
      ]
    SYNTHESIS_AGENT: "openai/gpt-5"
```

**Discussion, two rounds:**

```yaml
    AGENTIC_REVIEW: "true"
    REVIEW_MODE: discussion
    AGENTIC_DISCUSSION_ROUNDS: "2"
    AGENTS: "anthropic/claude-sonnet-4.5, google/gemini-2.5-pro, openai/gpt-5"
```

**Single agentic agent (simplest upgrade from standard review):**

```yaml
    AGENTIC_REVIEW: "true"
    LLM_MODEL: anthropic/claude-sonnet-4.5
```
