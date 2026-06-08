import { getInput, getMultilineInput } from "@actions/core";
import { AIProviderType } from "./ai";

/**
 * A reviewer agent. `model` is required; `provider`/`baseUrl`/`apiKey` fall back
 * to the top-level `llmProvider` / `llmBaseUrl` / `llmApiKey` when omitted, so a
 * single agent can be just a model name while another points at a completely
 * different provider/endpoint. `id` is a human label (logs, attribution) and
 * `instructions` is an optional focus/persona appended to that agent's prompt.
 *
 * Keys: use `apiKeyEnv` — the NAME of an environment variable holding the key
 * (e.g. "OPENROUTER_API_KEY") — so a panel can span platforms/billing without
 * putting raw secrets in the AGENTS config. Raw keys are intentionally not
 * accepted here. Resolution: env[apiKeyEnv] -> top-level LLM_API_KEY.
 */
export type AgentSpec = {
  id: string;
  model: string;
  instructions?: string;
  provider?: string;
  baseUrl?: string;
  apiKeyEnv?: string;
};

const DEFAULT_AGENTIC_MAX_STEPS = 12;
const DEFAULT_AGENTIC_DISCUSSION_ROUNDS = 2;

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const n = Number.parseInt(value || "", 10);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function parseBool(value: string | undefined): boolean {
  return value === "1" || value?.toLowerCase() === "true";
}

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

/** Normalize one parsed entry (object or bare model string) into an AgentSpec. */
function toAgentSpec(entry: unknown): AgentSpec | null {
  if (typeof entry === "string") {
    const model = entry.trim();
    return model ? { id: model, model } : null;
  }
  if (entry && typeof entry === "object") {
    const e = entry as Record<string, unknown>;
    const model = str(e.model);
    if (!model) return null;
    return {
      id: str(e.id) ?? model,
      model,
      instructions: str(e.instructions),
      provider: str(e.provider),
      baseUrl: str(e.baseUrl),
      apiKeyEnv: str(e.apiKeyEnv),
    };
  }
  return null;
}

/**
 * Parse the `AGENTS` setting into a list of {@link AgentSpec}.
 *
 * Two accepted forms:
 *  - JSON array of objects/strings:
 *      [{"id":"security","model":"anthropic/claude-sonnet-4.5","instructions":"..."}, "google/gemini-2.5-pro"]
 *  - Convenience comma-separated model names (id defaults to the model):
 *      anthropic/claude-sonnet-4.5, google/gemini-2.5-pro
 *
 * Returns an empty array when unset/blank so the single `llmModel` takes effect.
 */
export function parseAgents(value: string | undefined): AgentSpec[] {
  const raw = (value ?? "").trim();
  if (!raw) {
    return [];
  }

  if (raw.startsWith("[")) {
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed
        .map(toAgentSpec)
        .filter((a): a is AgentSpec => a !== null);
    } catch (e) {
      console.error("Error parsing AGENTS as JSON:", e);
      return [];
    }
  }

  return raw
    .split(",")
    .map((m) => m.trim())
    .filter((m) => m.length > 0)
    .map((model) => ({ id: model, model }));
}

/**
 * Parse a single-agent setting (e.g. `SYNTHESIS_AGENT`) into one
 * {@link AgentSpec}. Accepts a JSON object or a bare model-name string. Returns
 * undefined when unset/blank so the role falls back to a default agent.
 */
export function parseAgent(value: string | undefined): AgentSpec | undefined {
  const raw = (value ?? "").trim();
  if (!raw) {
    return undefined;
  }
  if (raw.startsWith("{")) {
    try {
      return toAgentSpec(JSON.parse(raw)) ?? undefined;
    } catch (e) {
      console.error("Error parsing single agent as JSON:", e);
      return undefined;
    }
  }
  return { id: raw, model: raw };
}

export class Config {
  public llmApiKey: string | undefined;
  public llmModel: string | undefined;
  public llmProvider: string;
  public llmBaseUrl: string | undefined;
  public githubToken: string | undefined;
  public styleGuideRules: string | undefined;
  public githubApiUrl: string;
  public githubServerUrl: string;

  // Agentic review (opt-in, default off). When enabled, the reviewer reads the
  // diff with repo-context tools (files, CLAUDE.md/AGENTS.md/rules/skills) before
  // commenting, and can run a panel of agents together. See src/agentic/.
  public agenticReview: boolean;
  public agents: AgentSpec[]; // explorer panel; empty -> use single llmModel
  public synthesisAgent?: AgentSpec; // judge that merges notes -> structured review
  public agenticMaxSteps: number;
  public agenticDiscussionRounds: number; // peer-discussion rounds when 2+ agents

  public sapAiCoreClientId: string | undefined;
  public sapAiCoreClientSecret: string | undefined;
  public sapAiCoreTokenUrl: string | undefined;
  public sapAiCoreBaseUrl: string | undefined;
  public sapAiResourceGroup: string | undefined;

  constructor() {
    this.githubToken = process.env.GITHUB_TOKEN;
    if (!this.githubToken) {
      throw new Error("GITHUB_TOKEN is not set");
    }

    // Top-level provider/base URL/key are the single-model defaults and the
    // per-agent fallbacks. In multi-agent agentic mode each agent self-describes
    // (model + provider + baseUrl + key), so these may be left unset.
    this.llmProvider = process.env.LLM_PROVIDER || getInput("llm_provider");
    if (!this.llmProvider?.length) {
      this.llmProvider = AIProviderType.AI_SDK;
    }
    this.llmApiKey = process.env.LLM_API_KEY;
    const baseUrlFromEnv = process.env.LLM_BASE_URL;
    const baseUrlFromInput = getInput("llm_base_url");
    this.llmBaseUrl = baseUrlFromEnv || baseUrlFromInput || undefined;
    this.llmModel = process.env.LLM_MODEL || getInput("llm_model");

    // Agentic review configuration (all optional, default to the existing
    // single-shot behavior when unset).
    this.agenticReview = parseBool(
      process.env.AGENTIC_REVIEW || getInput("agentic_review")
    );

    // Explorer panel. Empty by default: when no agents are given, the single
    // `llmModel` takes effect. When non-empty, `agents` takes precedence and
    // each agent self-describes its model/provider/baseUrl/key.
    this.agents = parseAgents(process.env.AGENTS || getInput("agents"));

    // Synthesis (judge) agent (optional). Falls back to llmModel, else the first
    // agent, in the orchestrator when unset.
    this.synthesisAgent = parseAgent(
      process.env.SYNTHESIS_AGENT || getInput("synthesis_agent")
    );

    // When the agentic reviewer runs with its own panel, the top-level
    // LLM_MODEL / LLM_API_KEY are not required (each agent self-describes).
    const agenticMultiAgent = this.agenticReview && this.agents.length > 0;
    const isSapAiSdk = this.llmProvider === AIProviderType.SAP_AI_SDK;
    if (!this.llmModel?.length && !agenticMultiAgent) {
      throw new Error("LLM_MODEL is not set");
    }
    // SAP AI SDK does not require an API key.
    if (!this.llmApiKey && !isSapAiSdk && !agenticMultiAgent) {
      throw new Error("LLM_API_KEY is not set");
    }

    this.agenticMaxSteps = parsePositiveInt(
      process.env.AGENTIC_MAX_STEPS || getInput("agentic_max_steps"),
      DEFAULT_AGENTIC_MAX_STEPS
    );

    this.agenticDiscussionRounds = parsePositiveInt(
      process.env.AGENTIC_DISCUSSION_ROUNDS ||
        getInput("agentic_discussion_rounds"),
      DEFAULT_AGENTIC_DISCUSSION_ROUNDS
    );

    // SAP AI Core configuration
    this.sapAiCoreClientId = process.env.SAP_AI_CORE_CLIENT_ID;
    this.sapAiCoreClientSecret = process.env.SAP_AI_CORE_CLIENT_SECRET;
    this.sapAiCoreTokenUrl = process.env.SAP_AI_CORE_TOKEN_URL;
    this.sapAiCoreBaseUrl = process.env.SAP_AI_CORE_BASE_URL;
    this.sapAiResourceGroup = process.env.SAP_AI_RESOURCE_GROUP;
    if (
      isSapAiSdk &&
      (!this.sapAiCoreClientId ||
        !this.sapAiCoreClientSecret ||
        !this.sapAiCoreTokenUrl ||
        !this.sapAiCoreBaseUrl)
    ) {
      throw new Error(
        "SAP AI Core configuration is not set. Please set SAP_AI_CORE_CLIENT_ID, SAP_AI_CORE_CLIENT_SECRET, SAP_AI_CORE_TOKEN_URL, and SAP_AI_CORE_BASE_URL."
      );
    }

    // GitHub Enterprise Server support
    this.githubApiUrl =
      process.env.GITHUB_API_URL || getInput('github_api_url') || 'https://api.github.com';
    this.githubServerUrl =
      process.env.GITHUB_SERVER_URL || getInput('github_server_url') || 'https://github.com';

    if (!process.env.DEBUG) {
      return;
    }
    console.log("[debug] loading extra inputs from .env");

    this.styleGuideRules = process.env.STYLE_GUIDE_RULES;
  }

  public loadInputs() {
    if (process.env.DEBUG) {
      console.log("[debug] skip loading inputs");
      return;
    }

    // Custom style guide rules
    try {
      const styleGuideRules = getMultilineInput("style_guide_rules") || [];
      if (
        Array.isArray(styleGuideRules) &&
        styleGuideRules.length &&
        styleGuideRules[0].trim().length
      ) {
        this.styleGuideRules = styleGuideRules.join("\n");
      }
    } catch (e) {
      console.error("Error loading style guide rules:", e);
    }
  }
}

// For testing, we'll modify how the config instance is created
// This prevents the automatic loading when the module is imported
let configInstance: Config | null = null;

// If not in test environment, create and configure the instance
if (process.env.NODE_ENV !== "test") {
  configInstance = new Config();
  configInstance.loadInputs();
}

// Export the instance or a function to create one for tests
export default process.env.NODE_ENV === "test"
  ? {
      // Default values for tests
      githubToken: "mock-token",
      llmApiKey: "mock-api-key",
      llmModel: "mock-model",
      llmProvider: "mock-provider",
      llmBaseUrl: undefined,
      agenticReview: false,
      agents: [] as AgentSpec[],
      synthesisAgent: undefined as AgentSpec | undefined,
      agenticMaxSteps: DEFAULT_AGENTIC_MAX_STEPS,
      agenticDiscussionRounds: DEFAULT_AGENTIC_DISCUSSION_ROUNDS,
      styleGuideRules: "",
      sapAiCoreClientId: "mock-client-id",
      sapAiCoreClientSecret: "mock-client-secret",
      sapAiCoreTokenUrl: "mock-token-url",
      sapAiCoreBaseUrl: "mock-base-url",
      sapAiResourceGroup: "default",
      githubApiUrl: "https://api.github.com",
      githubServerUrl: "https://github.com",
      loadInputs: jest.fn(),
    }
  : configInstance!;
