import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { generateText, generateObject, type CoreTool } from "ai";
import { z } from "zod";
import config, { type AgentSpec } from "./config";
import { AISDKProvider } from "./providers/ai-sdk";
import { SAPAIProvider } from "./providers/sapaicore";

export enum AIProviderType {
  AI_SDK = "ai-sdk",
  SAP_AI_SDK = "sap-ai-sdk",
}

const LLM_MODELS: Record<AIProviderType, ModelConfig[]> = {
  [AIProviderType.AI_SDK]: [
    // Anthropic
    {
      name: "claude-3-5-sonnet-20240620",
      createAi: createAnthropic,
    },
    {
      name: "claude-3-5-sonnet-20241022",
      createAi: createAnthropic,
    },
    {
      name: "claude-3-7-sonnet-20250219",
      createAi: createAnthropic,
    },
    {
      name: "claude-sonnet-4-20250514",
      createAi: createAnthropic,
    },
    {
      name: "claude-opus-4-20250514",
      createAi: createAnthropic,
    },
    {
      name: "claude-opus-4-1-20250805",
      createAi: createAnthropic,
    },
    {
      name: "claude-sonnet-4-5-20250929",
      createAi: createAnthropic,
    },
    {
      name: "claude-sonnet-4-5",
      createAi: createAnthropic,
    },
    // OpenAI
    {
      name: "gpt-5",
      createAi: createOpenAI,
      temperature: 1,
    },
    {
      name: "gpt-5-mini",
      createAi: createOpenAI,
      temperature: 1,
    },
    {
      name: "gpt-5-nano",
      createAi: createOpenAI,
      temperature: 1,
    },
    {
      name: "gpt-4.1-mini",
      createAi: createOpenAI,
    },
    {
      name: "gpt-4o-mini",
      createAi: createOpenAI,
    },
    {
      name: "o1",
      createAi: createOpenAI,
    },
    {
      name: "o1-mini",
      createAi: createOpenAI,
    },
    {
      name: "o3-mini",
      createAi: createOpenAI,
      temperature: 1,
    },
    {
      name: "o4-mini",
      createAi: createOpenAI,
      temperature: 1,
    },
    {
      name: "gpt-4.1",
      createAi: createOpenAI,
    },
    // Google stable models https://ai.google.dev/gemini-api/docs/models/gemini
    {
      name: "gemini-2.0-flash-001",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.0-flash-lite-preview-02-05",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-1.5-flash",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-1.5-flash-latest",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-1.5-flash-8b",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-1.5-pro",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.5-pro",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.5-flash",
      createAi: createGoogleGenerativeAI,
    },
    // Google experimental models https://ai.google.dev/gemini-api/docs/models/experimental-models
    {
      name: "gemini-2.5-pro-preview-05-06",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.5-flash-preview-04-17",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.0-pro-exp-02-05",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.0-flash-thinking-exp-01-21",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.5-flash-preview-05-20",
      createAi: createGoogleGenerativeAI,
    },
    {
      name: "gemini-2.5-flash-lite-preview-06-17",
      createAi: createGoogleGenerativeAI,
    },
  ],
  [AIProviderType.SAP_AI_SDK]: [
    {
      name: "anthropic--claude-3.7-sonnet",
    },
    {
      name: "anthropic--claude-3.5-sonnet",
    },
    {
      name: "anthropic--claude-3-sonnet",
    },
    {
      name: "anthropic--claude-3-haiku",
    },
    {
      name: "anthropic--claude-3-opus",
    },
    {
      name: "gpt-4o",
    },
    {
      name: "gpt-4",
    },
    {
      name: "gpt-4o-mini",
    },
    {
      name: "o1",
    },
    {
      name: "gpt-4.1",
    },
    {
      name: "gpt-4.1-nano",
    },
    {
      name: "gpt-5",
    },
    {
      name: "gpt-5-mini",
    },
    {
      name: "gpt-5-nano",
    },
    {
      name: "o3-mini",
    },
    {
      name: "o3",
    },
    {
      name: "o4-mini",
    },
  ],
};

export type InferenceConfig = {
  prompt: string;
  temperature?: number;
  system?: string;
  schema: z.ZodObject<any, any>;
};

export interface AIProvider {
  runInference(params: InferenceConfig): Promise<any>;
}

class AIProviderFactory {
  static getProvider(
    provider: AIProviderType,
    modelConfig: ModelConfig
  ): AIProvider {
    switch (provider) {
      case AIProviderType["AI_SDK"]:
        if (!modelConfig.createAi) {
          throw new Error(
            `No createAi function found for model ${modelConfig.name}`
          );
        }
        return new AISDKProvider(modelConfig.createAi, modelConfig.name);
      case AIProviderType["SAP_AI_SDK"]:
        return new SAPAIProvider(modelConfig.name);
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }
}

type ModelConfig = {
  name: string;
  createAi?: any;
  temperature?: number;
};

export async function runPrompt({
  prompt,
  systemPrompt,
  schema,
}: {
  prompt: string;
  systemPrompt?: string;
  schema: z.ZodObject<any, any>;
}) {
  if (
    !Object.values(AIProviderType).includes(
      config.llmProvider as AIProviderType
    )
  ) {
    throw new Error(
      `Unknown LLM provider: ${
        config.llmProvider
      }. Valid providers are: ${Object.keys(AIProviderType).join(", ")}`
    );
  }
  const providerType = config.llmProvider as AIProviderType;
  const providerModels = LLM_MODELS[providerType];
  let modelConfig = providerModels.find((m) => m.name === config.llmModel);

  // When using a custom base URL, skip whitelist validation and use OpenAI SDK
  if (!modelConfig && config.llmBaseUrl && providerType === AIProviderType.AI_SDK) {
    modelConfig = {
      name: config.llmModel!,
      createAi: createOpenAI,
    };
  }

  if (!modelConfig) {
    throw new Error(
      `Unknown LLM model: ${config.llmModel}. For provider ${
        config.llmProvider
      }, supported models are: ${providerModels.map((m) => m.name).join(", ")}`
    );
  }

  // Get the appropriate provider for this model
  const provider = AIProviderFactory.getProvider(providerType, modelConfig);

  // Run the inference using the provider
  return await provider.runInference({
    prompt,
    temperature: modelConfig.temperature,
    system: systemPrompt,
    schema,
  });
}

// ===== Agentic review support (additive; used only by src/agentic) =====
//
// Resolves an AI SDK language model for a single agent, reusing the same
// allowlist + LLM_BASE_URL fallback rules as runPrompt above. Unknown models
// are allowed through the OpenAI-compatible client when a base URL is set
// (e.g. OpenRouter), otherwise they error with the supported-model list.
function createAiSdkLanguageModel(spec: {
  model: string;
  baseUrl?: string;
  apiKey?: string;
}) {
  const modelName = spec.model;
  const baseUrl = spec.baseUrl ?? config.llmBaseUrl;
  const apiKey = spec.apiKey ?? config.llmApiKey;

  const providerModels = LLM_MODELS[AIProviderType.AI_SDK];
  let modelConfig = providerModels.find((m) => m.name === modelName);

  // When using a custom base URL, skip whitelist validation and use OpenAI SDK.
  if (!modelConfig && baseUrl) {
    modelConfig = { name: modelName, createAi: createOpenAI };
  }

  if (!modelConfig || !modelConfig.createAi) {
    throw new Error(
      `Unknown LLM model: ${modelName}. Set LLM_BASE_URL for OpenAI-compatible ` +
        `providers (e.g. OpenRouter), or use one of: ` +
        `${providerModels.map((m) => m.name).join(", ")}`
    );
  }

  const provider = modelConfig.createAi({
    apiKey,
    ...(baseUrl && { baseURL: baseUrl }),
  });
  return provider(modelName);
}

// Validate an agent is usable for an agentic (ai-sdk) call and return its
// resolved {model, baseUrl, apiKey}. Throws on non-ai-sdk providers / no model.
// API key resolution: env[apiKeyEnv] -> top-level LLM_API_KEY. Raw keys are not
// accepted in the agent spec (so secrets never live in the AGENTS config).
function resolveAgent(agent?: AgentSpec): {
  model: string;
  baseUrl?: string;
  apiKey?: string;
} {
  const provider = agent?.provider ?? config.llmProvider;
  if (provider !== AIProviderType.AI_SDK) {
    throw new Error(
      `Agentic review requires the '${AIProviderType.AI_SDK}' provider ` +
        `(got '${provider}'). Set LLM_PROVIDER=ai-sdk.`
    );
  }
  const model = agent?.model ?? config.llmModel ?? "";
  if (!model) {
    throw new Error("No model configured for agentic review");
  }

  let apiKey: string | undefined;
  if (agent?.apiKeyEnv) {
    apiKey = process.env[agent.apiKeyEnv];
    if (!apiKey) {
      console.warn(
        `Agent '${agent.id}': apiKeyEnv '${agent.apiKeyEnv}' is not set in the environment.`
      );
    }
  }
  apiKey = apiKey ?? config.llmApiKey;

  return { model, baseUrl: agent?.baseUrl, apiKey };
}

export type RunAgentResult = {
  text: string;
  steps: number;
};

/**
 * Run a tool-using agent loop with the AI SDK (generateText + maxSteps).
 *
 * Additive, used only by the opt-in agentic reviewer. Accepts a per-call
 * AgentSpec so a multi-agent review can run several models, each with its own
 * provider/base URL/API key. Only the `ai-sdk` provider is supported.
 */
export async function runAgent({
  systemPrompt,
  prompt,
  tools,
  maxSteps,
  model,
}: {
  systemPrompt: string;
  prompt: string;
  tools: Record<string, CoreTool>;
  maxSteps?: number;
  model?: AgentSpec;
}): Promise<RunAgentResult> {
  const llm = createAiSdkLanguageModel(resolveAgent(model));
  const result = await generateText({
    model: llm,
    system: systemPrompt,
    prompt,
    tools,
    maxSteps: maxSteps ?? 12,
    temperature: 0,
  });

  return { text: result.text, steps: result.steps?.length ?? 0 };
}

/**
 * Structured (generateObject) call with a chosen agent model. Used by the
 * agentic synthesis step so the judge can run a different model than the
 * explorers. Only the `ai-sdk` provider is supported.
 */
export async function runStructured({
  systemPrompt,
  prompt,
  schema,
  model,
}: {
  systemPrompt: string;
  prompt: string;
  schema: z.ZodObject<any, any>;
  model?: AgentSpec;
}): Promise<any> {
  const llm = createAiSdkLanguageModel(resolveAgent(model));
  const { object } = await generateObject({
    model: llm,
    system: systemPrompt,
    prompt,
    schema,
    temperature: 0,
  });
  return object;
}
