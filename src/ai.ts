import config from "./config";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { generateObject, generateText } from "ai";
import { info } from "@actions/core";
import { z } from "zod";

const LLM_MODELS = [
  // Anthropic
  {
    name: "claude-3-5-sonnet-20240620",
    createAi: createAnthropic,
  },
  {
    name: "claude-3-5-sonnet-20241022",
    createAi: createAnthropic,
  },
  // OpenAI
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
    name: "gemini-1.5-flash-8b",
    createAi: createGoogleGenerativeAI,
  },
  {
    name: "gemini-1.5-pro",
    createAi: createGoogleGenerativeAI,
  },
  // Google experimental models https://ai.google.dev/gemini-api/docs/models/experimental-models
  {
    name: "gemini-2.0-pro-exp-02-05",
    createAi: createGoogleGenerativeAI,
  },
  {
    name: "gemini-2.0-flash-thinking-exp-01-21",
    createAi: createGoogleGenerativeAI,
  },
];

async function runGeminiThinkingPrompt({
  prompt,
  systemPrompt,
  schema,
  model,
  llm,
  retryCount = 0,
  previousErrors = [],
}: {
  prompt: string;
  systemPrompt?: string;
  schema: z.ZodObject<any, any>;
  model: typeof LLM_MODELS[number];
  llm: any;
  retryCount?: number;
  previousErrors?: { error: string; response: string }[];
}) {
  const MAX_RETRIES = 5;
  if (retryCount >= MAX_RETRIES) {
    throw new Error(
      `Failed to parse AI response as JSON after ${MAX_RETRIES} attempts. Latest error: ${previousErrors[previousErrors.length - 1]?.error
      }`
    );
  }

  const schemaDescription = JSON.stringify(schema.shape, null, 2);
  let enhancedSystemPrompt = `${systemPrompt || ""}
Please format your response as a valid JSON object matching this schema:
${schemaDescription}

IMPORTANT: Your response must be a single, valid JSON object that matches the schema exactly.`;

  if (previousErrors.length > 0) {
    enhancedSystemPrompt += `\n\nPrevious attempts failed with the following errors:
${previousErrors
        .map(
          (attempt, i) => `
Attempt ${i + 1}:
Error: ${attempt.error}
Response: ${attempt.response}
`
        )
        .join("\n")}

Please fix these issues and ensure the response is valid JSON.`;
  }

  const { text, usage } = await generateText({
    model: llm(model.name),
    prompt,
    system: enhancedSystemPrompt,
  });

  if (process.env.DEBUG) {
    info(`usage: \n${JSON.stringify(usage, null, 2)}`);
  }

  try {
    const jsonResponse = JSON.parse(text);
    return schema.parse(jsonResponse);
  } catch (err) {
    const error = err as Error;
    info(`Failed to parse AI response as JSON: ${error.message}`);

    return runGeminiThinkingPrompt({
      prompt,
      systemPrompt,
      schema,
      model,
      llm,
      retryCount: retryCount + 1,
      previousErrors: [...previousErrors, { error: error.message, response: text }],
    });
  }
}

export async function runPrompt({
  prompt,
  systemPrompt,
  schema,
}: {
  prompt: string;
  systemPrompt?: string;
  schema: z.ZodObject<any, any>;
}) {
  const model = LLM_MODELS.find((m) => m.name === config.llmModel);
  if (!model) {
    throw new Error(`Unknown LLM model: ${config.llmModel}`);
  }

  const llm = model.createAi({ apiKey: config.llmApiKey });

  if (model.name.includes("gemini") && model.name.includes("thinking")) {
    // Gemini thinking models doesn't support generate object (json output)
    // https://ai.google.dev/gemini-api/docs/thinking#limitations
    return runGeminiThinkingPrompt({ prompt, systemPrompt, schema, model, llm });
  }

  // Generate object (json output)
  const { object, usage } = await generateObject({
    model: llm(model.name),
    prompt,
    system: systemPrompt,
    schema,
  });

  if (process.env.DEBUG) {
    info(`usage: \n${JSON.stringify(usage, null, 2)}`);
  }

  return object;
}
