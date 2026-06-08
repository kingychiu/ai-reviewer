import { runAgent, runStructured, AIProviderType } from "../ai";
import { generateText, generateObject } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { z } from "zod";

jest.mock("ai", () => ({
  __esModule: true,
  generateText: jest.fn(async () => ({
    text: "agent-output",
    steps: [{}, {}, {}],
  })),
  generateObject: jest.fn(async () => ({ object: { ok: true } })),
  tool: (def: any) => def,
}));

jest.mock("@ai-sdk/openai", () => ({
  __esModule: true,
  createOpenAI: jest.fn(() => (name: string) => ({ id: name })),
}));
jest.mock("@ai-sdk/anthropic", () => ({
  __esModule: true,
  createAnthropic: jest.fn(() => (name: string) => ({ id: name })),
}));
jest.mock("@ai-sdk/google", () => ({
  __esModule: true,
  createGoogleGenerativeAI: jest.fn(() => (name: string) => ({ id: name })),
}));

const mockGenerateText = generateText as jest.Mock;
const mockGenerateObject = generateObject as jest.Mock;
const mockCreateOpenAI = createOpenAI as jest.Mock;

const agent = (over: Record<string, any> = {}) => ({
  id: over.id ?? over.model ?? "gpt-4o-mini",
  model: "gpt-4o-mini",
  provider: AIProviderType.AI_SDK,
  ...over,
});

beforeEach(() => jest.clearAllMocks());

describe("runAgent", () => {
  test("runs generateText for a whitelisted model and returns text + step count", async () => {
    const result = await runAgent({
      systemPrompt: "sys",
      prompt: "user",
      tools: {},
      model: agent(),
    });

    expect(result).toEqual({ text: "agent-output", steps: 3 });
    expect(mockGenerateText).toHaveBeenCalledTimes(1);
    expect(mockGenerateText).toHaveBeenCalledWith(
      expect.objectContaining({ system: "sys", prompt: "user", maxSteps: 12 })
    );
  });

  test("honors a custom maxSteps", async () => {
    await runAgent({
      systemPrompt: "s",
      prompt: "p",
      tools: {},
      maxSteps: 5,
      model: agent(),
    });
    expect(mockGenerateText.mock.calls[0][0].maxSteps).toBe(5);
  });

  test("routes unknown models through OpenAI-compatible client when baseUrl is set", async () => {
    await runAgent({
      systemPrompt: "s",
      prompt: "p",
      tools: {},
      model: agent({
        id: "claude-or",
        model: "anthropic/claude-sonnet-4.5",
        baseUrl: "https://openrouter.ai/api/v1",
        apiKey: "or-key",
      }),
    });

    expect(mockCreateOpenAI).toHaveBeenCalledWith(
      expect.objectContaining({
        apiKey: "or-key",
        baseURL: "https://openrouter.ai/api/v1",
      })
    );
  });

  test("rejects non ai-sdk providers", async () => {
    await expect(
      runAgent({
        systemPrompt: "s",
        prompt: "p",
        tools: {},
        model: agent({ id: "x", model: "x", provider: AIProviderType.SAP_AI_SDK }),
      })
    ).rejects.toThrow(/requires the 'ai-sdk' provider/);
    expect(mockGenerateText).not.toHaveBeenCalled();
  });

  test("rejects unknown models when no base URL is configured", async () => {
    await expect(
      runAgent({
        systemPrompt: "s",
        prompt: "p",
        tools: {},
        model: agent({ id: "u", model: "totally-unknown-model" }),
      })
    ).rejects.toThrow(/Unknown LLM model/);
  });
});

describe("runStructured", () => {
  test("runs generateObject with the chosen model and returns the object", async () => {
    const schema = z.object({ ok: z.boolean() });
    const result = await runStructured({
      systemPrompt: "sys",
      prompt: "user",
      schema,
      model: agent(),
    });

    expect(result).toEqual({ ok: true });
    expect(mockGenerateObject).toHaveBeenCalledWith(
      expect.objectContaining({ system: "sys", prompt: "user", schema })
    );
  });

  test("rejects non ai-sdk providers", async () => {
    await expect(
      runStructured({
        systemPrompt: "s",
        prompt: "p",
        schema: z.object({}),
        model: agent({ id: "x", model: "x", provider: AIProviderType.SAP_AI_SDK }),
      })
    ).rejects.toThrow(/requires the 'ai-sdk' provider/);
  });
});
