import {
  runAgenticReview,
  resolveExploreAgents,
  resolveSynthesisAgent,
} from "../agentic/review";
import { runAgent, runStructured } from "../ai";
import type { PullRequestReview } from "../prompts";
import config from "../config";

jest.mock("../ai", () => ({
  __esModule: true,
  runAgent: jest.fn(),
  runStructured: jest.fn(),
  AIProviderType: { AI_SDK: "ai-sdk", SAP_AI_SDK: "sap-ai-sdk" },
}));

jest.mock("../diff", () => ({
  __esModule: true,
  generateFileCodeDiff: jest.fn(() => "DIFF-BLOCK"),
}));

jest.mock("@actions/core", () => ({
  info: jest.fn(),
  warning: jest.fn(),
}));

jest.mock("../config", () => ({
  __esModule: true,
  default: {
    llmModel: "default-model",
    agents: [],
    synthesisAgent: undefined,
    llmProvider: "ai-sdk",
    llmBaseUrl: undefined,
    llmApiKey: "key",
    reviewMode: "single",
    agenticMaxSteps: 12,
    agenticDiscussionRounds: 1,
    styleGuideRules: "",
  },
}));

const mockRunAgent = runAgent as jest.Mock;
const mockRunStructured = runStructured as jest.Mock;

const sampleReview: PullRequestReview = {
  review: {
    estimated_effort_to_review: 2,
    score: 80,
    has_relevant_tests: true,
    security_concerns: "No",
  },
  comments: [
    {
      file: "a.ts",
      start_line: 1,
      end_line: 1,
      content: "issue",
      header: "h",
      highlighted_code: "code",
      label: "bug",
      critical: true,
    },
  ],
};

const pr = {
  prTitle: "Test PR",
  prDescription: "desc",
  prSummary: "summary",
  files: [{ filename: "a.ts" } as any],
  repoRoot: "/tmp/fake-root",
};

beforeEach(() => {
  jest.clearAllMocks();
  config.llmModel = "default-model";
  (config as any).agents = [];
  (config as any).synthesisAgent = undefined;
  config.llmProvider = "ai-sdk";
  config.llmBaseUrl = undefined as any;
  config.llmApiKey = "key";
  (config as any).reviewMode = "single";
  (config as any).agenticMaxSteps = 12;
  (config as any).agenticDiscussionRounds = 1;
  config.styleGuideRules = "";
});

describe("agent resolution", () => {
  test("explore agents fall back to the single llmModel when empty", () => {
    expect(resolveExploreAgents()).toEqual([
      {
        id: "default-model",
        model: "default-model",
        provider: "ai-sdk",
        baseUrl: undefined,
        apiKey: "key",
      },
    ]);
  });

  test("explore agents use AGENTS when set", () => {
    (config as any).agents = [
      { id: "security", model: "anthropic/claude" },
      { id: "perf", model: "google/gemini" },
    ];
    expect(resolveExploreAgents().map((a) => a.id)).toEqual([
      "security",
      "perf",
    ]);
  });

  test("synthesis agent defaults to LLM_MODEL, not an explorer", () => {
    // Panel is m1, but synthesis falls back to the base model.
    (config as any).agents = [{ id: "a1", model: "m1" }];
    expect(resolveSynthesisAgent().id).toBe("default-model");
    expect(resolveSynthesisAgent().model).toBe("default-model");
  });

  test("synthesis agent uses explicit config when set", () => {
    (config as any).agents = [{ id: "a1", model: "m1" }];
    (config as any).synthesisAgent = { id: "judge", model: "j" };
    expect(resolveSynthesisAgent().id).toBe("judge");
  });
});

describe("runAgenticReview - single agent", () => {
  test("explores once then synthesizes into the review shape", async () => {
    mockRunAgent.mockResolvedValue({ text: "notes-from-A", steps: 3 });
    mockRunStructured.mockResolvedValue(sampleReview);

    const result = await runAgenticReview(pr);

    expect(mockRunAgent).toHaveBeenCalledTimes(1);
    expect(mockRunAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        maxSteps: 12,
        model: expect.objectContaining({ model: "default-model" }),
      })
    );
    expect(mockRunStructured).toHaveBeenCalledTimes(1);
    expect(mockRunStructured.mock.calls[0][0].prompt).toContain("notes-from-A");
    expect(result).toEqual(sampleReview);
  });
});

describe("runAgenticReview - multi agent, single mode", () => {
  beforeEach(() => {
    (config as any).agents = [
      { id: "sec", model: "m1" },
      { id: "perf", model: "m2" },
    ];
  });

  test("explores each agent concurrently and synthesizes all notes", async () => {
    mockRunAgent
      .mockResolvedValueOnce({ text: "notes-A", steps: 1 })
      .mockResolvedValueOnce({ text: "notes-B", steps: 1 });
    mockRunStructured.mockResolvedValue(sampleReview);

    const result = await runAgenticReview(pr);

    expect(mockRunAgent).toHaveBeenCalledTimes(2);
    const synthPrompt = mockRunStructured.mock.calls[0][0].prompt;
    expect(synthPrompt).toContain("notes-A");
    expect(synthPrompt).toContain("notes-B");
    expect(synthPrompt).toContain("sec");
    expect(synthPrompt).toContain("perf");
    expect(result).toEqual(sampleReview);
  });

  test("does NOT discuss in single mode", async () => {
    mockRunAgent.mockResolvedValue({ text: "notes", steps: 1 });
    mockRunStructured.mockResolvedValue(sampleReview);

    await runAgenticReview(pr);

    expect(mockRunAgent).toHaveBeenCalledTimes(2); // 2 explore, no discussion
  });
});

describe("runAgenticReview - discussion mode", () => {
  beforeEach(() => {
    (config as any).agents = [
      { id: "sec", model: "m1" },
      { id: "perf", model: "m2" },
    ];
    (config as any).reviewMode = "discussion";
  });

  test("each agent discusses the others (1 round), then synthesizes", async () => {
    mockRunAgent
      .mockResolvedValueOnce({ text: "explore-A", steps: 1 }) // explore sec
      .mockResolvedValueOnce({ text: "explore-B", steps: 1 }) // explore perf
      .mockResolvedValueOnce({ text: "discuss-A", steps: 2 }) // sec discusses
      .mockResolvedValueOnce({ text: "discuss-B", steps: 2 }); // perf discusses
    mockRunStructured.mockResolvedValue(sampleReview);

    await runAgenticReview(pr);

    // 2 explore + 2 discussion (one per agent) = 4 agent calls.
    expect(mockRunAgent).toHaveBeenCalledTimes(4);

    // 'sec' discussion call should see perf's findings but be its own model.
    const secDiscuss = mockRunAgent.mock.calls[2][0];
    expect(secDiscuss.model).toEqual(expect.objectContaining({ id: "sec" }));
    expect(secDiscuss.prompt).toContain("explore-B"); // sees the other's findings
    expect(secDiscuss.prompt).toContain("explore-A"); // and its own previous

    // Synthesis sees the post-discussion notes.
    const synthPrompt = mockRunStructured.mock.calls[0][0].prompt;
    expect(synthPrompt).toContain("discuss-A");
    expect(synthPrompt).toContain("discuss-B");
  });

  test("honors AGENTIC_DISCUSSION_ROUNDS", async () => {
    (config as any).agenticDiscussionRounds = 2;
    mockRunAgent.mockResolvedValue({ text: "notes", steps: 1 });
    mockRunStructured.mockResolvedValue(sampleReview);

    await runAgenticReview(pr);

    // 2 explore + 2 agents * 2 rounds = 6 agent calls.
    expect(mockRunAgent).toHaveBeenCalledTimes(6);
  });

  test("a single agent does not discuss (no peer)", async () => {
    (config as any).agents = [{ id: "solo", model: "m1" }];
    mockRunAgent.mockResolvedValue({ text: "notes", steps: 1 });
    mockRunStructured.mockResolvedValue(sampleReview);

    await runAgenticReview(pr);

    expect(mockRunAgent).toHaveBeenCalledTimes(1); // explore only
  });

  test("an agent's discussion failure keeps its prior position", async () => {
    mockRunAgent
      .mockResolvedValueOnce({ text: "explore-A", steps: 1 })
      .mockResolvedValueOnce({ text: "explore-B", steps: 1 })
      .mockRejectedValueOnce(new Error("discuss boom")) // sec discussion fails
      .mockResolvedValueOnce({ text: "discuss-B", steps: 2 }); // perf discussion ok
    mockRunStructured.mockResolvedValue(sampleReview);

    const result = await runAgenticReview(pr);

    const synthPrompt = mockRunStructured.mock.calls[0][0].prompt;
    expect(synthPrompt).toContain("explore-A"); // sec fell back to its explore note
    expect(synthPrompt).toContain("discuss-B"); // perf updated
    expect(result).toEqual(sampleReview);
  });
});

describe("runAgenticReview - resilience", () => {
  test("one explorer failing still synthesizes from the survivor", async () => {
    (config as any).agents = [
      { id: "sec", model: "m1" },
      { id: "perf", model: "m2" },
    ];
    mockRunAgent
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce({ text: "notes-B", steps: 1 });
    mockRunStructured.mockResolvedValue(sampleReview);

    const result = await runAgenticReview(pr);

    expect(mockRunStructured).toHaveBeenCalledTimes(1);
    const synthPrompt = mockRunStructured.mock.calls[0][0].prompt;
    expect(synthPrompt).toContain("notes-B");
    expect(synthPrompt).not.toContain("notes-A");
    expect(result).toEqual(sampleReview);
  });

  test("all explorers failing returns an empty review without synthesizing", async () => {
    (config as any).agents = [
      { id: "sec", model: "m1" },
      { id: "perf", model: "m2" },
    ];
    mockRunAgent.mockRejectedValue(new Error("boom"));

    const result = await runAgenticReview(pr);

    expect(mockRunStructured).not.toHaveBeenCalled();
    expect(result.comments).toEqual([]);
    expect(result.review.score).toBe(0);
  });

  test("no configured model returns an empty review", async () => {
    config.llmModel = "" as any;
    (config as any).agents = [];

    const result = await runAgenticReview(pr);

    expect(mockRunAgent).not.toHaveBeenCalled();
    expect(mockRunStructured).not.toHaveBeenCalled();
    expect(result.comments).toEqual([]);
  });
});
