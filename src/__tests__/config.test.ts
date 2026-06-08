import { Config, parseAgents, parseAgent } from '../config';
import * as core from '@actions/core';

// Create manual mocks for core functions
const mockGetInput = jest.fn().mockImplementation(() => '');
const mockGetMultilineInput = jest.fn().mockImplementation(() => []);

// Mock the entire module
jest.mock('@actions/core', () => ({
  getInput: (...args: any[]) => mockGetInput(...args),
  getMultilineInput: (...args: any[]) => mockGetMultilineInput(...args),
  info: jest.fn(),
  warning: jest.fn()
}));

describe('Config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    mockGetInput.mockImplementation(() => '');
    mockGetMultilineInput.mockImplementation(() => []);

    // Reset environment
    process.env = { ...originalEnv };
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  test('throws error when GITHUB_TOKEN is not set', () => {
    process.env.GITHUB_TOKEN = '';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';

    expect(() => new Config()).toThrow('GITHUB_TOKEN is not set');
  });

  test('throws error when LLM_API_KEY is not set', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = '';
    process.env.LLM_MODEL = 'test-model';

    expect(() => new Config()).toThrow('LLM_API_KEY is not set');
  });

  test('throws error when LLM_MODEL is not set', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = '';

    expect(() => new Config()).toThrow('LLM_MODEL is not set');
  });

  test('loads style guide rules from action inputs', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';
    process.env.DEBUG = '';

    const styleGuideRules = ['Rule 1', 'Rule 2', 'Rule 3'];
    mockGetMultilineInput.mockImplementation((name) => {
      if (name === 'style_guide_rules') return styleGuideRules;
      return [];
    });

    const config = new Config();
    config.loadInputs();

    expect(config.styleGuideRules).toBe(styleGuideRules.join('\n'));
  });

  test('uses default GitHub URLs when not provided', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';

    const config = new Config();

    expect(config.githubApiUrl).toBe('https://api.github.com');
    expect(config.githubServerUrl).toBe('https://github.com');
  });

  test('loads GitHub Enterprise Server URLs from environment variables', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';
    process.env.GITHUB_API_URL = 'https://github.example.com/api/v3';
    process.env.GITHUB_SERVER_URL = 'https://github.example.com';

    const config = new Config();

    expect(config.githubApiUrl).toBe('https://github.example.com/api/v3');
    expect(config.githubServerUrl).toBe('https://github.example.com');
  });

  test('loads GitHub Enterprise Server URLs from action inputs', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';

    mockGetInput.mockImplementation((name) => {
      if (name === 'github_api_url') return 'https://github.example.com/api/v3';
      if (name === 'github_server_url') return 'https://github.example.com';
      return '';
    });

    const config = new Config();

    expect(config.githubApiUrl).toBe('https://github.example.com/api/v3');
    expect(config.githubServerUrl).toBe('https://github.example.com');
  });

  //   test('skips loading inputs when DEBUG is set', () => {
  //     process.env.GITHUB_TOKEN = 'test-token';
  //     process.env.LLM_API_KEY = 'test-api-key';
  //     process.env.LLM_MODEL = 'test-model';
  //     process.env.DEBUG = 'true';
  //     process.env.STYLE_GUIDE_RULES = 'Debug rule';

  //     const config = new Config();
  //     config.loadInputs();

  //     expect(config.styleGuideRules).toBe('Debug rule');
  //     expect(core.getMultilineInput).not.toHaveBeenCalled();
  //   });

  test('loads LLM_BASE_URL from environment variable', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';
    process.env.LLM_BASE_URL = 'https://openrouter.ai/api/v1';

    const config = new Config();

    expect(config.llmBaseUrl).toBe('https://openrouter.ai/api/v1');
  });

  test('llmBaseUrl is undefined when not set', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';

    const config = new Config();

    expect(config.llmBaseUrl).toBeUndefined();
  });

  test('loads LLM_BASE_URL from action input', () => {
    process.env.GITHUB_TOKEN = 'test-token';
    process.env.LLM_API_KEY = 'test-api-key';
    process.env.LLM_MODEL = 'test-model';

    mockGetInput.mockImplementation((name) => {
      if (name === 'llm_base_url') return 'https://anyscale.com/api/v1';
      return '';
    });

    const config = new Config();

    expect(config.llmBaseUrl).toBe('https://anyscale.com/api/v1');
  });

  describe('agentic review settings', () => {
    beforeEach(() => {
      process.env.GITHUB_TOKEN = 'test-token';
      process.env.LLM_API_KEY = 'test-api-key';
      process.env.LLM_MODEL = 'test-model';
    });

    test('agenticReview defaults to false and agents to empty', () => {
      const config = new Config();
      expect(config.agenticReview).toBe(false);
      expect(config.agents).toEqual([]);
      expect(config.synthesisAgent).toBeUndefined();
      expect(config.agenticMaxSteps).toBe(12);
      expect(config.agenticDiscussionRounds).toBe(2);
    });

    test('agenticReview parses truthy env values', () => {
      process.env.AGENTIC_REVIEW = 'true';
      expect(new Config().agenticReview).toBe(true);

      process.env.AGENTIC_REVIEW = '1';
      expect(new Config().agenticReview).toBe(true);

      process.env.AGENTIC_REVIEW = 'false';
      expect(new Config().agenticReview).toBe(false);
    });

    test('agenticMaxSteps parses positive ints and ignores junk', () => {
      process.env.AGENTIC_MAX_STEPS = '20';
      expect(new Config().agenticMaxSteps).toBe(20);

      process.env.AGENTIC_MAX_STEPS = '-5';
      expect(new Config().agenticMaxSteps).toBe(12);

      process.env.AGENTIC_MAX_STEPS = 'abc';
      expect(new Config().agenticMaxSteps).toBe(12);
      delete process.env.AGENTIC_MAX_STEPS;
    });

    test('parses AGENTS as a comma-separated list (id defaults to model)', () => {
      process.env.AGENTS =
        'anthropic/claude-sonnet-4.5, google/gemini-2.5-pro';
      const config = new Config();
      expect(config.agents).toEqual([
        { id: 'anthropic/claude-sonnet-4.5', model: 'anthropic/claude-sonnet-4.5' },
        { id: 'google/gemini-2.5-pro', model: 'google/gemini-2.5-pro' },
      ]);
      delete process.env.AGENTS;
    });

    test('parses AGENTS as JSON with id, instructions, provider, base URL', () => {
      process.env.AGENTS = JSON.stringify([
        {
          id: 'security',
          model: 'anthropic/claude-sonnet-4.5',
          instructions: 'Focus on security.',
          provider: 'ai-sdk',
          baseUrl: 'https://openrouter.ai/api/v1',
        },
        'google/gemini-2.5-pro',
      ]);
      const config = new Config();
      expect(config.agents).toEqual([
        {
          id: 'security',
          model: 'anthropic/claude-sonnet-4.5',
          instructions: 'Focus on security.',
          provider: 'ai-sdk',
          baseUrl: 'https://openrouter.ai/api/v1',
          apiKey: undefined,
        },
        { id: 'google/gemini-2.5-pro', model: 'google/gemini-2.5-pro' },
      ]);
      delete process.env.AGENTS;
    });

    test('parses SYNTHESIS_AGENT (object and bare string)', () => {
      process.env.SYNTHESIS_AGENT = JSON.stringify({
        id: 'judge',
        model: 'openai/gpt-5',
        instructions: 'Be conservative.',
      });
      const config = new Config();
      expect(config.synthesisAgent).toEqual({
        id: 'judge',
        model: 'openai/gpt-5',
        instructions: 'Be conservative.',
        provider: undefined,
        baseUrl: undefined,
        apiKey: undefined,
      });
      delete process.env.SYNTHESIS_AGENT;
    });

    test('LLM_MODEL and LLM_API_KEY are optional in multi-agent agentic mode', () => {
      process.env.LLM_MODEL = '';
      process.env.LLM_API_KEY = '';
      process.env.AGENTIC_REVIEW = 'true';
      process.env.AGENTS = 'deepseek/deepseek-v4-flash, moonshotai/kimi-k2.6';

      expect(() => new Config()).not.toThrow();
      const config = new Config();
      expect(config.agents.map((a) => a.model)).toEqual([
        'deepseek/deepseek-v4-flash',
        'moonshotai/kimi-k2.6',
      ]);

      delete process.env.AGENTIC_REVIEW;
      delete process.env.AGENTS;
    });

    test('LLM_MODEL still required when agentic review has no AGENTS', () => {
      process.env.LLM_MODEL = '';
      process.env.LLM_API_KEY = 'k';
      process.env.AGENTIC_REVIEW = 'true';

      expect(() => new Config()).toThrow('LLM_MODEL is not set');

      delete process.env.AGENTIC_REVIEW;
    });

    test('parses AGENTIC_DISCUSSION_ROUNDS, ignoring junk (default 2)', () => {
      process.env.AGENTIC_DISCUSSION_ROUNDS = '3';
      expect(new Config().agenticDiscussionRounds).toBe(3);

      process.env.AGENTIC_DISCUSSION_ROUNDS = '0';
      expect(new Config().agenticDiscussionRounds).toBe(2);

      process.env.AGENTIC_DISCUSSION_ROUNDS = 'abc';
      expect(new Config().agenticDiscussionRounds).toBe(2);
      delete process.env.AGENTIC_DISCUSSION_ROUNDS;
    });
  });
});

describe('parseAgents', () => {
  test('returns empty array for unset/blank input', () => {
    expect(parseAgents(undefined)).toEqual([]);
    expect(parseAgents('')).toEqual([]);
    expect(parseAgents('   ')).toEqual([]);
  });

  test('parses comma-separated names with id defaulting to model', () => {
    expect(parseAgents('a, b ,c')).toEqual([
      { id: 'a', model: 'a' },
      { id: 'b', model: 'b' },
      { id: 'c', model: 'c' },
    ]);
  });

  test('parses JSON objects and bare strings', () => {
    expect(parseAgents('[{"id":"x","model":"m","provider":"ai-sdk"}, "y"]')).toEqual([
      {
        id: 'x',
        model: 'm',
        instructions: undefined,
        provider: 'ai-sdk',
        baseUrl: undefined,
        apiKeyEnv: undefined,
      },
      { id: 'y', model: 'y' },
    ]);
  });

  test('parses apiKeyEnv (secret env name, not a raw key)', () => {
    const agents = parseAgents(
      '[{"model":"anthropic/claude-sonnet-4.5","baseUrl":"https://openrouter.ai/api/v1","apiKeyEnv":"OPENROUTER_API_KEY"}]'
    );
    expect(agents[0]).toMatchObject({
      id: 'anthropic/claude-sonnet-4.5',
      model: 'anthropic/claude-sonnet-4.5',
      baseUrl: 'https://openrouter.ai/api/v1',
      apiKeyEnv: 'OPENROUTER_API_KEY',
    });
  });

  test('drops entries without a model and handles bad json', () => {
    expect(parseAgents('[{"id":"no-model"}, "ok"]')).toEqual([
      { id: 'ok', model: 'ok' },
    ]);
    expect(parseAgents('[not valid json')).toEqual([]);
  });
});

describe('parseAgent', () => {
  test('returns undefined for unset/blank', () => {
    expect(parseAgent(undefined)).toBeUndefined();
    expect(parseAgent('   ')).toBeUndefined();
  });

  test('parses a bare model name', () => {
    expect(parseAgent('openai/gpt-5')).toEqual({
      id: 'openai/gpt-5',
      model: 'openai/gpt-5',
    });
  });

  test('parses a JSON object', () => {
    expect(parseAgent('{"id":"judge","model":"m","instructions":"x"}')).toEqual({
      id: 'judge',
      model: 'm',
      instructions: 'x',
      provider: undefined,
      baseUrl: undefined,
      apiKey: undefined,
    });
  });

  test('returns undefined for object without model / bad json', () => {
    expect(parseAgent('{"id":"no-model"}')).toBeUndefined();
    expect(parseAgent('{bad json')).toBeUndefined();
  });
});
