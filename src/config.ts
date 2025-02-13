import { getInput, getMultilineInput } from "@actions/core";

export class Config {
  public llmApiKey: string | undefined;
  public llmModel: string | undefined;
  public githubToken: string | undefined;
  public styleGuideRules: string | undefined;

  constructor() {
    
    this.githubToken = process.env.GITHUB_TOKEN;
    if (!this.githubToken) {
      throw new Error("GITHUB_TOKEN is not set");
    }

    this.llmApiKey = process.env.LLM_API_KEY;
    if (!this.llmApiKey) {
      throw new Error("LLM_API_KEY is not set");
    }

    this.llmModel = process.env.LLM_MODEL || getInput("llm_model");
    if (!this.llmModel?.length) {
      throw new Error("LLM_MODEL is not set");
    }

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
    const styleGuideRules = getMultilineInput('style_guide_rules');
    if (styleGuideRules.length && styleGuideRules[0].trim().length) {
      this.styleGuideRules = styleGuideRules.join("\n");
    }
  }
}

const config = new Config();
config.loadInputs();

export default config;
