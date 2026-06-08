import { z } from "zod";

/**
 * Zod schema for the structured review result. This mirrors the shape produced
 * by `runReviewPrompt` in src/prompts.ts (PullRequestReview) so the agentic
 * path is a drop-in replacement and feeds the same `submitReview` downstream.
 *
 * Kept in its own module (rather than imported from prompts.ts) so the existing
 * single-shot review path is left completely untouched.
 */
export function buildReviewResultSchema() {
  const commentSchema = z.object({
    file: z.string().describe("The full file path of the relevant file"),
    start_line: z
      .number()
      .describe(
        "The relevant line number, from a '__new hunk__' section, where the comment starts (inclusive). If the comment spans a single line, it should equal 'end_line'."
      ),
    end_line: z
      .number()
      .describe(
        "The relevant line number, from a '__new hunk__' section, where the comment ends (inclusive). If the comment spans a single line, it should equal 'start_line'."
      ),
    content: z
      .string()
      .describe(
        "An actionable comment to enhance, improve or fix the new code introduced in the PR. Use markdown formatting."
      ),
    header: z
      .string()
      .describe(
        "A concise, single-sentence overview of the comment. Focus on the 'what'."
      ),
    highlighted_code: z
      .string()
      .describe(
        "A short code snippet from a '__new hunk__' section that the comment applies to. Include only complete code lines, without line numbers."
      ),
    label: z
      .string()
      .describe(
        "A single descriptive label: 'security', 'possible bug', 'possible issue', 'performance', 'enhancement', 'best practice', 'maintainability', 'readability', or another relevant label."
      ),
    critical: z
      .boolean()
      .describe(
        "True if the comment is critical and the PR should not be merged without addressing it. False otherwise."
      ),
  });

  const reviewSchema = z.object({
    estimated_effort_to_review: z
      .number()
      .min(1)
      .max(5)
      .describe(
        "Estimate, on a scale of 1-5 (inclusive), the time and effort required to review this PR. 1 = short and easy, 5 = long and hard."
      ),
    score: z
      .number()
      .min(0)
      .max(100)
      .describe(
        "Rate this PR on a scale of 0-100 (inclusive), where 0 is the worst possible code and 100 is the highest quality, ready to merge."
      ),
    has_relevant_tests: z
      .boolean()
      .describe("True if the PR includes relevant tests added or updated."),
    security_concerns: z
      .string()
      .describe(
        "Describe any security concerns (e.g. secret exposure, SQL injection, XSS, CSRF). Answer 'No' if there are none."
      ),
  });

  return z.object({
    review: reviewSchema.describe("The full review of the PR"),
    comments: z
      .array(commentSchema)
      .describe(
        "Comments about possible bugs, security concerns, code quality, typos or regressions introduced in this PR."
      ),
  });
}
