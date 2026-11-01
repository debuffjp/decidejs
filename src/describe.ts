// Label descriptions written once by an LLM. Zero-shot decide() with a one-sentence description per label, in the
// user's voice, was 7 to 10 points more accurate than label names on two intent datasets (eval/NOTES-accuracy.md).

export interface DescribeOptions {
  /** What the texts are, so descriptions match them. For example "messages to a bank's support chat". */
  context?: string;
}

/** The prompt `describeLabels()` sends. Exported so you can run it yourself and pass the reply to `parseDescriptions()`. */
export function describePrompt(labels: readonly string[], { context }: DescribeOptions = {}): string {
  return [
    `Write one short sentence for each label below, phrased as a typical ${context ?? "message"} that belongs under it, in the voice of the person writing it.`,
    "Don't repeat the label's words if a more natural phrasing exists. Reply with only a JSON object mapping each label to its sentence.",
    "",
    ...labels.map((l) => `- ${l}`),
  ].join("\n");
}

/** Reads an LLM's reply to `describePrompt()`. Labels the reply missed, or answered with anything but a string, are left out. */
export function parseDescriptions<const L extends string>(labels: readonly L[], reply: string): Partial<Record<L, string>> {
  const json = reply.slice(reply.indexOf("{"), reply.lastIndexOf("}") + 1);
  let parsed: Record<string, unknown> = {};
  try { parsed = JSON.parse(json); } catch {}
  return Object.fromEntries(labels.filter((l) => typeof parsed[l] === "string" && (parsed[l] as string).trim()).map((l) => [l, (parsed[l] as string).trim()])) as Partial<Record<L, string>>;
}

/**
 * Asks an LLM once for a one-sentence description of each label and returns a record `decide()` takes as its
 * options. A label the LLM skipped gets its own name, so every label is present. Save the result and reuse it;
 * there's no need to call this on every load.
 */
export async function describeLabels<const L extends string>(labels: readonly L[], complete: (prompt: string) => string | Promise<string>, options?: DescribeOptions): Promise<Record<L, string>> {
  const got = parseDescriptions(labels, await complete(describePrompt(labels, options)));
  return Object.fromEntries(labels.map((l) => [l, got[l] ?? l.replace(/[_-]+/g, " ")])) as Record<L, string>;
}
