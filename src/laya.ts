/**
 * HTTP client for the local Laya sidecar (POST /v1/systemone).
 * A down sidecar or an unloaded model throws. This module never fills in scores.
 * Abstention is min_confidence / answer abstention flags only.
 */

export const LAYA_NOT_RUNNING = "Laya not running";

export class LayaNotRunning extends Error {
  readonly detail?: string;
  constructor(detail?: string) {
    super(LAYA_NOT_RUNNING);
    this.name = "LayaNotRunning";
    this.detail = detail;
  }
}

export type ChoiceQuestion = {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
};

export type ScoreQuestion = {
  type: "score";
  instructions: string;
  criteria: string[];
};

export type NoulQuestion = {
  type: "noul";
  instructions: string;
};

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion;

export type RawAnswer = {
  type?: string;
  choice?: string;
  score?: number | string;
  legend?: unknown;
  noul?: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  answer_confidence?: number;
  low_confidence?: boolean;
  abstention?: string;
  abstention_threshold?: number;
};

export type PredictResult = {
  model: string;
  answers: Record<string, RawAnswer>;
  usage: { input_tokens: number; output_tokens: number } | null;
};

const base = () => (process.env.LAYA_URL ?? "http://127.0.0.1:8101").replace(/\/$/, "");

async function post(path: string, body: unknown): Promise<{ status: number; data: unknown }> {
  let res: Response;
  try {
    res = await fetch(base() + path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15 * 60 * 1000),
    });
  } catch (err) {
    throw new LayaNotRunning(err instanceof Error ? err.message : String(err));
  }
  const data: unknown = await res.json().catch(() => null);
  return { status: res.status, data };
}

function fail(status: number, data: unknown): never {
  if (status === 503 || status === 502 || status === 500) throw new LayaNotRunning();
  let message = "laya request failed";
  if (typeof data === "object" && data !== null) {
    const err = (data as { error?: unknown; detail?: unknown }).error ?? (data as { detail?: unknown }).detail;
    if (typeof err === "string" && err.trim()) message = err;
  }
  if (message === LAYA_NOT_RUNNING) throw new LayaNotRunning();
  throw new Error(message);
}

function asResult(data: unknown, questions: Record<string, Question>): PredictResult {
  if (typeof data !== "object" || data === null) throw new LayaNotRunning("empty body");
  const row = data as Record<string, unknown>;
  if (row.source === "mock" || row.fake === true || row.model === "mock" || row.model === "fake") {
    throw new LayaNotRunning("fake payload");
  }
  const answers = row.answers;
  if (typeof answers !== "object" || answers === null || Array.isArray(answers)) {
    throw new LayaNotRunning("no answers");
  }
  for (const name of Object.keys(questions)) {
    if (!(name in answers)) throw new LayaNotRunning("missing answer");
  }
  const usage = row.usage;
  let parsed: PredictResult["usage"] = null;
  if (typeof usage === "object" && usage !== null) {
    const u = usage as Record<string, unknown>;
    if (typeof u.input_tokens === "number" && typeof u.output_tokens === "number") {
      parsed = { input_tokens: u.input_tokens, output_tokens: u.output_tokens };
    }
  }
  return {
    model: typeof row.model === "string" && row.model !== "mock" ? row.model : "laya",
    answers: answers as Record<string, RawAnswer>,
    usage: parsed,
  };
}

export async function predict(
  state: unknown,
  questions: Record<string, Question>,
  minConfidence: number,
): Promise<PredictResult> {
  const { status, data } = await post("/v1/systemone", {
    state,
    questions,
    min_confidence: minConfidence,
  });
  if (status !== 200) fail(status, data);
  return asResult(data, questions);
}

export async function predictBatch(
  states: unknown[],
  questions: Record<string, Question>,
  minConfidence: number,
): Promise<PredictResult[]> {
  if (states.length === 0) return [];
  const { status, data } = await post("/v1/systemone/batch", {
    states,
    questions,
    min_confidence: minConfidence,
  });
  if (status !== 200) fail(status, data);
  if (typeof data !== "object" || data === null || !Array.isArray((data as { results?: unknown }).results)) {
    throw new LayaNotRunning("bad batch");
  }
  const results = (data as { results: unknown[] }).results;
  if (results.length !== states.length) throw new LayaNotRunning("bad batch");
  return results.map((row) => asResult(row, questions));
}

export async function layaHealth(): Promise<
  { ok: true; modelLoaded: boolean; version: string | null } | { ok: false }
> {
  try {
    const res = await fetch(base() + "/health", { signal: AbortSignal.timeout(3000) });
    const data = (await res.json().catch(() => null)) as {
      model_loaded?: boolean;
      laya_version?: string;
      error?: string;
    } | null;
    if (!res.ok || !data || data.error) return { ok: false };
    return {
      ok: true,
      modelLoaded: Boolean(data.model_loaded),
      version: typeof data.laya_version === "string" ? data.laya_version : null,
    };
  } catch {
    return { ok: false };
  }
}

/** True when min_confidence abstained, the answer is missing, or the gate did not pass. */
export function isAbstained(answer: RawAnswer | undefined, minConfidence: number): boolean {
  if (!answer) return true;
  if (answer.low_confidence === true) return true;
  if (answer.abstention === "abstained" || answer.abstention === "unevaluated") return true;
  if (answer.abstention === "passed") return false;
  const ac = answer.answer_confidence;
  if (typeof ac === "number" && Number.isFinite(ac)) return ac < minConfidence;
  return true;
}

export function choiceOf(answer: RawAnswer | undefined, allowed: readonly string[]): string | null {
  if (!answer || typeof answer.choice !== "string") return null;
  return allowed.includes(answer.choice) ? answer.choice : null;
}

export function noulOf(answer: RawAnswer | undefined): number | null {
  if (!answer || typeof answer.noul !== "number" || !Number.isFinite(answer.noul)) return null;
  return Math.min(1, Math.max(0, answer.noul));
}

export function scoreOf(
  answer: RawAnswer | undefined,
  levels: readonly string[],
): { value: number; level: string } | null {
  if (!answer || levels.length === 0) return null;
  const probs = answer.probabilities;
  if (probs && typeof probs === "object") {
    let best: string | null = null;
    let bestP = -1;
    for (const level of levels) {
      const p = probs[level];
      if (typeof p === "number" && Number.isFinite(p) && p > bestP) {
        best = level;
        bestP = p;
      }
    }
    if (best) {
      const raw = typeof answer.score === "number" && Number.isFinite(answer.score) ? answer.score : levels.indexOf(best);
      return { value: raw, level: best };
    }
  }
  if (typeof answer.score === "string" && levels.includes(answer.score)) {
    return { value: levels.indexOf(answer.score), level: answer.score };
  }
  if (typeof answer.score === "number" && Number.isFinite(answer.score)) {
    const idx = Math.max(0, Math.min(levels.length - 1, Math.round(answer.score)));
    const level = levels[idx];
    if (!level) return null;
    return { value: answer.score, level };
  }
  return null;
}

export function confidenceOf(answer: RawAnswer | undefined): number | null {
  if (!answer) return null;
  if (typeof answer.answer_confidence === "number" && Number.isFinite(answer.answer_confidence)) {
    return answer.answer_confidence;
  }
  if (typeof answer.confidence === "number" && Number.isFinite(answer.confidence)) return answer.confidence;
  return null;
}
