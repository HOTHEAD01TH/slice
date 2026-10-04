import { Hono } from "hono";
import {
  LayaNotRunning,
  LAYA_NOT_RUNNING,
  choiceOf,
  confidenceOf,
  isAbstained,
  layaHealth,
  noulOf,
  predict,
  predictBatch,
  scoreOf,
  type Question,
} from "./laya.ts";

const PORT = 3001;
const MIN_CONFIDENCE = 0.6;

const QUEUES = ["billing", "technical", "account", "sales"] as const;
const URGENCY = ["can wait", "soon", "today", "blocking"] as const;

const questions: Record<string, Question> = {
  queue: {
    type: "choice",
    instructions: "Which queue should own this inbox item? Pick one team. Do not draft a reply.",
    criteria: {
      billing: "Invoices, payments, refunds, charges",
      technical: "Bugs, outages, product behavior, integrations",
      account: "Login, access, profile, workspace administration",
      sales: "Demos, new work, upgrades, inbound interest",
    },
  },
  urgency: {
    type: "score",
    instructions: "How urgent is this inbox item?",
    criteria: [...URGENCY],
  },
  churn: {
    type: "noul",
    instructions: "Does the sender threaten to cancel, churn, or leave?",
  },
  refund: {
    type: "noul",
    instructions: "Does the sender ask for a refund?",
  },
};

type Kind = "email" | "ticket" | "lead";

type Scored<T> = { value: T; confidence: number | null };

type Item = {
  id: string;
  kind: Kind;
  from: string;
  subject: string;
  body: string;
  status: "pending" | "unsure" | "routed";
  queue: string | null;
  urgency: (Scored<number> & { level: string }) | null;
  churn: Scored<number> | null;
  refund: Scored<number> | null;
  model: string | null;
  createdAt: string;
};

const items: Item[] = [];

const seeds: Array<{ id: string; kind: Kind; from: string; subject: string; body: string }> = [
  {
    id: "seed-1",
    kind: "email",
    from: "ap@northwind-loom.example",
    subject: "Past due notice",
    body: "Our last invoice is past due. Please point us at the right billing contact. This is the whole note.",
  },
  {
    id: "seed-2",
    kind: "ticket",
    from: "sam@paperlane.example",
    subject: "Export endpoint 500",
    body: "POST /v1/exports returns 500 after the last deploy. The stack trace is in the ticket. This is the whole note.",
  },
  {
    id: "seed-3",
    kind: "email",
    from: "rina@harborline.example",
    subject: "Cannot reset password",
    body: "Cannot sign in. Password reset never lands.",
  },
  {
    id: "seed-4",
    kind: "lead",
    from: "jules@kite-and-co.example",
    subject: "Demo request",
    body: "A studio wants a walkthrough of the inbox product. No other request.",
  },
  {
    id: "seed-5",
    kind: "email",
    from: "ops@lantern-row.example",
    subject: "Charge question and a broken webhook",
    body: "Two issues in one note: a charge question and a webhook that retries.",
  },
  {
    id: "seed-6",
    kind: "ticket",
    from: "dev@millhouse.example",
    subject: "SSO metadata rotation",
    body: "SAML certificate rotation. New metadata is attached. No other request.",
  },
  {
    id: "seed-7",
    kind: "lead",
    from: "ada@fieldnote.example",
    subject: "Maybe later",
    body: "Saw the product at a meetup. Not sure what we need yet. We may leave the trial if this stays unclear.",
  },
  {
    id: "seed-8",
    kind: "email",
    from: "billing@cedar-post.example",
    subject: "Seat refund request",
    body: "Please refund the seats we removed. If this is not fixed we will cancel the workspace.",
  },
];

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function str(v: unknown, max: number): string | null {
  if (typeof v !== "string") return null;
  const t = v.trim();
  if (!t || t.length > max) return null;
  return t;
}

function blank(input: { id: string; kind: Kind; from: string; subject: string; body: string }): Item {
  return {
    id: input.id,
    kind: input.kind,
    from: input.from,
    subject: input.subject,
    body: input.body,
    status: "pending",
    queue: null,
    urgency: null,
    churn: null,
    refund: null,
    model: null,
    createdAt: new Date().toISOString(),
  };
}

function stateOf(item: Item) {
  return { kind: item.kind, from: item.from, subject: item.subject, body: item.body };
}

function applyResult(item: Item, result: Awaited<ReturnType<typeof predict>>): Item {
  const queueAns = result.answers.queue;
  const urgencyAns = result.answers.urgency;
  const churnAns = result.answers.churn;
  const refundAns = result.answers.refund;
  const queueName = choiceOf(queueAns, QUEUES);
  const urgency = scoreOf(urgencyAns, URGENCY);
  const churn = noulOf(churnAns);
  const refund = noulOf(refundAns);
  const queueAbs = isAbstained(queueAns, MIN_CONFIDENCE) || queueName === null;
  const urgencyAbs = isAbstained(urgencyAns, MIN_CONFIDENCE) || urgency === null;
  const churnAbs = isAbstained(churnAns, MIN_CONFIDENCE) || churn === null;
  const refundAbs = isAbstained(refundAns, MIN_CONFIDENCE) || refund === null;
  const unsure = queueAbs || urgencyAbs || churnAbs || refundAbs;
  return {
    ...item,
    status: unsure ? "unsure" : "routed",
    queue: unsure ? null : queueName,
    urgency: urgencyAbs || !urgency ? null : { value: urgency.value, level: urgency.level, confidence: confidenceOf(urgencyAns) },
    churn: churnAbs || churn === null ? null : { value: churn, confidence: confidenceOf(churnAns) },
    refund: refundAbs || refund === null ? null : { value: refund, confidence: confidenceOf(refundAns) },
    model: result.model,
  };
}

let tail: Promise<void> = Promise.resolve();
function enqueue<T>(fn: () => Promise<T>): Promise<T> {
  const run = tail.then(fn, fn);
  tail = run.then(() => undefined, () => undefined);
  return run;
}

async function scorePending(): Promise<void> {
  const pending = items.filter((item) => item.status === "pending");
  if (pending.length === 0) return;
  const results = await predictBatch(pending.map(stateOf), questions, MIN_CONFIDENCE);
  pending.forEach((item, i) => {
    const next = applyResult(item, results[i]!);
    const idx = items.findIndex((row) => row.id === item.id);
    if (idx >= 0) items[idx] = next;
  });
}

const KINDS = new Set<Kind>(["email", "ticket", "lead"]);

const app = new Hono();
const page = await Bun.file(new URL("../public/index.html", import.meta.url)).text();

app.get("/", (c) => c.html(page));

app.get("/api/health", async (c) => {
  const health = await layaHealth();
  if (!health.ok) return c.json({ ok: false, error: LAYA_NOT_RUNNING }, 503);
  return c.json({ ok: true, model_loaded: health.modelLoaded, laya_version: health.version });
});

app.get("/api/items", (c) => c.json({ items, min_confidence: MIN_CONFIDENCE }));

app.post("/api/score", async (c) => {
  try {
    await enqueue(scorePending);
    return c.json({ items });
  } catch (err) {
    if (err instanceof LayaNotRunning) return c.json({ error: LAYA_NOT_RUNNING }, 503);
    throw err;
  }
});

app.post("/api/ingest", async (c) => {
  const raw: unknown = await c.req.json().catch(() => null);
  if (!isRecord(raw)) return c.json({ error: "expected a JSON object" }, 400);
  const kind = raw.kind;
  if (typeof kind !== "string" || !KINDS.has(kind as Kind)) {
    return c.json({ error: "kind must be email, ticket, or lead" }, 400);
  }
  const from = str(raw.from, 200);
  const subject = str(raw.subject, 300);
  const body = str(raw.body, 20000);
  if (!from || !subject || !body) return c.json({ error: "from, subject, and body are required" }, 400);
  const item = blank({ id: crypto.randomUUID(), kind: kind as Kind, from, subject, body });
  items.push(item);
  try {
    await enqueue(async () => {
      const result = await predict(stateOf(item), questions, MIN_CONFIDENCE);
      const idx = items.findIndex((row) => row.id === item.id);
      if (idx >= 0) items[idx] = applyResult(items[idx]!, result);
    });
    return c.json({ item: items.find((row) => row.id === item.id) }, 201);
  } catch (err) {
    if (err instanceof LayaNotRunning) return c.json({ error: LAYA_NOT_RUNNING }, 503);
    throw err;
  }
});

app.onError((err, c) => {
  console.error(err);
  if (err instanceof LayaNotRunning || (err instanceof Error && err.message === LAYA_NOT_RUNNING)) {
    return c.json({ error: LAYA_NOT_RUNNING }, 503);
  }
  return c.json({ error: "internal error" }, 500);
});

if (import.meta.main) {
  for (const seed of seeds) items.push(blank(seed));
  Bun.serve({ port: PORT, fetch: app.fetch });
  console.log(`slice listening on http://localhost:${PORT}`);
}
