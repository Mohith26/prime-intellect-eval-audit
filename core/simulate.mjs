// Seeded simulator for a document-grounded QA environment evaluated by 14 models
// (4 rollouts per example). It plants the four failure classes described in Prime
// Intellect's "Learnings from Auditing Public Evaluations" post: wrong answer keys,
// brittle graders that reject correct answers, ambiguous prompts with two defensible
// readings, and rollouts that fetch the published solutions from the web.
// The audit (core/audit.mjs) never reads the hidden truth; tests use it to score recall.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)];

export const DOMAINS = [
  'fiscaldata.treasury.gov',
  'govinfo.gov',
  'fraser.stlouisfed.org',
  'en.wikipedia.org',
  'github.com',
  'huggingface.co',
  'modelscope.cn',
];

export const MODELS = [
  { id: 'google/gemini-3.1-pro-preview', provider: 'google', theta: 1.9, verbose: 0.15, leak: 0 },
  { id: 'openai/gpt-5.4', provider: 'openai', theta: 1.7, verbose: 0.1, leak: 0, effort: 'medium' },
  { id: 'anthropic/claude-opus-4.5', provider: 'anthropic', theta: 1.55, verbose: 0.75, leak: 0 },
  { id: 'anthropic/claude-sonnet-4.5', provider: 'anthropic', theta: 1.15, verbose: 0.7, leak: 0 },
  { id: 'qwen/qwen3.5-397b-a17b', provider: 'qwen', theta: 1.05, verbose: 0.25, leak: 0 },
  { id: 'moonshotai/kimi-k2.6', provider: 'moonshot', theta: 0.95, verbose: 0.6, leak: 0 },
  { id: 'z-ai/glm-5.1', provider: 'zai', theta: 0.85, verbose: 0.65, leak: 0 },
  { id: 'deepseek/deepseek-v4-flash', provider: 'deepseek', theta: 0.55, verbose: 0.2, leak: 0.5 },
  { id: 'minimax/minimax-m2.5', provider: 'minimax', theta: 0.45, verbose: 0.3, leak: 0.22 },
  { id: 'xiaomi/mimo-v2.5-pro', provider: 'xiaomi', theta: 0.25, verbose: 0.2, leak: 0 },
  { id: 'openai/gpt-5-mini', provider: 'openai', theta: -0.05, verbose: 0.1, leak: 0 },
  { id: 'Qwen/Qwen3-32B', provider: 'qwen', theta: -0.6, verbose: 0.2, leak: 0 },
  { id: 'google/gemma-4-26B-A4B-it', provider: 'google', theta: -0.8, verbose: 0.15, leak: 0 },
  { id: 'openai/gpt-4.1', provider: 'openai', theta: -0.95, verbose: 0.1, leak: 0 },
];

const LINE_ITEMS = [
  'customs receipts', 'internal revenue receipts', 'interest on the public debt',
  'refunds of receipts', 'postal deficit', 'total budget outlays',
  'outstanding marketable securities', 'gold held in the Treasury',
  'miscellaneous receipts', 'veterans benefits outlays', 'national defense outlays',
  'excise tax collections', 'trust fund receipts', 'net federal debt held by the public',
];
const DOCS = [
  'Treasury Bulletin', 'Annual Report of the Secretary of the Treasury',
  'Monthly Treasury Statement', 'Combined Statement of Receipts and Outlays',
];
const CATS = ['Single-table lookup', 'Cross-document sum', 'Year-over-year change', 'Share of total', 'Fiscal-year attribution'];

function fmtNum(v, dec) {
  return v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
}

function makeItem(rng, idx) {
  const cat = CATS[Math.floor(rng() * CATS.length)];
  const li = pick(rng, LINE_ITEMS);
  const doc = pick(rng, DOCS);
  const y = 1921 + Math.floor(rng() * 96);
  let value, dec, unit, question;
  if (cat === 'Single-table lookup') {
    value = Math.round((50 + rng() * 9000) * 10) / 10; dec = 1; unit = 'm';
    question = `According to the ${y + 1} ${doc}, what were ${li} in fiscal year ${y}, in millions of dollars?`;
  } else if (cat === 'Cross-document sum') {
    value = Math.round((200 + rng() * 18000) * 10) / 10; dec = 1; unit = 'm';
    question = `Using the ${doc} for FY${y} and FY${y + 1}, what were combined ${li} across both fiscal years, in millions of dollars?`;
  } else if (cat === 'Year-over-year change') {
    value = Math.round((rng() * 40 - 15) * 10) / 10; dec = 1; unit = '%';
    question = `By what percent did ${li} change from FY${y} to FY${y + 1}? Round to one decimal place.`;
  } else if (cat === 'Share of total') {
    value = Math.round((0.5 + rng() * 38) * 100) / 100; dec = 2; unit = '%';
    question = `What share of total receipts did ${li} represent in FY${y}, per the ${doc}? Answer in percent, two decimals.`;
  } else {
    value = y; dec = 0; unit = 'fy';
    const thr = fmtNum(Math.round(100 + rng() * 5000), 0);
    question = `In which fiscal year did ${li} first exceed $${thr} million, per the ${doc}?`;
  }
  return { idx, category: cat, question, lineItem: li, year: y, doc, value, dec, unit };
}

function distractors(rng, it) {
  const v = it.value;
  const r = (x) => Math.round(x * Math.pow(10, it.dec)) / Math.pow(10, it.dec);
  if (it.unit === 'fy') return [v + (rng() < 0.5 ? 1 : -1), v + 2, v - 3, v + 5];
  if (it.unit === '%') {
    const d1 = r(v * (1 + (rng() < 0.5 ? 1 : -1) * (0.12 + rng() * 0.25)) + (rng() < 0.5 ? 0.3 : -0.3));
    return [d1, r(-v), r(v * 1.9 + 0.7), r(v + 2.4), r(v - 3.1)];
  }
  const d1 = r(v * (1 + (rng() < 0.5 ? 1 : -1) * (0.04 + rng() * 0.12)));
  return [d1, r(v * 1000), r(v * 0.5), r(v * 1.27), r(v * 0.81)];
}

function render(rng, it, val, verbose) {
  const n = fmtNum(val, it.dec);
  if (it.unit === 'fy') return verbose ? `FY${val} (fiscal year ${val})` : `FY${val}`;
  if (it.unit === '%') return verbose ? `From FY${it.year} to FY${it.year + 1}: ${n}%` : `${n}%`;
  return verbose ? `FY${it.year} ${it.lineItem}: $${n} million` : n;
}

export function generate(seed = 20261103, opts = {}) {
  const rng = mulberry32(seed);
  const I = opts.items ?? 240;
  const K = opts.rollouts ?? 4;
  const models = MODELS.map((m) => ({ ...m }));
  const items = [];
  for (let i = 0; i < I; i++) {
    const it = makeItem(rng, i);
    const easy = rng() < 0.3;
    it.b = easy ? -2.2 + gauss(rng) * 0.5 : 0.35 + gauss(rng) * 1.05;
    it.a = Math.exp(0.35 + gauss(rng) * 0.3);
    it.issue = 'none';
    items.push(it);
  }
  // Plant issues on distinct non-easy items.
  const order = items.map((_, i) => i).filter((i) => items[i].b > -1.2);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const plan = [['wrong_key', 9], ['brittle_grader', 8], ['ambiguous', 7]];
  let p = 0;
  for (const [kind, cnt] of plan) for (let c = 0; c < cnt; c++) items[order[p++]].issue = kind;

  const truth = { seed, issues: {}, leakModels: [], trueAccuracy: {} };
  for (const it of items) {
    it.dist = distractors(rng, it);
    it.trueAnswer = it.value;
    it.key = it.value;
    if (it.issue === 'wrong_key') {
      it.key = it.dist[0]; // key equals the most common mistake
      it.b = Math.min(Math.max(it.b, -0.6), 1.2);
    }
    if (it.issue === 'brittle_grader') {
      it.b = Math.min(Math.max(it.b, -0.8), 1.0);
    }
    if (it.issue === 'ambiguous') {
      it.altAnswer = it.dist[0];
      const extra = it.unit === 'fy' ? it.value - 6 : Math.round(it.value * 0.66 * Math.pow(10, it.dec)) / Math.pow(10, it.dec);
      it.dist = [...it.dist.slice(1), extra];
      it.b = Math.min(Math.max(it.b, -1.0), 0.8);
      it.interp = models.map(() => (rng() < 0.5 ? 1 : 0));
      // make sure each reading has at least 3 models
      let ones = it.interp.reduce((s, x) => s + x, 0);
      while (ones < 4) { const j = Math.floor(rng() * models.length); if (!it.interp[j]) { it.interp[j] = 1; ones++; } }
      while (ones > models.length - 4) { const j = Math.floor(rng() * models.length); if (it.interp[j]) { it.interp[j] = 0; ones--; } }
    }
    if (it.issue !== 'none') truth.issues[it.idx] = it.issue;
  }

  const DI = Object.fromEntries(DOMAINS.map((d, i) => [d, i]));
  const rollouts = []; // [model][item] -> array of [answerIndex, reward, domainMask]
  const answers = items.map(() => []); // answer strings per item
  const ansIndex = items.map(() => new Map());
  const addAns = (i, s) => {
    const m = ansIndex[i];
    if (!m.has(s)) { m.set(s, answers[i].length); answers[i].push(s); }
    return m.get(s);
  };
  const keyStr = (it) => render(rng, it, it.key, false);
  for (const it of items) it.keyString = keyStr(it);

  const grade = (it, s) => {
    const nums = numbersIn(s);
    const keyNums = numbersIn(it.keyString);
    if (!nums.length) return 0;
    const last = nums[nums.length - 1];
    const ok = Math.abs(last - it.key) <= 1e-9 + Math.abs(it.key) * 1e-9;
    if (it.issue === 'brittle_grader') return ok && nums.length === keyNums.length ? 1 : 0;
    return ok ? 1 : 0;
  };

  models.forEach((m, mi) => {
    const row = [];
    let knewTrue = 0, total = 0;
    for (const it of items) {
      const cell = [];
      const eta = gauss(rng) * 0.7;
      const pk = sigmoid(it.a * (m.theta - it.b) + eta);
      for (let k = 0; k < K; k++) {
        const knows = rng() < pk;
        const verbose = rng() < m.verbose;
        let mask = 0;
        if (rng() < 0.7) mask |= 1 << DI['fiscaldata.treasury.gov'];
        if (rng() < 0.5) mask |= 1 << DI['govinfo.gov'];
        if (rng() < 0.35) mask |= 1 << DI['fraser.stlouisfed.org'];
        if (rng() < 0.08) mask |= 1 << DI['github.com'];
        if (rng() < (knows ? 0.08 : 0.35)) mask |= 1 << DI['en.wikipedia.org'];
        let val;
        let leaked = false;
        if (knows) {
          val = it.issue === 'ambiguous' && it.interp[mi] ? it.altAnswer : it.trueAnswer;
        } else if (m.leak > 0 && rng() < m.leak) {
          leaked = true;
          val = it.key; // copies the published reference answer
          mask |= 1 << DI[rng() < 0.8 ? 'huggingface.co' : 'modelscope.cn'];
        } else {
          const u = rng();
          if (u < 0.45) val = it.dist[0];
          else if (u < 0.6) val = it.dist[1];
          else if (u < 0.72) val = it.dist[2];
          else if (u < 0.82) val = it.dist[3];
          else {
            const jit = it.unit === 'fy' ? Math.round(gauss(rng) * 8) || 7 : (it.unit === '%' ? Math.round(gauss(rng) * 60) / 10 : Math.round(it.value * (0.5 + rng()) * 10) / 10);
            val = it.unit === 'fy' ? it.value + jit : (it.unit === '%' ? Math.round((it.value + jit) * Math.pow(10, it.dec)) / Math.pow(10, it.dec) : jit);
          }
        }
        if (!leaked && rng() < 0.015) mask |= 1 << DI['huggingface.co']; // benign visits (model cards, docs)
        const s = render(rng, it, val, verbose);
        const reward = grade(it, s);
        cell.push([addAns(it.idx, s), reward, mask, leaked ? 1 : 0]);
        total++;
        if (knows) knewTrue++;
      }
      row.push(cell);
    }
    rollouts.push(row);
    truth.trueAccuracy[m.id] = knewTrue / total;
  });
  truth.leakModels = models.filter((m) => m.leak > 0).map((m) => m.id);
  truth.theta = Object.fromEntries(models.map((m) => [m.id, m.theta]));
  truth.itemB = items.map((it) => it.b);

  const leakedFlags = rollouts.map((row) => row.map((cell) => cell.map((r) => r[3])));
  const data = {
    environment: { owner: 'primeintellect', name: 'fiscal-docs-qa', version: '0.2.1', rollouts: K },
    domains: DOMAINS,
    models: models.map((m) => ({ id: m.id, provider: m.provider, effort: m.effort ?? null })),
    items: items.map((it) => ({ idx: it.idx, category: it.category, question: it.question, key: it.keyString, answers: answers[it.idx] })),
    rollouts: rollouts.map((row) => row.map((cell) => cell.map((r) => [r[0], r[1], r[2]]))),
  };
  truth.leaked = leakedFlags;
  return { data, truth };
}

export function numbersIn(s) {
  const out = [];
  const re = /-?\d[\d,]*(?:\.\d+)?/g;
  let m;
  while ((m = re.exec(String(s)))) {
    const v = parseFloat(m[0].replace(/,/g, ''));
    if (!Number.isNaN(v)) out.push(v);
  }
  return out;
}
