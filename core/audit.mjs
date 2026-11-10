// Eval Audit core: cross-model statistical audit of every published run of one
// environment. Pure functions, no DOM. Used by the page (app.js) and by node --test.
//
// Pipeline
//   1. leakScan        Mantel-Haenszel odds ratio of reward given a fetched domain,
//                      stratified by (model, example) so ability and difficulty cancel.
//                      Domains whose lower 95% bound clears a threshold are treated as
//                      solution sources; rollouts that fetched them are "retrieved".
//   2. fitIRT          2PL item response model (MAP, alternating Newton) on the
//                      non-retrieved rollouts: ability per model, difficulty and
//                      discrimination per example (discrimination may go negative).
//   3. graderTest      failing rollouts whose normalized answer equals the normalized key
//                      (the grader rejected a correct answer).
//   4. keyErrorTest    Dawid-Skene style likelihood ratio: is the reference answer or
//                      the strongest non-key consensus the better explanation of what
//                      14 models x K rollouts actually answered?
//   5. ambiguityTest   strong models split into >=2 internally stable answers:
//                      Cramer's V of model x answer with a permutation p-value.
//   6. repair + rescore, item-clustered bootstrap, test information (saturation).

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

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

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

// Canonical form of a free-form answer: the final number (6 significant digits),
// or the lowercased text if there is no number.
export function canonical(s) {
  const nums = numbersIn(s);
  if (nums.length) return String(Number(nums[nums.length - 1].toPrecision(6)));
  return String(s).trim().toLowerCase().replace(/\s+/g, ' ');
}

// ---------- 1. leak scan (Mantel-Haenszel with Robins-Breslow-Greenland variance)
export function mantelHaenszel(strata, opts = {}) {
  // strata: array of [a, b, c, d] = [exposed&success, exposed&fail, unexposed&success, unexposed&fail]
  // opts.haldane: add 0.5 to every cell of informative strata that contain a zero
  let R = 0, S = 0, sPR = 0, sPSQR = 0, sQS = 0, used = 0;
  for (let [a, b, c, d] of strata) {
    if (a + b === 0 || c + d === 0) continue;
    if (opts.haldane && (a === 0 || b === 0 || c === 0 || d === 0)) { a += 0.5; b += 0.5; c += 0.5; d += 0.5; }
    const n = a + b + c + d;
    if (n < 2) continue;
    const r = (a * d) / n, s = (b * c) / n;
    const P = (a + d) / n, Q = (b + c) / n;
    R += r; S += s; sPR += P * r; sPSQR += P * s + Q * r; sQS += Q * s; used++;
  }
  if (R === 0 || S === 0) return { or: R === 0 ? 0 : Infinity, lo: NaN, hi: NaN, strata: used };
  const or = R / S;
  const v = sPR / (2 * R * R) + sPSQR / (2 * R * S) + sQS / (2 * S * S);
  const se = Math.sqrt(v);
  return { or, lo: Math.exp(Math.log(or) - 1.96 * se), hi: Math.exp(Math.log(or) + 1.96 * se), se, strata: used };
}

export function leakScan(data, opts = {}) {
  const threshold = opts.orLowerBound ?? 2;
  const D = data.domains.length;
  const res = [];
  for (let d = 0; d < D; d++) {
    const bit = 1 << d;
    const strata = [];
    let visits = 0, visitWins = 0, total = 0;
    data.rollouts.forEach((row) => row.forEach((cell) => {
      let a = 0, b = 0, c = 0, e = 0;
      for (const [, r, mask] of cell) {
        total++;
        if (mask & bit) { visits++; if (r) { a++; visitWins++; } else b++; } else if (r) c++; else e++;
      }
      strata.push([a, b, c, e]);
    }));
    let mh = mantelHaenszel(strata);
    if (!Number.isFinite(mh.lo) && visits > 0) {
      // every informative stratum is concordant (S = 0): fall back to a pooled
      // Haldane-corrected 2x2 with a Woolf interval so the bound stays finite.
      const t = strata.reduce((acc, x) => acc.map((v, j) => v + x[j]), [0, 0, 0, 0]).map((v) => v + 0.5);
      const or = (t[0] * t[3]) / (t[1] * t[2]);
      const se = Math.sqrt(1 / t[0] + 1 / t[1] + 1 / t[2] + 1 / t[3]);
      mh = { or: mh.or, lo: Math.exp(Math.log(or) - 1.96 * se), hi: Math.exp(Math.log(or) + 1.96 * se), se, strata: mh.strata, pooled: true };
    }
    const byModel = data.models.map((_, mi) => {
      let v = 0, n = 0;
      data.rollouts[mi].forEach((cell) => cell.forEach(([, , mask]) => { n++; if (mask & bit) v++; }));
      return v / n;
    });
    res.push({ domain: data.domains[d], visits, visitRate: visits / total, visitSuccess: visits ? visitWins / visits : 0, ...mh, byModel,
      solutionSource: Number.isFinite(mh.lo) && mh.lo > threshold });
  }
  const leakMask = res.reduce((m, r, d) => (r.solutionSource ? m | (1 << d) : m), 0);
  // Per-model attribution. A model is attributed retrieval when (1) its rate of fetching a
  // solution-source domain is far above the cross-model baseline (median rate; exact
  // binomial upper tail, Bonferroni over models) and (2) within its own rollouts on the
  // same example, fetching raises the odds of reward (MH point estimate above modelOr).
  // Occasional benign visits (model cards, docs) by other models are not counted.
  const modelOr = opts.modelOr ?? 2;
  const alpha = (opts.modelAlpha ?? 0.01) / data.models.length;
  const visitsBy = data.models.map((_, mi) => {
    let v = 0, n = 0;
    data.rollouts[mi].forEach((cell) => cell.forEach(([, , mask]) => { n++; if (mask & leakMask) v++; }));
    return { v, n };
  });
  const rates = visitsBy.map((x) => x.v / x.n).sort((x, y) => x - y);
  const baseline = Math.max(1e-4, rates[Math.floor(rates.length / 2)]);
  const modelEvidence = data.models.map((_, mi) => {
    const strata = [];
    data.rollouts[mi].forEach((cell) => {
      let a = 0, b = 0, c = 0, e = 0;
      for (const [, r, mask] of cell) { if (mask & leakMask) { if (r) a++; else b++; } else if (r) c++; else e++; }
      strata.push([a, b, c, e]);
    });
    const mh = leakMask ? mantelHaenszel(strata) : { or: NaN, lo: NaN, hi: NaN };
    const { v, n } = visitsBy[mi];
    const pRate = binomUpper(v, n, baseline);
    const retriever = leakMask !== 0 && pRate < alpha && (mh.or > modelOr || mh.or === Infinity);
    return { ...mh, visits: v, visitRate: v / n, baseline, pRate, retriever };
  });
  const retrieved = data.rollouts.map((row, mi) => row.map((cell) => cell.map(([, , mask]) => ((mask & leakMask) && modelEvidence[mi].retriever ? 1 : 0))));
  const perModel = data.models.map((_, mi) => {
    let n = 0, k = 0, wins = 0;
    retrieved[mi].forEach((cell, i) => cell.forEach((f, j) => { n++; if (f) { k++; if (data.rollouts[mi][i][j][1]) wins++; } }));
    return { retrieved: k, rate: k / n, retrievedWins: wins, ...modelEvidence[mi] };
  });
  return { domains: res, leakMask, retrieved, perModel };
}

// exact binomial upper tail P(X >= k | n, p), computed in log space
export function binomUpper(k, n, p) {
  if (k <= 0) return 1;
  const lg = (x) => { // log-gamma via Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lg(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  };
  const lc = (j) => lg(n + 1) - lg(j + 1) - lg(n - j + 1) + j * Math.log(p) + (n - j) * Math.log(1 - p);
  let mx = -Infinity; const terms = [];
  for (let j = k; j <= n; j++) { const t = lc(j); terms.push(t); if (t > mx) mx = t; }
  return Math.min(1, Math.exp(mx) * terms.reduce((s2, t) => s2 + Math.exp(t - mx), 0));
}

// ---------- 2. 2PL IRT (MAP)
export function cellCounts(data, exclude) {
  const M = data.models.length, I = data.items.length;
  const s = [], n = [];
  for (let m = 0; m < M; m++) {
    s.push(new Float64Array(I)); n.push(new Float64Array(I));
    for (let i = 0; i < I; i++) {
      data.rollouts[m][i].forEach((r, k) => {
        if (exclude && exclude[m][i][k]) return;
        n[m][i] += 1; s[m][i] += r[1];
      });
    }
  }
  return { s, n };
}

export function fitIRT(s, n, opts = {}) {
  const M = s.length, I = s[0].length;
  const iters = opts.iters ?? 80;
  const muA = opts.muA ?? 1.0, sdA = opts.sdA ?? 0.8, sdB = opts.sdB ?? 2.0;
  const theta = new Float64Array(M), a = new Float64Array(I).fill(1), b = new Float64Array(I);
  // init: theta from logit of overall rate, b from logit of item rate
  for (let m = 0; m < M; m++) {
    let ss = 0, nn = 0; for (let i = 0; i < I; i++) { ss += s[m][i]; nn += n[m][i]; }
    const p = (ss + 0.5) / (nn + 1); theta[m] = Math.log(p / (1 - p));
  }
  for (let i = 0; i < I; i++) {
    let ss = 0, nn = 0; for (let m = 0; m < M; m++) { ss += s[m][i]; nn += n[m][i]; }
    const p = (ss + 0.5) / (nn + 1); b[i] = -Math.log(p / (1 - p));
  }
  const clip = (x, c) => Math.max(-c, Math.min(c, x));
  for (let it = 0; it < iters; it++) {
    for (let m = 0; m < M; m++) {
      let g = -theta[m], h = -1;
      for (let i = 0; i < I; i++) {
        if (!n[m][i]) continue;
        const P = sigmoid(a[i] * (theta[m] - b[i]));
        g += a[i] * (s[m][i] - n[m][i] * P);
        h -= a[i] * a[i] * n[m][i] * P * (1 - P);
      }
      theta[m] -= clip(g / h, 1);
    }
    // standardize abilities (identification), rescale items accordingly
    let mean = 0; for (let m = 0; m < M; m++) mean += theta[m]; mean /= M;
    let sd = 0; for (let m = 0; m < M; m++) sd += (theta[m] - mean) ** 2; sd = Math.sqrt(sd / M) || 1;
    for (let m = 0; m < M; m++) theta[m] = (theta[m] - mean) / sd;
    for (let i = 0; i < I; i++) { b[i] = (b[i] - mean) / sd; a[i] = a[i] * sd; }
    for (let i = 0; i < I; i++) {
      let gb = -b[i] / (sdB * sdB), hb = -1 / (sdB * sdB);
      let ga = -(a[i] - muA) / (sdA * sdA), ha = -1 / (sdA * sdA);
      for (let m = 0; m < M; m++) {
        if (!n[m][i]) continue;
        const P = sigmoid(a[i] * (theta[m] - b[i]));
        const r = s[m][i] - n[m][i] * P, w = n[m][i] * P * (1 - P);
        ga += (theta[m] - b[i]) * r; ha -= (theta[m] - b[i]) ** 2 * w;
        gb -= a[i] * r; hb -= a[i] * a[i] * w;
      }
      a[i] -= clip(ga / ha, 0.5);
      b[i] -= clip(gb / hb, 0.75);
      b[i] = clip(b[i], 6);
    }
  }
  let ll = 0;
  for (let m = 0; m < M; m++) for (let i = 0; i < I; i++) {
    if (!n[m][i]) continue;
    const P = Math.min(1 - 1e-9, Math.max(1e-9, sigmoid(a[i] * (theta[m] - b[i]))));
    ll += s[m][i] * Math.log(P) + (n[m][i] - s[m][i]) * Math.log(1 - P);
  }
  return { theta: Array.from(theta), a: Array.from(a), b: Array.from(b), loglik: ll };
}

export function itemRest(s, n, i) {
  // point-biserial: model pass rate on item i vs model mean on all other items
  const M = s.length, I = s[0].length;
  const x = [], y = [];
  for (let m = 0; m < M; m++) {
    if (!n[m][i]) continue;
    let ss = 0, nn = 0; for (let j = 0; j < I; j++) if (j !== i) { ss += s[m][j]; nn += n[m][j]; }
    x.push(s[m][i] / n[m][i]); y.push(ss / nn);
  }
  return pearson(x, y);
}

export function pearson(x, y) {
  const k = x.length; if (k < 3) return 0;
  const mx = x.reduce((p, c) => p + c, 0) / k, my = y.reduce((p, c) => p + c, 0) / k;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < k; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
}

// ---------- per-item answer profile
export function answerProfile(data, i, retrieved) {
  const item = data.items[i];
  const keyC = canonical(item.key);
  const perModel = data.models.map((_, m) => data.rollouts[m][i].map((r, k) => ({
    c: canonical(item.answers[r[0]]), raw: item.answers[r[0]], reward: r[1], retrieved: retrieved ? retrieved[m][i][k] : 0,
  })));
  const counts = new Map();
  perModel.forEach((rs) => rs.forEach((r) => { if (!r.retrieved) counts.set(r.c, (counts.get(r.c) || 0) + 1); }));
  return { keyC, perModel, counts };
}

// ---------- 3. grader false negatives
export function graderTest(profile, opts = {}) {
  const minRollouts = opts.minRollouts ?? 3, minModels = opts.minModels ?? 2;
  let fn = 0, fails = 0; const models = new Set(); const examples = [];
  let extraNums = 0;
  profile.perModel.forEach((rs, m) => rs.forEach((r) => {
    if (r.retrieved || r.reward) return;
    fails++;
    if (r.c === profile.keyC) { fn++; models.add(m); if (examples.length < 3) examples.push({ model: m, raw: r.raw }); extraNums += numbersIn(r.raw).length; }
  }));
  return { fn, fails, models: models.size, examples, meanNumbers: fn ? extraNums / fn : 0, flagged: fn >= minRollouts && models.size >= minModels };
}

// ---------- 4. answer-key error (ability-aware likelihood ratio)
// Under hypothesis "truth = t", model m answers t with probability
// sigmoid(slope * (theta_m - beta)), with beta profiled out by grid search, and
// otherwise picks a wrong answer c with smoothed empirical frequency w(c | t).
// A wrong key shows up as a non-key answer whose rate RISES with ability, while
// the key is matched mostly by weaker models (negative discrimination). A merely
// common mistake on a hard item has no such gradient and stays with the key.
export function profileLL(profile, theta, t, slope, tau) {
  const V = profile.counts.size + 1;
  const total = [...profile.counts.values()].reduce((p, c) => p + c, 0);
  const ct = profile.counts.get(t) || 0;
  let wrongLL = 0;
  const x = [], nn = [];
  profile.perModel.forEach((rs, m) => {
    let hit = 0, n = 0;
    rs.forEach((r) => {
      if (r.retrieved) return;
      n++;
      if (r.c === t) hit++;
      else wrongLL += Math.log(((profile.counts.get(r.c) || 0) + 0.5) / (total - ct + 0.5 * V));
    });
    x.push(hit); nn.push(n);
  });
  let best = -Infinity, bestBeta = 0;
  for (let beta = -4; beta <= 6.0001; beta += 0.05) {
    let ll = 0;
    for (let m = 0; m < theta.length; m++) {
      if (!nn[m]) continue;
      const p = Math.min(1 - 1e-6, Math.max(1e-6, sigmoid(slope * (theta[m] - beta))));
      ll += x[m] * Math.log(p) + (nn[m] - x[m]) * Math.log(1 - p);
    }
    if (ll > best) { best = ll; bestBeta = beta; }
  }
  return { ll: tau * (best + wrongLL), beta: bestBeta, hits: x };
}

export function keyErrorTest(profile, theta, slope, opts = {}) {
  const prior = opts.prior ?? 0.02, tau = opts.tau ?? 0.5;
  const M = theta.length;
  const order = theta.map((t, m) => [t, m]).sort((x, y) => y[0] - x[0]);
  const strong = new Set(order.slice(0, Math.ceil(M / 2)).map((x) => x[1]));
  const strongCounts = new Map();
  profile.perModel.forEach((rs, m) => {
    if (!strong.has(m)) return;
    rs.forEach((r) => { if (!r.retrieved && r.c !== profile.keyC) strongCounts.set(r.c, (strongCounts.get(r.c) || 0) + 1); });
  });
  let alt = null, altN = 0;
  for (const [c, k] of strongCounts) if (k > altN) { alt = c; altN = k; }
  if (alt === null) return { alt: null, logLR: -Infinity, posterior: 0 };
  const H1 = profileLL(profile, theta, alt, slope, tau);
  const H0 = profileLL(profile, theta, profile.keyC, slope, tau);
  const logLR = H1.ll - H0.ll;
  const posterior = sigmoid(logLR + Math.log(prior / (1 - prior)));
  // ability gradient of each answer (Pearson of per-model match rate vs theta)
  const rate = (h) => h.map((k, m) => { const n = profile.perModel[m].filter((r) => !r.retrieved).length; return n ? k / n : 0; });
  return { alt, altStrongRollouts: altN, logLR, posterior, betaAlt: H1.beta, betaKey: H0.beta,
    gradAlt: pearson(theta, rate(H1.hits)), gradKey: pearson(theta, rate(H0.hits)) };
}

// ---------- 5. ambiguity (stable disagreement among strong models)
export function cramersV(table) {
  const R = table.length, C = table[0].length;
  const rs = table.map((r) => r.reduce((p, c) => p + c, 0));
  const cs = Array.from({ length: C }, (_, j) => table.reduce((p, r) => p + r[j], 0));
  const N = rs.reduce((p, c) => p + c, 0);
  if (!N) return 0;
  let chi = 0;
  for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
    const e = (rs[i] * cs[j]) / N; if (e > 0) chi += (table[i][j] - e) ** 2 / e;
  }
  const k = Math.min(R, C) - 1; return k > 0 ? Math.sqrt(chi / (N * k)) : 0;
}

export function ambiguityTest(profile, theta, rng, opts = {}) {
  const B = opts.permutations ?? 300;
  const M = theta.length;
  const order = theta.map((t, m) => [t, m]).sort((x, y) => y[0] - x[0]);
  const strong = order.slice(0, Math.ceil(M * (opts.ambShare ?? 0.5))).map((x) => x[1]);
  const minCons = opts.ambConsistency ?? 0.5;
  const modal = [];
  for (const m of strong) {
    const cnt = new Map(); let n = 0;
    profile.perModel[m].forEach((r) => { if (!r.retrieved) { cnt.set(r.c, (cnt.get(r.c) || 0) + 1); n++; } });
    let best = null, bn = 0; for (const [c, k] of cnt) if (k > bn) { best = c; bn = k; }
    modal.push({ m, c: best, consistency: n ? bn / n : 0 });
  }
  const holders = new Map();
  modal.forEach((x) => { if (x.c !== null && x.consistency >= minCons) holders.set(x.c, (holders.get(x.c) || 0) + 1); });
  const readings = [...holders.entries()].filter(([, k]) => k >= 2).sort((x, y) => y[1] - x[1]).map(([c, k]) => ({ c, models: k }));
  if (readings.length < 2) return { readings, V: 0, p: 1, flagged: false };
  const cols = readings.map((r) => r.c);
  const labels = []; const rows = [];
  strong.forEach((m, ri) => profile.perModel[m].forEach((r) => {
    if (r.retrieved) return; const j = cols.indexOf(r.c); labels.push(j === -1 ? cols.length : j); rows.push(ri);
  }));
  const tab = (lab) => {
    const t = strong.map(() => new Array(cols.length + 1).fill(0));
    lab.forEach((j, k) => { t[rows[k]][j]++; }); return t;
  };
  const V = cramersV(tab(labels));
  let ge = 0; const perm = labels.slice();
  for (let b = 0; b < B; b++) {
    for (let k = perm.length - 1; k > 0; k--) { const j = Math.floor(rng() * (k + 1)); [perm[k], perm[j]] = [perm[j], perm[k]]; }
    if (cramersV(tab(perm)) >= V - 1e-12) ge++;
  }
  const p = (ge + 1) / (B + 1);
  const meanConsistency = modal.reduce((s, x) => s + x.consistency, 0) / modal.length;
  return { readings, V, p, meanConsistency, flagged: p < (opts.alpha ?? 0.01) && readings.some((r) => r.c === profile.keyC) };
}

// ---------- 6. scores, bootstrap, information
export function itemScores(data, rewardFn) {
  // returns [model][item] mean reward under a (possibly repaired) grading function
  return data.models.map((_, m) => data.items.map((_, i) => {
    const cell = data.rollouts[m][i];
    let s = 0; cell.forEach((r, k) => { s += rewardFn(m, i, k, r); }); return s / cell.length;
  }));
}

export function bootstrap(scores, B, rng) {
  const M = scores.length, I = scores[0].length;
  const samples = scores.map(() => new Float64Array(B));
  const rank1 = new Array(M).fill(0);
  const idx = new Int32Array(I);
  for (let b = 0; b < B; b++) {
    for (let k = 0; k < I; k++) idx[k] = Math.floor(rng() * I);
    let best = -1, bm = -1;
    for (let m = 0; m < M; m++) {
      let s = 0; for (let k = 0; k < I; k++) s += scores[m][idx[k]];
      s /= I; samples[m][b] = s; if (s > best) { best = s; bm = m; }
    }
    rank1[bm]++;
  }
  return scores.map((row, m) => {
    const arr = Array.from(samples[m]).sort((x, y) => x - y);
    const q = (p) => arr[Math.min(B - 1, Math.max(0, Math.round(p * (B - 1))))];
    return { mean: row.reduce((p, c) => p + c, 0) / row.length, lo: q(0.05), hi: q(0.95), pTop: rank1[m] / B };
  });
}

export function testInformation(irt, grid) {
  return grid.map((t) => irt.a.reduce((s, a, i) => { const P = sigmoid(a * (t - irt.b[i])); return s + a * a * P * (1 - P); }, 0));
}

// ---------- orchestration
export function runAudit(data, opts = {}) {
  const seed = opts.seed ?? 7;
  const rng = mulberry32(seed);
  const keyThreshold = opts.keyThreshold ?? 0.9;
  const reviewThreshold = opts.reviewThreshold ?? 0.5;
  const K = data.environment.rollouts;
  const leak = leakScan(data, opts);
  const { s, n } = cellCounts(data, leak.retrieved);
  const irt = fitIRT(s, n);
  // slope for the key test: median fitted discrimination (typical item)
  const sortedA = irt.a.slice().sort((x, y) => x - y);
  const slope = Math.max(0.5, sortedA[Math.floor(sortedA.length / 2)]);
  const items = data.items.map((item, i) => {
    const prof = answerProfile(data, i, leak.retrieved);
    const grader = graderTest(prof, opts);
    const key = keyErrorTest(prof, irt.theta, slope, opts);
    const amb = ambiguityTest(prof, irt.theta, rng, opts);
    let cls = 'ok';
    if (grader.flagged) cls = 'grader';
    else if (amb.flagged) cls = 'ambiguous';
    else if (key.posterior >= keyThreshold) cls = 'key';
    else if (key.posterior >= reviewThreshold) cls = 'review';
    let ss = 0, nn = 0; for (let m = 0; m < s.length; m++) { ss += s[m][i]; nn += n[m][i]; }
    return { i, cls, a: irt.a[i], b: irt.b[i], rest: itemRest(s, n, i), passRate: nn ? ss / nn : 0, grader, key, amb, keyC: prof.keyC };
  });
  // repaired grading
  const repaired = (m, i, k, r) => {
    if (leak.retrieved[m][i][k]) return 0;
    const it = items[i];
    const c = canonical(data.items[i].answers[r[0]]);
    if (it.cls === 'grader') return c === it.keyC ? 1 : 0;
    if (it.cls === 'key') return c === it.key.alt ? 1 : 0;
    if (it.cls === 'ambiguous') return it.amb.readings.some((x) => x.c === c) ? 1 : 0;
    return r[1];
  };
  const published = itemScores(data, (m, i, k, r) => r[1]);
  const audited = itemScores(data, repaired);
  const B = opts.bootstrap ?? 500;
  const pubBoot = bootstrap(published, B, mulberry32(seed + 1));
  const audBoot = bootstrap(audited, B, mulberry32(seed + 1));
  const rankOf = (arr) => { const o = arr.map((x, m) => [x.mean, m]).sort((p, q) => q[0] - p[0]); const r = new Array(arr.length); o.forEach(([, m], k) => { r[m] = k + 1; }); return r; };
  const pubRank = rankOf(pubBoot), audRank = rankOf(audBoot);
  // saturation on repaired data
  const sr = data.models.map((_, m) => Float64Array.from(audited[m].map((x) => x * K)));
  const nr = data.models.map(() => new Float64Array(data.items.length).fill(K));
  const irt2 = fitIRT(sr, nr);
  const grid = []; for (let t = -3; t <= 3.0001; t += 0.1) grid.push(Math.round(t * 10) / 10);
  const info = testInformation(irt2, grid);
  const top = audRank.indexOf(1);
  const topTheta = irt2.theta[top];
  const infoTop = testInformation(irt2, [topTheta])[0];
  const order = audBoot.map((x, m) => [x.mean, m]).sort((p, q) => q[0] - p[0]).map((x) => x[1]);
  const top5 = order.slice(0, 5);
  let saturated = 0; for (let i = 0; i < data.items.length; i++) if (top5.every((m) => audited[m][i] === 1)) saturated++;
  const flagged = items.filter((x) => x.cls !== 'ok');
  const counts = { key: 0, grader: 0, ambiguous: 0, review: 0 };
  flagged.forEach((x) => { counts[x.cls]++; });
  // unexpected passes: successful, non-repaired cells with low IRT probability
  const surprises = [];
  data.models.forEach((_, m) => data.items.forEach((_, i) => {
    const P = sigmoid(irt.a[i] * (irt.theta[m] - irt.b[i]));
    data.rollouts[m][i].forEach((r, k) => {
      if (r[1] && P < 0.08) surprises.push({ m, i, k, P, retrieved: leak.retrieved[m][i][k], mask: r[2] });
    });
  }));
  surprises.sort((x, y) => x.P - y.P);
  const totalRetrieved = leak.perModel.reduce((p, c) => p + c.retrieved, 0);
  return {
    leak, irt, irt2, slope, items, flagged, counts,
    published, audited, pubBoot, audBoot, pubRank, audRank,
    grid, info, topModel: top, topTheta, seTop: 1 / Math.sqrt(K * infoTop), saturated, saturatedShare: saturated / data.items.length,
    surprises, totalRetrieved, totalRollouts: data.models.length * data.items.length * K,
    rankChanges: pubRank.filter((r, m) => r !== audRank[m]).length,
  };
}
