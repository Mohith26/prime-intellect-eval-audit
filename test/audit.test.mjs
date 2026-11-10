import test from 'node:test';
import assert from 'node:assert/strict';
import { generate, mulberry32 as simRng } from '../core/simulate.mjs';
import { canonical, numbersIn, mantelHaenszel, fitIRT, runAudit, cramersV, bootstrap, mulberry32, sigmoid } from '../core/audit.mjs';

const { data, truth } = generate(20261103);
const A = runAudit(data);

test('answer normalization handles verbose and formatted answers', () => {
  assert.deepEqual(numbersIn('FY2014 (fiscal year 2014)'), [2014, 2014]);
  assert.equal(canonical('FY2014 (fiscal year 2014)'), canonical('FY2014'));
  assert.equal(canonical('FY1934 customs receipts: $4,812.6 million'), canonical('4,812.6'));
  assert.equal(canonical('From FY1950 to FY1951: -4.3%'), '-4.3');
  assert.notEqual(canonical('4,812.6'), canonical('4,812,600.0'));
});

test('Mantel-Haenszel odds ratio matches a hand-computed value and is 1 under independence', () => {
  // two strata with common OR = 4: (a d)/(b c) = 4 in each
  const mh = mantelHaenszel([[8, 2, 5, 5], [4, 4, 2, 8]]);
  assert.ok(Math.abs(mh.or - 4) < 1e-9, `or=${mh.or}`);
  assert.ok(mh.lo < 4 && mh.hi > 4);
  const flat = mantelHaenszel([[5, 5, 5, 5], [3, 7, 3, 7]]);
  assert.ok(Math.abs(flat.or - 1) < 1e-9);
});

test('2PL fit recovers known parameters on a large synthetic matrix', () => {
  const rng = mulberry32(3);
  const M = 40, I = 120, n = 6;
  const th = Array.from({ length: M }, (_, m) => -2 + 4 * m / (M - 1));
  const b = Array.from({ length: I }, () => -2 + 4 * rng());
  const a = Array.from({ length: I }, () => 0.6 + 1.6 * rng());
  const s = [], nn = [];
  for (let m = 0; m < M; m++) {
    s.push(new Float64Array(I)); nn.push(new Float64Array(I).fill(n));
    for (let i = 0; i < I; i++) for (let k = 0; k < n; k++) if (rng() < sigmoid(a[i] * (th[m] - b[i]))) s[m][i]++;
  }
  const fit = fitIRT(s, nn);
  const corr = (x, y) => { const k = x.length, mx = x.reduce((p, c) => p + c) / k, my = y.reduce((p, c) => p + c) / k; let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < k; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; } return sxy / Math.sqrt(sxx * syy); };
  assert.ok(corr(fit.theta, th) > 0.98, 'theta');
  assert.ok(corr(fit.b, b) > 0.95, 'difficulty');
  assert.ok(corr(fit.a, a) > 0.6, 'discrimination');
});

test('IRT on the demo environment orders models like the simulator does', () => {
  const t = data.models.map((m) => truth.theta[m.id]);
  const rank = (v) => { const o = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]); const r = []; o.forEach(([, i], k) => { r[i] = k; }); return r; };
  const r1 = rank(A.irt.theta), r2 = rank(t);
  const d2 = r1.reduce((s, r, i) => s + (r - r2[i]) ** 2, 0);
  const rho = 1 - 6 * d2 / (t.length * (t.length ** 2 - 1));
  assert.ok(rho > 0.95, `spearman ${rho}`);
});

test('leak scan flags only the solution-source domains', () => {
  const src = A.leak.domains.filter((d) => d.solutionSource).map((d) => d.domain).sort();
  assert.deepEqual(src, ['huggingface.co', 'modelscope.cn']);
  const fd = A.leak.domains.find((d) => d.domain === 'fiscaldata.treasury.gov');
  assert.ok(fd.lo < 1 && fd.hi > 1, 'reference-data domain is neutral');
  // retrieved rollouts concentrate in the two models that leak in the simulator
  const leakers = truth.leakModels.map((id) => data.models.findIndex((m) => m.id === id));
  const inLeakers = leakers.reduce((s, m) => s + A.leak.perModel[m].retrieved, 0);
  assert.ok(inLeakers / A.totalRetrieved > 0.9);
});

test('planted issues are recovered on the demo seed with no false flags', () => {
  const map = { wrong_key: 'key', brittle_grader: 'grader', ambiguous: 'ambiguous' };
  for (const [kind, cls] of Object.entries(map)) {
    const planted = Object.entries(truth.issues).filter(([, v]) => v === kind).map(([k]) => +k);
    const hit = planted.filter((i) => A.items[i].cls === cls).length;
    assert.ok(hit / planted.length >= 0.7, `${kind} recall ${hit}/${planted.length}`);
  }
  const clean = A.items.filter((x) => x.cls !== 'ok' && !truth.issues[x.i]).length;
  assert.ok(clean <= 2, `clean false flags ${clean}`);
});

test('wrong keys show the signature: consensus answer rises with ability, key does not', () => {
  const keys = A.items.filter((x) => x.cls === 'key');
  assert.ok(keys.length > 0);
  for (const x of keys) assert.ok(x.key.gradAlt > x.key.gradKey, `item ${x.i}`);
});

test('across 6 seeds: high precision, most planted issues recovered', () => {
  const map = { wrong_key: 'key', brittle_grader: 'grader', ambiguous: 'ambiguous' };
  const tot = { wrong_key: [0, 0], brittle_grader: [0, 0], ambiguous: [0, 0] };
  let clean = 0, cleanN = 0;
  for (let s = 1; s <= 6; s++) {
    const g = generate(1000 + s);
    const R = runAudit(g.data, { bootstrap: 50 });
    for (const [kind, cls] of Object.entries(map)) {
      const planted = Object.entries(g.truth.issues).filter(([, v]) => v === kind).map(([k]) => +k);
      tot[kind][0] += planted.filter((i) => R.items[i].cls === cls).length; tot[kind][1] += planted.length;
    }
    R.items.forEach((x) => { if (!g.truth.issues[x.i]) { cleanN++; if (x.cls !== 'ok') clean++; } });
  }
  assert.ok(tot.brittle_grader[0] / tot.brittle_grader[1] >= 0.95);
  assert.ok(tot.wrong_key[0] / tot.wrong_key[1] >= 0.75);
  assert.ok(tot.ambiguous[0] / tot.ambiguous[1] >= 0.5);
  assert.ok(clean / cleanN < 0.01, `clean false-flag rate ${clean}/${cleanN}`);
});

test('audited leaderboard is closer to latent accuracy than the published one', () => {
  const mae = (b) => data.models.reduce((s, m, mi) => s + Math.abs(b[mi].mean - truth.trueAccuracy[m.id]), 0) / data.models.length;
  assert.ok(mae(A.audBoot) < mae(A.pubBoot) / 2, `${mae(A.audBoot)} vs ${mae(A.pubBoot)}`);
});

test('bootstrap is deterministic, ordered, and rank-1 probabilities sum to 1', () => {
  const b1 = bootstrap(A.audited, 200, mulberry32(9));
  const b2 = bootstrap(A.audited, 200, mulberry32(9));
  assert.deepEqual(b1, b2);
  b1.forEach((x) => assert.ok(x.lo <= x.mean + 1e-9 && x.mean <= x.hi + 1e-9));
  assert.ok(Math.abs(b1.reduce((s, x) => s + x.pTop, 0) - 1) < 1e-9);
});

test("Cramer's V is 0 for identical rows and 1 for a perfect split", () => {
  assert.equal(cramersV([[2, 2], [2, 2]]), 0);
  assert.ok(Math.abs(cramersV([[4, 0], [0, 4]]) - 1) < 1e-9);
});
