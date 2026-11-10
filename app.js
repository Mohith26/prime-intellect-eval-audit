import { runAudit, canonical, sigmoid } from './core/audit.mjs';

const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '∞');
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const ICON = (id) => {
  if (id.startsWith('google/gemma')) return 'gemma';
  if (id.startsWith('google/')) return 'gemini';
  if (id.startsWith('openai/')) return 'openai';
  if (id.startsWith('anthropic/')) return 'anthropic';
  if (/qwen/i.test(id)) return 'qwen';
  if (id.startsWith('moonshotai/')) return 'kimi';
  if (id.startsWith('z-ai/')) return 'zai';
  if (id.startsWith('deepseek/')) return 'deepseek';
  if (id.startsWith('minimax/')) return 'minimax';
  if (id.startsWith('xiaomi/')) return 'xiaomimimo';
  return 'openai';
};
const short = (id) => id.split('/')[1];
const modelCell = (id) => `<div class="model"><img src="assets/icons/${ICON(id)}.svg" alt="" />${esc(id)}</div>`;
const CLS = { key: ['Answer key', 'b-key'], grader: ['Grader', 'b-grader'], ambiguous: ['Ambiguous', 'b-ambiguous'], review: ['Review', 'b-review'], ok: ['OK', 'b-ok'] };
const badge = (cls) => `<span class="badge ${CLS[cls][1]}">${CLS[cls][0]}</span>`;
const COLORS = { key: '#22c55e', alt: '#8b5cf6', alt2: '#3b82f6', other: '#3a3a3a' };

let DATA, A, filter = 'all', selected = null, keyT = 0.9;
const state = { reps: [] };

function icons() { if (window.lucide) window.lucide.createIcons(); }

function displayMap(item) {
  // canonical -> shortest raw answer, for readable display
  const m = new Map();
  item.answers.forEach((raw) => { const c = canonical(raw); if (!m.has(c) || raw.length < m.get(c).length) m.set(c, raw); });
  return m;
}

function compute() {
  A = runAudit(DATA, { keyThreshold: keyT, bootstrap: 500 });
  state.reps = DATA.items.map(displayMap);
}

function modalAnswers(i) {
  return DATA.models.map((_, m) => {
    const cnt = new Map();
    DATA.rollouts[m][i].forEach((r, k) => { if (A.leak.retrieved[m][i][k]) return; const c = canonical(DATA.items[i].answers[r[0]]); cnt.set(c, (cnt.get(c) || 0) + 1); });
    let best = null, bn = 0; for (const [c, k] of cnt) if (k > bn) { best = c; bn = k; }
    return best;
  });
}

function evidence(x) {
  const rep = state.reps[x.i];
  const show = (c) => esc(rep.get(c) ?? c);
  if (x.cls === 'key' || x.cls === 'review') {
    const modal = modalAnswers(x.i);
    const n = modal.filter((c) => c === x.key.alt).length;
    return `<span class="mono">${show(x.key.alt)}</span> (${n}/${DATA.models.length}) · key <span class="mono">${show(x.keyC)}</span>`;
  }
  if (x.cls === 'grader') return `${x.grader.fn} correct answers rejected · ${x.grader.models} models`;
  if (x.cls === 'ambiguous') {
    const [r1, r2] = x.amb.readings;
    return `<span class="mono">${show(r1.c)}</span> (${r1.models}) vs <span class="mono">${show(r2.c)}</span> (${r2.models})`;
  }
  return '';
}
function strength(x) {
  if (x.cls === 'key' || x.cls === 'review') return `post ${x.key.posterior >= 0.995 ? '>0.99' : x.key.posterior.toFixed(2)}`;
  if (x.cls === 'grader') return `${x.grader.fn}/${x.grader.fails} fails`;
  if (x.cls === 'ambiguous') return `p ${x.amb.p < 0.005 ? '<0.005' : x.amb.p.toFixed(3)}`;
  return '';
}

function renderScope() {
  const K = DATA.environment.rollouts;
  $('scopeText').innerHTML = `Audit of <b>${DATA.models.length}</b> published evaluations · <b>${DATA.items.length}</b> examples × <b>${K}</b> rollouts · <b>${(DATA.models.length * DATA.items.length * K).toLocaleString()}</b> rollouts`;
}

function renderKPIs() {
  const c = A.counts;
  const src = A.leak.domains.filter((d) => d.solutionSource).map((d) => d.domain);
  const retrievers = A.leak.perModel.filter((p) => p.retriever).length;
  const moves = DATA.models.map((m, i) => ({ m, d: A.pubRank[i] - A.audRank[i], i })).sort((p, q) => Math.abs(q.d) - Math.abs(p.d));
  const big = moves[0];
  const top = A.topModel;
  const kpi = (t, v, small, s, ic) => `<div class="kpi"><div class="t">${t}</div><div class="v">${v}${small ? `<small>${small}</small>` : ''}</div><div class="s">${s}</div><div class="ic"><i data-lucide="${ic}"></i></div></div>`;
  $('kpis').innerHTML =
    kpi('Examples flagged', A.flagged.length, `of ${DATA.items.length}`, `${c.key} answer key · ${c.grader} grader · ${c.ambiguous} ambiguous${c.review ? ` · ${c.review} review` : ''}`, 'flag') +
    kpi('Retrieved-solution rollouts', A.totalRetrieved.toLocaleString(), pct(A.totalRetrieved / A.totalRollouts), `${src.join(', ') || 'none'} · ${retrievers} model${retrievers === 1 ? '' : 's'}`, 'globe') +
    kpi('Rank changes after fixes', A.rankChanges, `of ${DATA.models.length} models`, big && big.d !== 0 ? `Largest: ${short(big.m.id)} #${A.pubRank[big.i]} → #${A.audRank[big.i]}` : 'No rank changes', 'arrow-down-up') +
    kpi('Frontier resolution', pct(A.audBoot[top].pTop, 0), 'P(#1 holds)', `SE(θ) ${A.seTop.toFixed(2)} at #1 · ${A.saturated} examples saturated`, 'gauge');
}

function renderFilters() {
  const c = A.counts;
  const opts = [['all', 'All', A.flagged.length], ['key', 'Answer key', c.key], ['grader', 'Grader', c.grader], ['ambiguous', 'Ambiguous', c.ambiguous], ['review', 'Review', c.review]];
  $('filters').innerHTML = opts.map(([k, l, n]) => `<button data-f="${k}" class="${filter === k ? 'on' : ''}">${l}<span class="n">${n}</span></button>`).join('');
  $('filters').querySelectorAll('button').forEach((b) => b.onclick = () => { filter = b.dataset.f; const r = flaggedRows(); if (r.length) selected = r[0].i; renderFilters(); renderFlagged(); renderInspector(); syncHeights(); });
}

const ORDER = { key: 0, grader: 1, ambiguous: 2, review: 3 };
function flaggedRows() {
  return A.flagged.filter((x) => filter === 'all' || x.cls === filter)
    .sort((p, q) => ORDER[p.cls] - ORDER[q.cls] || (q.key.posterior - p.key.posterior) || (q.grader.fn - p.grader.fn) || p.i - q.i);
}

function renderFlagged() {
  const rows = flaggedRows();
  if (!rows.length) { $('flagTbl').innerHTML = `<tbody><tr><td class="ev" style="padding:28px;text-align:center">No examples in this category at the current threshold.</td></tr></tbody>`; return; }
  if (selected === null || !A.flagged.some((x) => x.i === selected)) selected = rows[0].i;
  $('flagTbl').innerHTML = `<thead><tr><th>#</th><th>Example</th><th>Issue</th><th>Evidence</th></tr></thead><tbody>` +
    rows.map((x) => `<tr data-i="${x.i}" class="${x.i === selected ? 'sel' : ''}"><td class="idx">${x.i}</td><td class="q" title="${esc(DATA.items[x.i].question)}">${esc(DATA.items[x.i].question)}</td><td>${badge(x.cls)}</td><td class="ev" data-tip="${strength(x)}">${evidence(x)}</td></tr>`).join('') + '</tbody>';
  $('flagTbl').querySelectorAll('tbody tr').forEach((tr) => tr.onclick = () => { selected = +tr.dataset.i; renderFlagged(); renderInspector(); syncHeights(); });
}

function abilityPlot(x) {
  // per-model match rate of key and consensus vs ability, with the profiled logistic curves
  const W = 420, H = 132, L = 30, R = 8, T = 8, B = 22;
  const th = A.irt.theta;
  const lo = Math.min(...th) - 0.3, hi = Math.max(...th) + 0.3;
  const X = (t) => L + (t - lo) / (hi - lo) * (W - L - R);
  const Y = (p) => T + (1 - p) * (H - T - B);
  const rate = (c) => DATA.models.map((_, m) => { let k = 0, n = 0; DATA.rollouts[m][x.i].forEach((r, j) => { if (A.leak.retrieved[m][x.i][j]) return; n++; if (canonical(DATA.items[x.i].answers[r[0]]) === c) k++; }); return n ? k / n : 0; });
  const rk = rate(x.keyC), ra = rate(x.key.alt);
  const curve = (beta) => { let d = ''; for (let k = 0; k <= 40; k++) { const t = lo + (hi - lo) * k / 40; d += `${k ? 'L' : 'M'}${X(t).toFixed(1)},${Y(sigmoid(A.slope * (t - beta))).toFixed(1)}`; } return d; };
  let g = '';
  [0, 0.5, 1].forEach((p) => { g += `<line x1="${L}" x2="${W - R}" y1="${Y(p)}" y2="${Y(p)}" stroke="#1f1f1f"/><text x="${L - 6}" y="${Y(p) + 3.5}" text-anchor="end">${p}</text>`; });
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">model ability θ (IRT)</text>`;
  g += `<path d="${curve(x.key.betaKey)}" fill="none" stroke="${COLORS.key}" stroke-width="1.5" opacity="0.55"/>`;
  g += `<path d="${curve(x.key.betaAlt)}" fill="none" stroke="${COLORS.alt}" stroke-width="1.5" opacity="0.7"/>`;
  th.forEach((t, m) => {
    g += `<circle cx="${X(t)}" cy="${Y(rk[m])}" r="3.2" fill="${COLORS.key}" data-tip="${esc(DATA.models[m].id)}: matches key ${pct(rk[m], 0)}"/>`;
    g += `<circle cx="${X(t)}" cy="${Y(ra[m])}" r="3.2" fill="${COLORS.alt}" data-tip="${esc(DATA.models[m].id)}: matches consensus ${pct(ra[m], 0)}"/>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" class="chart">${g}</svg>`;
}

function matrix(x) {
  const rep = state.reps[x.i];
  const item = DATA.items[x.i];
  const color = (c) => {
    if (c === x.keyC) return COLORS.key;
    if (x.cls === 'key' || x.cls === 'review') return c === x.key.alt ? COLORS.alt : COLORS.other;
    if (x.cls === 'ambiguous') { const other = x.amb.readings.find((r) => r.c !== x.keyC); return other && c === other.c ? COLORS.alt2 : COLORS.other; }
    return COLORS.other;
  };
  const order = A.irt.theta.map((t, m) => [t, m]).sort((p, q) => q[0] - p[0]).map((z) => z[1]);
  let h = `<div class="matrix"><div></div><div class="th">r1</div><div class="th">r2</div><div class="th">r3</div><div class="th">r4</div><div class="th">θ</div>`;
  for (const m of order) {
    h += `<div class="mn" title="${esc(DATA.models[m].id)}">${esc(short(DATA.models[m].id))}</div>`;
    DATA.rollouts[m][x.i].forEach((r, k) => {
      const raw = item.answers[r[0]]; const c = canonical(raw);
      const ret = A.leak.retrieved[m][x.i][k];
      let col = color(c);
      if (x.cls === 'grader' && c === x.keyC && !r[1]) col = '#f59e0b';
      h += `<div class="cell ${ret ? 'ret' : ''}" style="background:${col}" data-tip="${esc(short(DATA.models[m].id))} r${k + 1}: ${esc(raw)} · reward ${r[1]}${ret ? ' · fetched solution source' : ''}"></div>`;
    });
    h += `<div class="th">${A.irt.theta[m].toFixed(2)}</div>`;
  }
  h += '</div>';
  const lg = [[COLORS.key, `reference <span class="mono">${esc(rep.get(x.keyC) ?? x.keyC)}</span>`]];
  if (x.cls === 'key' || x.cls === 'review') lg.push([COLORS.alt, `consensus <span class="mono">${esc(rep.get(x.key.alt) ?? x.key.alt)}</span>`]);
  if (x.cls === 'ambiguous') { const o = x.amb.readings.find((r) => r.c !== x.keyC); if (o) lg.push([COLORS.alt2, `second reading <span class="mono">${esc(rep.get(o.c) ?? o.c)}</span>`]); }
  if (x.cls === 'grader') lg.push(['#f59e0b', 'correct but rejected']);
  lg.push([COLORS.other, 'other answers']);
  return h + `<div class="ans-legend">${lg.map(([c, t]) => `<span><i class="sw" style="background:${c}"></i>${t}</span>`).join('')}<span><i class="sw" style="background:#555;box-shadow:inset 0 0 0 3px #555, inset 0 0 0 5px #000"></i>dot = fetched a solution source</span></div>`;
}

function renderInspector() {
  const x = A.items[selected];
  if (!x) { $('inspector').innerHTML = ''; return; }
  const item = DATA.items[x.i];
  const rep = state.reps[x.i];
  let detail = '';
  if (x.cls === 'key' || x.cls === 'review') {
    detail = `<div class="ins-sec"><div class="ins-label"><span>Who matches which answer</span><span>slope ${A.slope.toFixed(2)}</span></div>${abilityPlot(x)}
      <div class="kv" style="margin-top:8px"><span class="k">P(reference is wrong)</span><span class="v r">${x.key.posterior >= 0.995 ? '> 0.99' : x.key.posterior.toFixed(3)}</span>
      <span class="k">log likelihood ratio</span><span class="v">${x.key.logLR.toFixed(2)}</span>
      <span class="k">ability gradient, consensus</span><span class="v g">${x.key.gradAlt.toFixed(2)}</span>
      <span class="k">ability gradient, reference</span><span class="v ${x.key.gradKey < 0 ? 'r' : ''}">${x.key.gradKey.toFixed(2)}</span></div></div>`;
  } else if (x.cls === 'grader') {
    const ex = x.grader.examples.slice(0, 2).map((e) => `<div class="raw"><span class="dim">${esc(short(DATA.models[e.model].id))} · reward 0</span><br>${esc(e.raw)}</div>`).join('');
    detail = `<div class="ins-sec"><div class="ins-label"><span>Rejected answers that normalize to the reference</span><span class="mono">${x.grader.fn} / ${x.grader.fails}</span></div>${ex}
      <div class="kv" style="margin-top:10px"><span class="k">reference</span><span class="v">${esc(item.key)}</span>
      <span class="k">numbers per rejected answer</span><span class="v">${x.grader.meanNumbers.toFixed(1)}</span>
      <span class="k">numbers in reference</span><span class="v">1</span>
      <span class="k">models affected</span><span class="v">${x.grader.models}</span></div></div>`;
  } else if (x.cls === 'ambiguous') {
    detail = `<div class="ins-sec"><div class="ins-label"><span>Stable readings among the strongest models</span><span class="mono">V ${x.amb.V.toFixed(2)}</span></div>
      <div class="kv">${x.amb.readings.map((r) => `<span class="k">${r.c === x.keyC ? 'reference reading' : 'other reading'} <span class="dim">${esc(rep.get(r.c) ?? r.c)}</span></span><span class="v">${r.models} models</span>`).join('')}
      <span class="k">permutation p (300 shuffles)</span><span class="v">${x.amb.p.toFixed(3)}</span>
      <span class="k">mean within-model consistency</span><span class="v">${pct(x.amb.meanConsistency, 0)}</span></div></div>`;
  }
  $('inspector').innerHTML = `
    <div class="ins-head"><div class="meta">${badge(x.cls)}<span>Example #${x.i}</span><span>·</span><span>${esc(item.category)}</span></div>
      <div class="qtext">${esc(item.question)}</div></div>
    <div class="ins-sec"><div class="ins-label"><span>Answers by model and rollout</span><span>sorted by ability</span></div>${matrix(x)}</div>
    ${detail}
    <div class="ins-sec"><div class="kv"><span class="k">pass rate (published)</span><span class="v">${pct(x.passRate)}</span>
      <span class="k">discrimination a</span><span class="v ${x.a < 0.3 ? 'r' : ''}">${x.a.toFixed(2)}</span>
      <span class="k">item-rest correlation</span><span class="v ${x.rest < 0 ? 'r' : ''}">${x.rest.toFixed(2)}</span></div>
      <div style="margin-top:10px"><span class="link" data-tip="Opens Prime Traces filtered to example #${x.i}"><i data-lucide="list-tree"></i>Open ${DATA.models.length * DATA.environment.rollouts} rollouts in Traces</span></div></div>`;
  icons();
}

function ciBar(lo, hi, mean, min, max) {
  const W = 110, X = (v) => 4 + (v - min) / (max - min) * (W - 8);
  return `<svg width="${W}" height="14"><line x1="4" x2="${W - 4}" y1="7" y2="7" stroke="#1f1f1f"/><line x1="${X(lo)}" x2="${X(hi)}" y1="7" y2="7" stroke="${'#855eee'}" stroke-width="3" stroke-linecap="round" opacity="0.55"/><circle cx="${X(mean)}" cy="7" r="3.3" fill="#a78bfa"/></svg>`;
}

function renderLeaderboard() {
  const K = DATA.environment.rollouts;
  const order = A.audBoot.map((b, m) => [b.mean, m]).sort((p, q) => q[0] - p[0]).map((z) => z[1]);
  const min = Math.min(...A.audBoot.map((b) => b.lo)) - 0.02, max = Math.max(...A.audBoot.map((b) => b.hi)) + 0.02;
  $('lbSub').textContent = `Published average reward vs. re-graded with the fixes above: re-keyed or both-readings examples, normalized grading, and retrieved-solution rollouts scored 0. Intervals are 90% bootstrap over examples (500 resamples).`;
  $('lbTbl').innerHTML = `<thead><tr><th>#</th><th>Model</th><th>Samples</th><th>Avg Reward</th><th>Audited</th><th>90% CI</th><th>Rank</th><th>Retrieved</th><th style="text-align:right">P(#1)</th></tr></thead><tbody>` +
    order.map((m) => {
      const d = A.pubRank[m] - A.audRank[m];
      const dl = d > 0 ? `<span class="delta-up">▲ ${d}</span>` : d < 0 ? `<span class="delta-down">▼ ${-d}</span>` : '<span class="delta-0">–</span>';
      const lp = A.leak.perModel[m];
      const b = A.audBoot[m];
      return `<tr><td class="idx">${A.audRank[m]}</td><td>${modelCell(DATA.models[m].id)}</td><td class="ev">${DATA.items.length} examples × ${K} rollouts</td>
        <td class="num">${A.pubBoot[m].mean.toFixed(3)} <span class="dim" style="font-size:12px">#${A.pubRank[m]}</span></td>
        <td class="num strong">${b.mean.toFixed(3)}</td>
        <td><div class="ci">${ciBar(b.lo, b.hi, b.mean, min, max)}<span class="ev mono">${b.lo.toFixed(3)}–${b.hi.toFixed(3)}</span></div></td>
        <td>${dl}</td>
        <td class="ev">${lp.retriever ? `<span class="badge b-src">${lp.retrieved} · ${pct(lp.rate)}</span>` : '<span class="delta-0">–</span>'}</td>
        <td class="num" style="text-align:right">${b.pTop >= 0.005 ? pct(b.pTop, 0) : '<span class="delta-0">&lt;1%</span>'}</td></tr>`;
    }).join('') + '</tbody>';
}

function renderSources() {
  const rows = A.leak.domains.slice().sort((p, q) => (q.or === Infinity ? 1e9 : q.or) - (p.or === Infinity ? 1e9 : p.or));
  $('srcTbl').innerHTML = `<thead><tr><th>Domain</th><th style="text-align:right">Visits</th><th style="text-align:right">OR</th><th style="text-align:right">95% CI</th></tr></thead><tbody>` +
    rows.map((d) => `<tr><td><span class="mono" style="font-size:11.5px">${esc(d.domain)}</span>${d.solutionSource ? ' <span class="badge b-src" style="margin-left:4px">Source</span>' : ''}</td><td class="num" style="text-align:right">${d.visits.toLocaleString()}</td>
      <td class="num" style="text-align:right;${d.solutionSource ? 'color:#c4b5fd;font-weight:600' : ''}">${fmt(d.or, 2)}</td><td class="num dim" style="text-align:right">${fmt(d.lo, 2)}–${fmt(d.hi, 2)}</td></tr>`).join('') + '</tbody>';
}

function renderInfo() {
  const el = $('infoChart');
  const W = Math.max(260, Math.floor(el.getBoundingClientRect().width) || 340), H = 214, L = 34, R = 10, T = 10, B = 40;
  const xs = A.grid, ys = A.info;
  const ymax = Math.ceil(Math.max(...ys) / 10) * 10;
  const X = (t) => L + (t - xs[0]) / (xs[xs.length - 1] - xs[0]) * (W - L - R);
  const Y = (v) => T + (1 - v / ymax) * (H - T - B);
  let g = '<defs><linearGradient id="ig" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#855eee" stop-opacity="0.45"/><stop offset="1" stop-color="#855eee" stop-opacity="0.04"/></linearGradient></defs>';
  for (let k = 0; k <= 4; k++) { const v = ymax * k / 4; g += `<line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="#1f1f1f"/><text x="${L - 6}" y="${Y(v) + 3.5}" text-anchor="end">${Math.round(v)}</text>`; }
  [-3, -2, -1, 0, 1, 2, 3].forEach((t) => { g += `<text x="${X(t)}" y="${H - B + 15}" text-anchor="middle">${t}</text>`; });
  g += `<text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">ability θ</text>`;
  const line = xs.map((t, k) => `${k ? 'L' : 'M'}${X(t).toFixed(1)},${Y(ys[k]).toFixed(1)}`).join('');
  g += `<path d="${line}L${X(xs[xs.length - 1])},${Y(0)}L${X(xs[0])},${Y(0)}Z" fill="url(#ig)"/><path d="${line}" fill="none" stroke="#855eee" stroke-width="2"/>`;
  const tx = X(A.topTheta);
  g += `<line x1="${tx}" x2="${tx}" y1="${T}" y2="${Y(0)}" stroke="#a3a3a3" stroke-dasharray="3 3" opacity="0.6"/>`;
  A.irt2.theta.forEach((t, m) => { g += `<circle cx="${X(Math.max(-3, Math.min(3, t)))}" cy="${Y(0) - 0.5}" r="3.5" fill="${m === A.topModel ? '#fafafa' : '#525252'}" stroke="#000" data-tip="${esc(DATA.models[m].id)} · θ ${t.toFixed(2)}"/>`; });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${g}</svg>`;
  $('infoSub').textContent = `Fisher information of the audited examples by ability. Dots are models; the dashed line is ${short(DATA.models[A.topModel].id)}, where SE(θ) is ${A.seTop.toFixed(2)}.`;
}

function renderSurprises() {
  const rows = A.surprises.slice(0, 7);
  $('surTbl').innerHTML = `<thead><tr><th>Model</th><th>Ex.</th><th style="text-align:right">P(pass)</th><th>Fetched</th></tr></thead><tbody>` +
    rows.map((s) => {
      const doms = DATA.domains.filter((_, d) => (s.mask & (1 << d)) && A.leak.domains[d].solutionSource);
      return `<tr data-i="${s.i}"><td class="mono" style="font-size:11.5px">${esc(short(DATA.models[s.m].id))}</td><td class="idx">#${s.i}</td><td class="num" style="text-align:right">${s.P.toFixed(3)}</td>
        <td>${doms.length ? `<span class="badge b-src">${esc(doms[0])}</span>` : '<span class="delta-0">none</span>'}</td></tr>`;
    }).join('') + '</tbody>';
  $('surTbl').querySelectorAll('tbody tr').forEach((tr) => tr.onclick = () => {
    const i = +tr.dataset.i; if (A.items[i].cls !== 'ok') { selected = i; filter = 'all'; renderFilters(); renderFlagged(); renderInspector(); syncHeights(); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  });
}

function syncHeights() {
  // the flagged list scrolls inside a card exactly as tall as the inspector
  const ins = $('inspector'), card = document.querySelector('.flagged'), wrap = card.querySelector('.tbl-wrap');
  wrap.style.maxHeight = 'none'; wrap.style.height = 'auto';
  const used = wrap.getBoundingClientRect().top - card.getBoundingClientRect().top + 16;
  const h = Math.max(320, ins.getBoundingClientRect().height - used);
  wrap.style.height = h + 'px';
}

function renderAll() {
  renderScope(); renderKPIs(); renderFilters(); renderFlagged(); renderInspector(); renderLeaderboard(); renderSources(); renderInfo(); renderSurprises(); icons();
  syncHeights();
}

// tooltips
document.addEventListener('mousemove', (e) => {
  const t = e.target.closest('[data-tip]');
  const tip = $('tip');
  if (!t) { tip.style.display = 'none'; return; }
  tip.textContent = t.getAttribute('data-tip');
  tip.style.display = 'block';
  tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, e.clientX + 12) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
});

async function main() {
  DATA = await (await fetch('data/runs.json')).json();
  $('envName').textContent = DATA.environment.name;
  $('envOwner').textContent = DATA.environment.owner;
  compute();
  renderAll();
  let timer = null;
  $('keyT').oninput = (e) => {
    keyT = +e.target.value; $('keyTv').textContent = keyT.toFixed(2);
    clearTimeout(timer); timer = setTimeout(() => { compute(); renderAll(); }, 120);
  };
  $('rerun').onclick = () => {
    const ic = $('rerun').querySelector('svg'); if (ic) { ic.style.transition = 'transform .5s'; ic.style.transform = 'rotate(360deg)'; }
    setTimeout(() => { compute(); renderAll(); }, 30);
  };
  window.addEventListener('resize', () => { renderInfo(); syncHeights(); });
}
main();
