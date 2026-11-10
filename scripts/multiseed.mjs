// Detection quality across seeds (recall per planted class, false flags on clean examples,
// leaderboard error vs the simulator's latent accuracy). Usage: node scripts/multiseed.mjs 10
import { generate } from '../core/simulate.mjs';
import { runAudit } from '../core/audit.mjs';
const S = Number(process.argv[2] ?? 10);
const map = { wrong_key: 'key', brittle_grader: 'grader', ambiguous: 'ambiguous' };
const tot = { wrong_key: [0, 0], brittle_grader: [0, 0], ambiguous: [0, 0] };
let clean = 0, cleanN = 0, maePub = 0, maeAud = 0, src = 0;
for (let s = 1; s <= S; s++) {
  const { data, truth } = generate(1000 + s);
  const A = runAudit(data, { bootstrap: 200 });
  for (const [kind, cls] of Object.entries(map)) {
    const planted = Object.entries(truth.issues).filter(([, v]) => v === kind).map(([k]) => +k);
    tot[kind][0] += planted.filter((i) => A.items[i].cls === cls).length; tot[kind][1] += planted.length;
  }
  A.items.forEach((x) => { if (!truth.issues[x.i]) { cleanN++; if (x.cls !== 'ok') clean++; } });
  const mae = (b) => data.models.reduce((acc, m, mi) => acc + Math.abs(b[mi].mean - truth.trueAccuracy[m.id]), 0) / data.models.length;
  maePub += mae(A.pubBoot); maeAud += mae(A.audBoot);
  const flagged = A.leak.domains.filter((d) => d.solutionSource).map((d) => d.domain).sort().join(',');
  if (flagged === 'huggingface.co,modelscope.cn') src++;
}
const out = Object.fromEntries(Object.entries(tot).map(([k, [h, n]]) => [k, `${h}/${n} (${(100 * h / n).toFixed(0)}%)`]));
out.clean_false_flags = `${clean}/${cleanN} (${(100 * clean / cleanN).toFixed(2)}%)`;
out.leaderboard_mae_published = (maePub / S).toFixed(4);
out.leaderboard_mae_audited = (maeAud / S).toFixed(4);
out.solution_sources_exact = `${src}/${S}`;
console.log(JSON.stringify(out, null, 1));
