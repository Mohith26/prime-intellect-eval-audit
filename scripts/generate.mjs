// Writes data/runs.json (what the platform already stores for published evals:
// models, examples, reference answers, per-rollout answer/reward/fetched domains)
// and data/truth.json (planted issues; used only by tests and the README table).
import { writeFileSync, mkdirSync } from 'node:fs';
import { generate } from '../core/simulate.mjs';
const seed = Number(process.argv[2] ?? 20261103);
const { data, truth } = generate(seed);
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
writeFileSync(new URL('../data/runs.json', import.meta.url), JSON.stringify(data));
const { leaked, ...rest } = truth;
writeFileSync(new URL('../data/truth.json', import.meta.url), JSON.stringify(rest, null, 1));
console.log(`seed ${seed}: ${data.models.length} models x ${data.items.length} examples x ${data.environment.rollouts} rollouts`);
