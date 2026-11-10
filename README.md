# Eval Audit for the Prime Intellect Environments Hub (concept demo)

**Live demo:** https://mohith26.github.io/prime-intellect-eval-audit/

An **Audit** tab for an environment page on the Environments Hub. It reads every published evaluation of the environment (models x examples x rollouts, which the platform already stores) and asks four questions automatically:

1. **Is the reference answer wrong?** The strongest models agree on a different answer, and that answer gets *more* common as model ability rises, while the key is matched mostly by weaker models.
2. **Is the grader rejecting correct answers?** Rollouts scored 0 whose normalized final answer equals the reference (for example `FY2014 (fiscal year 2014)` rejected because it contains two numbers).
3. **Is the task ambiguous?** Strong models split into two answers, and each model sticks to its own reading across rollouts.
4. **Did rollouts retrieve the published solutions?** Fetching a domain raises the odds of reward *within the same model and example*, and only for some models.

It then re-grades the leaderboard with the fixes (re-key, accept normalized matches, accept both readings, score retrieved-solution rollouts as 0), shows bootstrap intervals and the probability each model is really #1, and reports how much information the cleaned benchmark still has at the frontier.

This is the statistical version of the manual audit in Prime Intellect's [Learnings from Auditing Public Evaluations](https://www.primeintellect.ai/blog/auditing-public-evaluations) (Nov 3, 2026): brittle graders, ambiguous prompts, and models reading leaked solutions.

> Unofficial concept demo by Mohith Gajjela. Not affiliated with Prime Intellect. **All runs are synthetic** (seeded simulator below). Model names are used only to make the leaderboard look realistic; the scores say nothing about the real models.

## Why it is useful

Every published run on the Hub already contains the evidence: per-rollout answers, rewards, and tool calls. One model's run can't tell you the key is wrong; fourteen models' runs can. Auditing by hand means reading traces example by example. This ranks the examples worth reading, with the reason, and shows what the leaderboard would look like after the fixes.

## How it works (`core/audit.mjs`)

| Step | Method |
|---|---|
| Solution sources | Mantel-Haenszel odds ratio of reward given that a rollout fetched domain *d*, stratified by (model, example), Robins-Breslow-Greenland 95% CI. A domain is a source if the lower bound is above 2. A model is attributed retrieval only if its fetch rate is far above the cross-model median (exact binomial tail, Bonferroni) *and* fetching raises its own odds of reward, so occasional benign visits by other models don't count. |
| Ability and difficulty | 2PL item response model, MAP with alternating Newton steps, on non-retrieved rollouts. Discrimination is allowed to go negative. |
| Grader false negatives | Failing rollouts whose canonical answer (final number, 6 significant digits) equals the canonical reference. Flag if at least 3 rollouts from at least 2 models. |
| Wrong reference | Likelihood ratio between "truth = reference" and "truth = top non-reference answer of the strongest half of models". Under each hypothesis a model matches the truth with probability sigmoid(slope (theta - beta)), beta profiled out on a grid; wrong answers follow their smoothed empirical frequencies; rollouts tempered by 0.5 for within-model correlation; prior 2%. |
| Ambiguity | Among the strongest half of models, at least two answers that are each the stable modal answer of at least 2 models, one of them the reference; Cramer's V of model x answer with a 300-shuffle permutation p-value (p < 0.01). |
| Leaderboard | Re-graded scores, item-clustered bootstrap (500 resamples): 90% intervals and P(rank 1). |
| Saturation | Refit 2PL on the re-graded data; test information curve, SE of theta at the top model, examples solved in every rollout by the top 5. |

Everything on the page is computed in the browser from `data/runs.json` by this module; nothing is hard-coded. The threshold slider re-runs the whole audit.

## Data

`node scripts/generate.mjs` (seed 20261103) simulates a document-grounded QA environment (`fiscal-docs-qa`, 240 examples, 14 models, 4 rollouts each = 13,440 rollouts) with abilities, difficulties, per-model answer verbosity and per-example correlated rollouts. It plants 9 wrong keys, 8 brittle-grader examples, 7 ambiguous examples, and two models that fetch the solutions from `huggingface.co` / `modelscope.cn` when they don't know the answer. `data/truth.json` holds the planted issues; the audit never reads it.

## Tests

```
node --test test/audit.test.mjs     # 11 tests
node scripts/multiseed.mjs 20       # detection quality across 20 fresh seeds
```

Tests cover answer normalization, Mantel-Haenszel against a hand-computed value, 2PL parameter recovery on a known matrix, model ordering, leak attribution, planted-issue recovery, the wrong-key ability-gradient signature, multi-seed precision, leaderboard error, and bootstrap determinism.

Results across 20 seeds (not the demo seed):

| | |
|---|---|
| Wrong reference recovered | 148/180 (82%) |
| Brittle grader recovered | 160/160 (100%) |
| Ambiguous recovered | 96/140 (69%) |
| Clean examples falsely flagged | 26/4320 (0.6%) |
| Solution-source domains identified exactly | 20/20 |
| Leaderboard mean abs. error vs. latent accuracy | 0.038 published -> 0.005 audited |

The demo seed happens to be a clean one (9/9, 8/8, 5/7, no false flags); the table above is the honest average.

## Limitations

- Synthetic data. Real Hub runs have free-form answers that need a better canonicalizer than "final number" (units, dates, lists, code), and judge-graded environments need the judge's verdict in the same matrix.
- 14 models is a small panel. The wrong-key test needs several strong models; on a benchmark that every model fails, it cannot tell a wrong key from a shared mistake and stays quiet.
- Ambiguity recall is the weakest (69%): when only one strong model takes the second reading, there is nothing to separate from noise.
- Retrieval attribution uses fetched domains only. Leaks through search snippets, other APIs, or another model's rollouts on a public rollout API need trace-content features (the verifiers monitor is the right place for that).
- Assumes rollouts were run with comparable settings; mixed versions should be audited per environment version.

## Novelty

Checked the Prime Intellect changelog (through Oct 23, 2026), blog (through Nov 3, 2026), docs (`llms-full.txt`: no IRT, item analysis, grader-error or contamination tooling), the hosted-evaluations docs and the Environments Hub evaluation pages: the Hub shows leaderboards, per-sample viewers and reward histograms, and the Nov 3 audit was done by hand. The statistics build on published research: IRT/measurement-theory item flagging ("Fantastic Bugs and Where to Find Them in AI Benchmarks", NeurIPS 2025 D&B; Land and Bikel, "Auditing LLM Benchmarks with IRT", EMNLP 2026) and LLM transcript scanners (Transluce Docent, UK AISI Inspect Scout). What is new here is putting it inside the Hub as a built-in tab for free-form, multi-rollout agentic evals: an ability-aware wrong-key test, grader false negatives by normalization, within-model stability for ambiguity, stratified retrieval attribution from tool-call domains, and a re-graded leaderboard with uncertainty.

## Files

- `index.html`, `styles.css`, `app.js`: the page (IBM Plex Sans/Mono and Lucide icons, matching the Prime Intellect app; provider icons from lobe-icons, MIT)
- `core/audit.mjs`: the audit (pure functions)
- `core/simulate.mjs`, `scripts/generate.mjs`: seeded data
- `test/audit.test.mjs`, `scripts/multiseed.mjs`: tests and evaluation
