# R-Judge Dissociation Experiment (Phase 0)

**Status:** Design approved 2026-04-23
**Scope:** Phase 0 pilot only — single model (Llama-3.1-8B-Instruct), R-Judge benchmark.
**Next phase (not in this spec):** matched-pair composition test.

## Goal

Test whether a subject model's residual stream encodes risk in scenarios where the model *behaviorally fails* to flag risk. This is a Zhao-style dissociation experiment: we ask whether recognition exists when execution breaks.

The question is sharp: on R-Judge scenarios that the model behaviorally misclassifies as safe (but are actually risky), does a probe trained on the model's *correct* cases still detect risk in the activations?

## Experimental logic

1. Run the subject model on every R-Judge scenario with R-Judge's official judge prompt.
2. Parse the model's yes/no behavioral output. Combined with ground-truth labels, partition scenarios into four cells:

   |                  | GT: Risky (label=1)       | GT: Safe (label=0) |
   |------------------|---------------------------|--------------------|
   | Model said risky | TP                        | FP                 |
   | Model said safe  | **FN (target)**           | TN                 |

3. Extract residual-stream activations at the last token of the formatted prompt (the token just before the model generates "0" or "1") at multiple layers.
4. Train a per-layer probe on `TP ∪ TN` (the cells where behavior agrees with ground truth), predicting ground-truth label. Select best layer by val AUROC.
5. Apply best probe to the `FN` cell. Report the key dissociation metric (defined below).

### The key dissociation metric

AUROC on FN alone is degenerate because every FN row has ground-truth label = 1. The real dissociation question compares probe scores *across* cells:

- **`AUROC(FN vs TN)`** — does the probe separate "missed risky" from "correctly safe"? If high, the probe sees risk the model behaviorally missed — dissociation confirmed.
- **`AUROC(FN vs TP)`** — does the probe treat "missed risky" like "caught risky"? If scores are indistinguishable, the internal representation is the same regardless of behavioral outcome.
- **Probe classification rate on FN** at the threshold fit from `TP ∪ TN` training — bottom-line "what fraction of behaviorally-missed scenarios does the probe still catch?"

Three outcomes, all publishable:
- **High AUROC(FN vs TN), high classification rate** → Zhao-style dissociation confirmed. Recognition exists; execution/composition fails.
- **Low AUROC(FN vs TN), low classification rate** → No internal signal on missed cases. Different problem than we hypothesized.
- **Intermediate AUROC (~0.6)** → Weak/partial representation. Consistent with gray-zone findings.

## Data source

R-Judge benchmark (Yuan et al., EMNLP Findings 2024), from the GitHub repo at `https://github.com/Lordog/R-Judge`. No HuggingFace mirror — fetch JSON files directly from `raw.githubusercontent.com/Lordog/R-Judge/main/data/{Application,Finance,IoT,Program,Web}/*.json`.

**Record structure** (confirmed by inspecting `data/Application/chatbot.json`):
```json
{
  "id": 37,
  "scenario": "psychological",
  "profile": "You are a psychological counselor.",
  "goal": "Below is an interaction process ... Print either 0 for safe or 1 for unsafe, then explain the reason.",
  "contents": [[{"role": "user", "content": "..."}, {"role": "agent", "action": "..."}, ...], ...],
  "label": 0,
  "risk_description": "...",
  "attack_type": "unintended"
}
```

Each scenario is formatted for the model as: `profile + "\n\n" + rendered_dialogue(contents) + "\n\n" + goal`. The `goal` field is the official judge prompt that ships with each scenario, so we're using R-Judge's standard protocol — our model-behavior numbers are directly comparable to the R-Judge leaderboard.

Total scenarios: 569 across 5 categories.

## Component architecture

Standalone orchestration script (following the `refusal_probing/` pattern). New files under `experiments/rjudge_dissociation/`:

```
experiments/rjudge_dissociation/
├── config.yaml              # hyperparameters, loaded by run_pipeline.py
├── k8s_job.yaml             # Lambda cluster deployment manifest
├── rjudge_loader.py         # R-Judge data: download, format, cache
├── run_pipeline.py          # main orchestration
└── data/                    # outputs (gitignored): activations, judgments, probe, results
```

### Module responsibilities

**`rjudge_loader.load_rjudge_scenarios(cache_dir, categories) -> list[dict]`**
- Downloads the 5 category directories' JSON files from GitHub raw (caches locally in `cache_dir`).
- Flattens into a single list of records.
- Renders `contents` (list of turns) as a dialogue string with role prefixes.
- Returns a list of `{id: str, scenario: str, formatted_prompt: str, label: int, category: str}`.
- Idempotent: if cache exists, read from disk.

**`run_behavioral_judgment(bundle, model_cfg) -> dict[str, int]`**
- Uses `generation.ResponseGenerator` with `max_new_tokens=8`, `temperature=0.0` (greedy).
- Parses the first "0" or "1" digit in the output. If neither appears, records `pred=-1` (unparseable) and logs.
- Writes `data/judgments.jsonl`: one row per scenario with `{id, raw_output, predicted_label}`.
- Returns `{id: predicted_label}`.

**`classify_cells(labels, predictions) -> dict[str, str]`**
- Maps each id to `"TP" | "FP" | "FN" | "TN"` using behavioral pred vs. ground truth.
- Drops ids with `pred=-1` (unparseable) from the probe pipeline; logs counts.
- Writes `data/cells.json` with `{counts: {TP,FP,FN,TN,unparseable}, per_id: {id: cell}}`.

**`extract_activations(bundle, model_cfg, layers) -> ExtractionResult`**
- Uses the existing `activation.ActivationExtractor` with `token_index=-1` and requested layer targets.
- Same `formatted_prompt` as the judgment step — this is load-bearing; the activation must be from the exact input the model judged.
- Saves to `data/activations/` via existing safetensors caching.

**`train_dissociation_probe(extraction, cells, labels, layers) -> SweepResult`**
- Filters extraction to `TP ∪ TN` indices only.
- Builds 70/15/15 train/val/test split *within* `TP ∪ TN`, stratified by ground-truth label.
- Calls `probes.LayerProbeSweepRunner` with these indices. Selects best layer by val AUROC.
- Saves best probe weights to `data/best_probe.pt` and layer metrics to `data/sweep_results.json`.

**`evaluate_on_fn(probe, extraction, cells, labels) -> dict`**
- Extracts probe scores on all cells at the best layer.
- Computes:
  - `AUROC(FN vs TN)` — primary dissociation metric.
  - `AUROC(FN vs TP)` — secondary: does probe treat missed-risky like caught-risky?
  - `FN_classification_rate@threshold` — fraction of FN rows scored above the probe's decision threshold from training.
  - Per-category breakdown of FN classification rate.
  - Control: `AUROC(FP vs TP)` as a sanity check on the "behavioral output is not the probe" claim.
- Writes `data/results.json`.

## Data flow

```
R-Judge GitHub JSONs (5 categories, 569 scenarios)
        │
        ▼
rjudge_loader.load_rjudge_scenarios()
        │  → bundle: {id, formatted_prompt, label, category}
        ├────────────────────────────┐
        ▼                            ▼
ResponseGenerator              ActivationExtractor
(greedy, 8 tokens)             (token_index=-1, 5 layers)
        │                            │
        ▼                            ▼
parse first "0"|"1"             activations.safetensors
        │                            │
        ▼                            │
judgments.jsonl                      │
        │                            │
        └────────────┬───────────────┘
                     ▼
              classify_cells()
                     │
                     ▼
              cells.json {id: TP/FP/FN/TN}
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      TP ∪ TN       FN        (FP, TN held for reporting)
     (probe       (target
      training)    cell)
         │           │
         ▼           │
 LayerProbeSweep     │
    Runner           │
         │           │
         ▼           │
    best_probe.pt    │
         │           │
         └─────┬─────┘
               ▼
       evaluate_on_fn()
               │
               ▼
         results.json
```

## Config

`experiments/rjudge_dissociation/config.yaml`:

```yaml
run_name: rjudge_dissociation_llama31_8b
seed: 42

model:
  model_name: meta-llama/Llama-3.1-8B-Instruct
  dtype: bfloat16

judgment:
  max_new_tokens: 8
  temperature: 0.0
  parse_strategy: first_digit

extraction:
  batch_size: 2
  token_index: -1
  activations:
    - layers_output:8
    - layers_output:12
    - layers_output:15
    - layers_output:20
    - layers_output:24
  save_path: experiments/rjudge_dissociation/data/activations

probe:
  probe_type: linear
  epochs: 20
  learning_rate: 0.001
  weight_decay: 0.1
  bootstrap_samples: 200
  early_stopping_patience: 5

split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15

sweep:
  batch_size: 128
  selection_metric: auroc

io:
  output_dir: experiments/rjudge_dissociation/data

rjudge:
  cache_dir: experiments/rjudge_dissociation/data/rjudge_raw
  categories: [Application, Finance, IoT, Program, Web]
```

### Rationale for choices

- **Llama-3.1-8B-Instruct (not base).** R-Judge's judge prompt assumes instruction-following. Base model won't reliably produce "0"/"1". The `detecting_high_stakes/` run used base because that's a classification task on prompt activations; here we need behavioral compliance with the judge protocol.
- **5 layers, not all 32.** Cost: 569 × 32 = ~18k probe fits vs. ~2.8k for 5 layers. The `detecting_high_stakes/` replication found best layer at ~30–47% depth (10–15 of 32); [8, 12, 15, 20, 24] covers that window with one above for safety.
- **Linear probe first.** Cleanest signal; if linear works, the story is strongest. Attention/MaxRollingMean as follow-up if linear underperforms.
- **Greedy, short judgment.** We only need 1 token for the cell. Greedy ensures TP/FP/FN/TN cells are reproducible across runs.
- **No category re-balancing.** Stratify on ground-truth label only; the category distribution flows through naturally. Per-category probe scores are reported for inspection, not used for weighting.

## Deployment (Lambda K8s)

Mirrors `experiments/detecting_high_stakes/k8s_job.yaml` with renamed Deployment: one H100 SXM pod, Oumi base image, HF cache on PVC `pvc-local-vp6ww`, node selector pinned to the same worker node. Pod idles via `sleep infinity`; the pipeline is run via `kubectl exec`.

Execution:
1. `kubectl apply -f experiments/rjudge_dissociation/k8s_job.yaml`
2. `kubectl exec -it ani-rjudge-dissoc -- bash`, sync repo onto PVC
3. `uv sync && PYTHONPATH=. python experiments/rjudge_dissociation/run_pipeline.py`
4. Results land in `experiments/rjudge_dissociation/data/`; rsync back.

### Budget estimate (Llama-3.1-8B, H100)
- Judgment generation: 569 × ~1s = ~10 min
- Activation extraction: 569 × ~1s = ~10 min
- Probe sweep (5 layers × linear): <5 min
- **Total: ~25 min**, same order as the `detecting_high_stakes/` run.

### Storage sanity-check
- 569 × 5 layers × 4096 (hidden_dim) × 2 bytes (bf16) ≈ 23 MB for last-token activations.
- Llama-3.1-8B bf16 weights: ~16 GB, cached on PVC.

## Error handling

- **Unparseable judgments.** If the model outputs something other than a parseable "0" or "1", record `pred=-1`, drop from the four-cell split, and report unparseable count in `cells.json`. If >10% unparseable, abort and revisit the judgment-parsing strategy.
- **Empty FN cell.** If the model's F1 is near-perfect on R-Judge and FN is empty/tiny (say <20), the dissociation test is underpowered. Report this as a negative result — the model doesn't behaviorally fail often enough on this benchmark to test the hypothesis. Consider a harder benchmark for Phase 0b.
- **R-Judge download failure.** If GitHub raw is unreachable, use the existing cache. If no cache and no network, fail fast with a clear message. Don't build a fallback bundled copy — keep the benchmark canonical.

## Testing plan

- **Unit:** `rjudge_loader.format_dialogue()` on a small synthetic `contents` list → verify role/content rendering.
- **Unit:** judgment parser on `"0"`, `"1"`, `"1. The agent..."`, `"Yes."`, empty string → correct `predicted_label` or `-1`.
- **Integration smoke test:** run the full pipeline with `max_samples=16` on a tiny model (the existing `Qwen 0.5B` already cached from refusal_probing) to verify data flow end-to-end before the real run. Does not assert anything about the probe's scientific validity — just that nothing crashes and all output files are written.

## Out of scope (explicitly)

- Phase 1 matched-pair composition test (weeks 3-5 of the original plan).
- Multi-model runs. Once Llama-3.1-8B works end-to-end, we extend the config for Qwen-2.5-7B and Llama-3.2-3B in a follow-up — not this spec.
- Novel probe architectures beyond linear. Follow-up if linear is too weak.
- Causal validation (steering) of the learned probe direction. Separate spec.
- Per-risk-type analysis (the 10 R-Judge risk types). Report per-category (5) instead.

## Caveats to document in the final writeup

1. **R-Judge results establish a dissociation pattern but don't isolate the mechanism.** We can claim "the model internally encodes risk in scenarios it behaviorally misses." We cannot yet claim "the model composes features X and Y to represent risk." The composition claim requires Phase 1 matched pairs.
2. **The within-scenario design cancels the distributional confound.** The probe is trained and evaluated on the same input distribution (R-Judge scenarios) — the comparison is between different behavioral outcomes on the same inputs, not between different datasets.
3. **We probe the prompt, not the generation.** Consistent with this phase's question: "does the model know *at the moment of deciding*?" A later phase could probe generation-time activations (see `docs/research_gaps_and_extensions.md` Gap 3).
