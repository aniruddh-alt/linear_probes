# R-Judge Dissociation Experiment

Phase 0 pilot testing whether Llama-3.1-8B-Instruct internally encodes risk in R-Judge scenarios it *behaviorally* misclassifies as safe.

**Question:** Does the model know when it gets it wrong?

**Design spec:** `docs/superpowers/specs/2026-04-23-rjudge-dissociation-design.md`

## What this does

1. Loads R-Judge (569 agent-safety scenarios, 5 categories, EMNLP Findings 2024).
2. Runs Llama-3.1-8B-Instruct on each scenario with R-Judge's official judge prompt; saves the full (256-token) response to `responses.jsonl`.
3. **LLM-as-judge labeling** (Oumi synth + Claude Sonnet): reads each verbose/refusal response and extracts the intended safe / unsafe classification into `labeled.jsonl`.
4. Partitions scenarios into four cells by {behavioral label} × {ground truth}: TP, FP, FN, TN.
5. Extracts residual-stream activations at the last prompt token at layers [8, 12, 15, 20, 24] using the same chat-template-formatted inputs the judgment step saw.
6. Trains a linear probe on `TP ∪ TN` (model-agreement cells), 70/15/15 split, selects best layer by val AUROC.
7. Applies best probe to the `FN` cell — reports the dissociation metrics.

**Requires:** `ANTHROPIC_API_KEY` in the environment for step 3. Cost: ~\execute.50 per 571-scenario run using Claude Sonnet 4.

## How to run

### On Lambda K8s (recommended — takes ~25 min on 1x H100)

```bash
# 1. Bring up the pod
kubectl apply -f experiments/rjudge_dissociation/k8s_job.yaml

# 2. Exec in, sync the repo onto the PVC, run the pipeline
kubectl exec -it deploy/ani-rjudge-dissoc -- bash
cd /data/ani && git clone <this-repo> sonde  # or rsync
cd /data/ani/sonde
uv sync
PYTHONPATH=. python experiments/rjudge_dissociation/run_pipeline.py

# 3. Rsync results back
kubectl cp ani-rjudge-dissoc:/data/ani/sonde/experiments/rjudge_dissociation/data ./data
```

### Locally (needs a GPU with ~20GB VRAM for Llama-3.1-8B bf16)

```bash
uv sync
PYTHONPATH=. python experiments/rjudge_dissociation/run_pipeline.py
```

## Outputs

All written to `experiments/rjudge_dissociation/data/` (gitignored):

| File | Contents |
|------|----------|
| `rjudge_raw/*/combined.json` | Cached per-category R-Judge JSONs (first run only downloads). |
| `activations/activations.safetensors` | Last-prompt-token activations at 5 layers, 569 scenarios. |
| `activations/activations_manifest.pt` | Activation metadata + sample ids + labels. |
| `responses.jsonl` | Subject-model responses to each R-Judge scenario (full 256-token output). |
| `labeled.jsonl` | Oumi-synth output: responses + `safety_label` in {safe, unsafe, unclear} from the Claude judge. |
| `cells.json` | `{counts, per_id}` from the 4-cell classification. |
| `best_probe.pt` | Weights of the best-layer linear probe. |
| `results.json` | Summary: cells, sweep best layer, test metrics, controls, and the dissociation numbers. |

## How to read results

The key numbers in `results.json["dissociation"]`:

| Field | Meaning |
|-------|---------|
| `auroc_fn_vs_tn` | **Primary dissociation metric.** High → probe separates "missed risky" from "correctly safe" → model internally encodes risk even when behavior misses. |
| `auroc_fn_vs_tp` | Secondary. If ≈ 0.5, probe treats missed-risky indistinguishably from caught-risky → strong dissociation. |
| `fn_classification_rate` | Fraction of FN rows the probe classifies as risky at its training threshold. Bottom-line "what fraction did the probe catch that the model missed?" |
| `per_category` | Same metric broken down by R-Judge category (Application/Finance/IoT/Program/Web). |
| `auroc_fp_vs_tp` | Sanity control: if probe just echoes behavior, this would be high. If low, probe is doing its own thing. |

## Known caveats

- R-Judge results establish the dissociation pattern but don't isolate mechanism. See Phase 1 (matched-pair composition test) for the mechanism claim.
- The within-scenario design cancels the distributional confound flagged by Wang et al. 2509.03888.
- We probe the prompt activation, not the generation. A follow-up can probe during generation (see `docs/research_gaps_and_extensions.md` Gap 3).
