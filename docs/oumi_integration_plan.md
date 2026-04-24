# Project rename + Oumi integration plan

Status: **proposal — awaiting user sign-off.** Nothing destructive has been
applied. This doc captures the rename options, the dep/structural compat audit
between this repo (`sonde`) and `oumi`, and a phased integration plan.

---

## 1. Naming

### Top recommendation: **`oumi-lens`**

- Distribution name: `oumi-lens` · Python package: `oumi_lens`
- Fits the established mech-interp "lens" lineage (logit-lens, tuned-lens,
  prisma, future-lens) so researchers immediately know what kind of tool this is.
- Plays well with the rest of the Oumi family (`oumi`, `oumi-mcp`, `oumi-lens`).
- Doesn't overpromise: "lens" covers extraction, probes, steering, and any
  future intervention work — not just probes.

### Runners-up

| Name | Distribution | Pros | Cons |
| --- | --- | --- | --- |
| `oumi-scope` | `oumi-scope` | Microscope analogy, distinct from lens family | Slight overlap with profiling tools ("scope" = telescope/observability) |
| `oumi-strata` | `oumi-strata` | "Layers" metaphor; unique | Less self-explanatory; researchers won't know what it is from the name |
| `oumi-probe` | `oumi-probe` | Direct, descriptive | "Probe" is overloaded (PEFT probes, hardware probes, profiling probes) |

### Names rejected

- `linear-lens` — implies linear-only, but the toolkit ships attention,
  softmax, and rolling-mean probes today and will host non-linear tools later.
- `oumi-interp` — too generic; "interpretability" is broad.
- `oumi-loom` — already taken by janus' Loom interface.
- `oumi-insight` — collides with `nnsight`.

**Recommendation: `oumi-lens` unless you object.** Decision blocks the rest.

---

## 2. Dependency compatibility audit

### Versions

| Package | `sonde` | `oumi` | Verdict | Action |
| --- | --- | --- | --- | --- |
| Python | `>=3.10` | `>=3.10,<3.15` | ✅ Compatible | Tighten `oumi-lens` to `>=3.10,<3.15`. |
| `omegaconf` | `>=2.3.0` | `==2.4.0.dev4` (pinned dev) | ✅ Within range | Tighten to `==2.4.0.dev4` to match oumi exactly. |
| `datasets` | `>=3.0.0` | `>=3.2,<4.8.5` | ✅ Within range | Tighten to oumi's range. |
| `safetensors` | unpinned | `>=0.6,<0.8` | ✅ | Pin to oumi's range. |
| `torch` | unpinned | `>=2.6,<2.11.0` | ✅ | Pin to oumi's range. |
| `transformers` | (transitive via `nnterp`) | `>=4.57,<5.6` (override `>=5.5,<5.6`) | ⚠️ Verify | `nnterp` 1.2.x supports transformers ≥4.40; need to confirm it works on the override `>=5.5,<5.6`. |
| `typing_extensions` | unpinned | unpinned | ✅ | No change. |
| `torchmetrics` | unpinned | not listed | ➕ New dep for oumi | Add to extras group. |
| `einops` | unpinned | not listed | ➕ New dep for oumi | Add to extras group. |
| `nnterp` | `>=1.2.2` | not listed | ➕ New dep | Add to extras group; pulls in `nnsight`. |
| `nnsight` | (transitive) | not listed | ➕ New dep | Transitive; document. |
| `matplotlib` | `>=3.10.8` | not listed | ➕ Optional | Move to a `[viz]` extras subgroup. |

### Risk: `transformers` upper bound

Oumi pins `transformers>=5.5,<5.6` via `[tool.uv] override-dependencies`.
`nnterp` and `nnsight` historically lag transformers releases by 1–2 versions.
Before merging `oumi-lens` into oumi proper, we must:

1. Pin `nnterp` to a version known to support transformers 5.5.x.
2. Add a CI smoke test that imports `nnterp.StandardizedTransformer` and runs
   one extraction against a tiny model under oumi's transformers pin.

If upstream `nnterp` doesn't yet support transformers 5.5, the integration
ships in `experimental/` (no install-by-default) until it does.

### `ModelParams` overlap

Both projects define a `ModelParams` dataclass. Field comparison:

| Field | `sonde` | `oumi` |
| --- | --- | --- |
| `model_name` | `str = ""` | `str = MISSING` |
| `revision` | ✅ | ❌ (uses `model_kwargs`) |
| `device` | ✅ | ✅ (richer: `device_map`, `model_max_length`) |
| `dtype` | `str` whitelist | ✅ (richer: torch_dtype string with helper) |
| `load_in_8bit` / `load_in_4bit` | ✅ | ❌ (uses `quantization_config` instead) |
| `attn_implementation` | ✅ | ✅ |
| `trust_remote_code` | ✅ | ✅ |
| `output_attentions/hidden_states` | ✅ | ❌ |
| `adapter_model` | ❌ | ✅ |
| `tokenizer_*` | ❌ | ✅ (rich) |
| `processor_kwargs` | ❌ | ✅ (VLM support) |

**Implication.** When `oumi-lens` lands inside oumi, it should consume
`oumi.core.configs.params.model_params.ModelParams` directly and adapt its own
extractor to that surface — eliminating the duplicate dataclass. The current
`sonde.ModelParams` only uses 6 of its 11 fields when constructing
`nnterp.StandardizedTransformer`, so the migration is small.

### Config base class overlap

| Aspect | `sonde.BaseConfig` | `oumi.core.configs.BaseConfig` / `BaseParams` |
| --- | --- | --- |
| Backbone | OmegaConf-only | OmegaConf + custom validation pipeline |
| API | `from_yaml`, `from_dict`, `to_yaml`, `to_yaml_str` | `from_yaml`, `from_args`, `to_yaml`, `finalize_and_validate` |
| Validation | Per-field via `__post_init__` | `__finalize_and_validate__` recursive walk |

`oumi.BaseParams` is a strict superset. Migration plan: replace
`sonde.BaseConfig` parents with `oumi.core.configs.params.BaseParams`,
rename `__post_init__` validation hooks to `__finalize_and_validate__`. ~10
files to touch, mechanical.

---

## 3. Structural integration: where does it live in oumi?

Oumi's `src/` ships **two** top-level packages today, both auto-discovered by
setuptools:

```
src/oumi/...           # main, supported package
src/experimental/...   # experimental holding pen (ships in wheel as a top-level package)
```

`experimental/`'s `__init__.py` is explicit:
> Warning: features in this package are experimental and may be subject to
> significant changes or removal without notice.

### Three placement options

#### Option A — `src/oumi/lens/` (first-class)

```
src/oumi/lens/
├── __init__.py
├── activation/
├── probes/
├── token_selectors.py
├── trainers/
└── ...
```

- Import: `from oumi.lens import ActivationExtractor, build_probe`
- Name: `oumi.lens` reads naturally; aligns with `oumi-lens` distribution name.
- Lifecycle: bound to oumi's release cadence and stability bar.
- **Best fit if** the goal is "ship as a real oumi capability".

#### Option B — `src/experimental/lens/` (incubator)

- Import: `from experimental.lens import ...` (awkward — `experimental` is
  a top-level name, not under `oumi`).
- Lifecycle: free to break weekly; explicit "use at your own risk".
- **Best fit if** the goal is "stake the claim, iterate, then promote".

#### Option C — Sub-project under `projects/oumi-lens/` (sibling repo)

```
projects/
├── oumi/
└── oumi-lens/      # workspace member, depends on oumi as a library
```

- Independent `pyproject.toml`, independent versioning, independent CI.
- Imported as its own package: `import oumi_lens`.
- **Best fit if** we want strong independence and the option to publish to PyPI
  separately. Closest to the current `sonde` reality.

### Recommendation

**Phased: B → A.**

1. **Phase 1 (this PR).** Create `src/experimental/lens/` in the oumi repo on a
   branch. Drop in just a `__slim__` skeleton: `__init__.py` + `INTEGRATION_PLAN.md`
   + an `extras` group `oumi[lens]` in `pyproject.toml`. **No code copy yet.**
   This stakes the namespace and proves the pyproject changes don't break oumi.
2. **Phase 2 (follow-up PR).** Copy the runtime modules under
   `src/experimental/lens/` (or vendor them as the standalone `oumi-lens`
   package builds). Wire one CI smoke test.
3. **Phase 3 (post-stabilization).** Promote `experimental/lens/` →
   `oumi/lens/` once: (a) deps are stable on oumi's transformers pin,
   (b) `BaseParams` migration is done, (c) `oumi.lens` API is locked.

Keep this `sonde` repo alive throughout phases 1–2 as the primary
research workspace; it becomes a thin shim in phase 3.

---

## 4. Concrete diffs queued for sign-off

### A. In **this repo** (`sonde`):

1. `pyproject.toml`:
   - `name = "oumi-lens"`
   - `description = "Mechanistic interpretability toolkit for the Oumi ecosystem — activation extraction, probing, steering."`
   - `[project.scripts] interp = "cli.main:entrypoint"` → `oumi-lens = "cli.main:entrypoint"`
   - Tighten `omegaconf`, `datasets`, `safetensors`, `torch` ranges to match oumi.
2. Primer relocated to `docs/linear-probes-primer.md` as part of the rebrand to sonde.
3. **Defer** internal module-name rename (`activation/` → `oumi_lens/activation/`)
   to a follow-up; that change touches every test and import. Keeping
   modules at top-level for now is fine because the `src/oumi/lens/` mirror
   inside oumi will use proper namespacing from day one.

### B. In the **oumi** repo (new branch `aniruddh-alt/lens-experimental`):

1. Create `src/experimental/lens/` with:
   - `__init__.py` (one-liner pointing at the standalone repo).
   - `INTEGRATION_PLAN.md` (mirror of this doc, scoped).
2. `pyproject.toml`: add `lens = ["nnterp>=1.2.2", "einops", "torchmetrics"]`
   under `[project.optional-dependencies]`.
3. No code copy yet, no `nnterp` import in `__init__.py`. Smoke test optional.

Both diffs run through normal review before commit.

---

## 5. Open questions

- **Name confirmation.** Sticking with `oumi-lens`?
- **Placement.** Phase-1 incubator (`experimental/lens/`) or jump straight to
  `oumi/lens/`?
- **Branch base.** You're currently on `aniruddh-alt/agent-deterministic-environment`
  with uncommitted changes in `oumi/`. Branch off `main` for the lens work, or
  off your current branch?
- **`nnterp`/transformers.** Want me to verify the latest `nnterp` works under
  `transformers>=5.5,<5.6` before scaffolding the extras group?
