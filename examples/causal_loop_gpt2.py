"""End-to-end causal loop on gpt2 (CPU): extract -> direction -> ablate -> measure.

Demonstrates the full sonde stack closing the loop, with the specificity
control the intervention design requires:

  * train a diff-of-means direction for a concept (weather vs food),
  * confirm it linearly separates held-out activations (baseline AUROC),
  * ablate THAT direction in a forward pass -> the residual's projection onto it
    collapses to ~0,
  * ablate a RANDOM direction (matched norm) -> the projection is preserved.

The contrast between the real-direction and random-direction ablations is a
mechanism sanity check: the write removes the targeted component and leaves an
unrelated (near-orthogonal) direction essentially untouched. It is NOT full
behavioural specificity — that requires an off-target capability-preservation
measure (docs/intervention_design.md §7.5 control #2) and, for refusal
specifically, an instruction-tuned model + GPU. gpt2 is a base model with no
"refusal" behaviour.

Run:  python examples/causal_loop_gpt2.py
"""

from __future__ import annotations

import torch
from nnterp import StandardizedTransformer

from sonde.directions import DiffMeansEstimator, evaluate_projection
from sonde.interventions import InterventionContext

MODEL = "openai-community/gpt2"
LAYER = 6

WEATHER = [
    "The forecast calls for heavy rain and strong winds tomorrow.",
    "A cold front is moving in, bringing snow to the mountains.",
    "It was a sunny afternoon with a gentle breeze and clear skies.",
    "Thunderstorms are expected across the region this evening.",
    "The humidity made the summer heat feel even more oppressive.",
    "Fog rolled in off the coast, reducing visibility on the road.",
    "Temperatures will drop below freezing overnight with light frost.",
    "The hurricane intensified as it approached the warm coastal waters.",
    "Bright sunshine followed the morning drizzle by midday.",
    "A heatwave is forecast to last through the end of the week.",
    "Sleet and freezing rain are coating the streets tonight.",
    "Clear blue skies stretched over the valley all weekend.",
]
FOOD = [
    "She simmered the tomato sauce with garlic, basil, and olive oil.",
    "The bakery's fresh sourdough had a crisp, golden crust.",
    "He grilled the salmon and served it with roasted vegetables.",
    "A bowl of spicy ramen with a soft-boiled egg hit the spot.",
    "They shared a platter of cheese, cured meats, and crackers.",
    "The chocolate cake was rich, moist, and topped with ganache.",
    "I marinated the chicken in lemon, herbs, and yogurt overnight.",
    "Steaming dumplings arrived with a tangy soy dipping sauce.",
    "The chef plated the risotto with shaved parmesan and truffle.",
    "Crispy tacos filled with seasoned beef and fresh salsa.",
    "A warm slice of apple pie with a scoop of vanilla ice cream.",
    "The curry was fragrant with cumin, coriander, and coconut milk.",
]


def last_token_acts(model, prompts, layer):
    """Extract the last-token residual at `layer` for each prompt (no ablation)."""
    rows = []
    for p in prompts:
        with model.trace(p):
            h = model.layers_output[layer].save()
        rows.append(h[0, -1])
    return torch.stack(rows)


def last_token_acts_ablated(model, prompts, layer, vector, factor=1.0):
    """Extract last-token residuals while ablating `vector` at `layer`."""
    ctx = InterventionContext(model).add_steering(
        layers=[layer], vector=vector, mode="project_subtract", factor=factor
    )
    rows = []
    for p in prompts:
        with model.trace(p):
            ctx.apply()
            h = model.layers_output[layer].save()
        rows.append(h[0, -1])
    return torch.stack(rows)


def main() -> dict[str, float]:
    torch.manual_seed(0)
    model = StandardizedTransformer(MODEL, device_map="cpu")

    prompts = WEATHER + FOOD
    labels = torch.tensor([1] * len(WEATHER) + [0] * len(FOOD))
    # Simple, deterministic split: alternating train/test within each class.
    train_idx = [i for i in range(len(prompts)) if i % 2 == 0]
    test_idx = [i for i in range(len(prompts)) if i % 2 == 1]

    acts = last_token_acts(model, prompts, LAYER)
    direction = (
        DiffMeansEstimator()
        .fit(acts[train_idx], labels[train_idx], key=f"layers_output:{LAYER}")
        .direction
    )

    def mean_abs_proj(x: torch.Tensor) -> float:
        return (x @ direction).abs().mean().item()

    base_auroc = evaluate_projection(acts[test_idx], labels[test_idx], direction)[
        "auroc"
    ]
    base_proj = mean_abs_proj(acts[test_idx])

    # Ablate the trained direction -> the residual component along it vanishes.
    abl = last_token_acts_ablated(model, prompts, LAYER, direction)
    abl_proj = mean_abs_proj(abl[test_idx])

    # Control: ablate a random matched-norm direction -> the component survives.
    rand = torch.randn_like(direction)
    rand = rand / rand.norm()
    ctrl = last_token_acts_ablated(model, prompts, LAYER, rand)
    ctrl_proj = mean_abs_proj(ctrl[test_idx])
    ctrl_auroc = evaluate_projection(ctrl[test_idx], labels[test_idx], direction)[
        "auroc"
    ]

    print("=" * 64)
    print("Causal loop on gpt2 — weather vs food, layer", LAYER)
    print("=" * 64)
    print("Primary metric: mean |projection onto the concept direction|")
    print(f"  baseline (no ablation):            {base_proj:8.4f}")
    print(f"  ablate trained direction:          {abl_proj:8.4f}  (collapses ~0)")
    print(f"  ablate random direction (control): {ctrl_proj:8.4f}  (preserved)")
    print("-" * 64)
    print("Decodability (AUROC of that projection vs label):")
    print(f"  baseline:                          {base_auroc:.3f}")
    print(f"  control (random ablation):         {ctrl_auroc:.3f}")
    print("  (ablating the trained direction leaves ~0 projection, so its AUROC")
    print("   is numerically degenerate — the projection-magnitude row is the")
    print("   honest measure.)")
    print("-" * 64)
    print("Mechanism check: ablating the trained direction removes the concept's")
    print("linear component; ablating a random (near-orthogonal) direction does not.")
    print("This is NOT behavioural specificity — see docs/intervention_design.md §7.5.")
    return {
        "base_proj": base_proj,
        "ablated_proj": abl_proj,
        "control_proj": ctrl_proj,
        "base_auroc": base_auroc,
        "control_auroc": ctrl_auroc,
    }


if __name__ == "__main__":
    main()
