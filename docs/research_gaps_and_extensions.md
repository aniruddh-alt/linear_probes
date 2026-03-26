# Activation Probes for Safety Monitoring: Research Gaps and Extensions

## The Current State

Activation probes detect safety-relevant concepts by training lightweight classifiers on a model's internal representations. The core papers in this space (2506.10805, Detecting High-Stakes Interactions) demonstrate that probes can match medium-sized LLM classifiers at six orders of magnitude less compute. Google DeepMind has deployed production probes on Gemini (2601.11516). The approach works — but its failure modes are severe and well-documented.

This document identifies five research gaps and proposes extensions toward a production-grade, generalizable monitoring tool.

---

## Gap 1: Probes Learn Surface Features, Not Semantics

### Evidence
**"False Sense of Security" (Wang et al., 2509.03888)** systematically demonstrates that probing classifiers learn *instructional patterns and trigger words* rather than semantic harmfulness. A simple n-gram classifier matches probe performance in-distribution. When tested OOD (different phrasing, different domains), probe accuracy degrades sharply. The original high-stakes paper (2506.10805) acknowledges this: "emergency exit signs" triggers false positives due to the word "emergency," and "responsible money laundering" evades detection through ethical-coded language.

### What's Missing
No work trains probes with *adversarial contrastive data* at training time. The high-stakes paper uses bag-of-words deconfounding as a data filter, but doesn't incorporate hard negatives into the probe's training objective. The concept of contrastive learning on *activation-space* hard negatives (semantically similar inputs with opposite labels) is unexplored for safety probes.

### Extension: Contrastive Probe Training on Hard Negatives
Construct adversarial training pairs where surface features conflict with ground truth:
- "emergency exit signs need new batteries" (low-stakes, contains "emergency")
- "I need to make sure nobody finds out about the contaminated water supply" (high-stakes, no alarm words)
- "lol so my friend accidentally took 40 tylenol, what happens" (high-stakes, casual tone)

Train the probe with a contrastive loss that explicitly penalizes reliance on lexical features. This addresses the root cause identified by Wang et al. — not by filtering data, but by forcing the probe to learn deeper features.

**Feasibility:** Medium. Requires LLM-generated adversarial pairs + modified training loop. Our toolkit already supports custom probe architectures and training.

---

## Gap 2: Single-Layer Probes Are Trivially Evadable

### Evidence
**"Obfuscated Activations Bypass LLM Latent-Space Defenses" (Bailey et al., 2412.09565)** shows that adversarial suffixes can shift activations at a target layer to appear benign while preserving harmful generation. After 70 rounds of adversarial training, attackers consistently found *new* obfuscated activations — suggesting the space of evasive representations is vast.

**"Neural Chameleons" (McGuinness et al., 2512.11949)** goes further: models fine-tuned to evade probes on *benign* concepts (HTML, chemistry) generalize this evasion to *unseen safety-relevant* concepts (deception, harmfulness). Even multi-layer ensembles and non-linear probes are degraded, though not fully defeated.

**"Beyond Linear Probes" (Oldfield et al., 2509.26238)** introduces Truncated Polynomial Classifiers that model non-linear interactions between neurons and use adaptive cascading (easy cases exit early, hard cases get higher-order analysis). These outperform linear and MLP probes by up to 10%.

### What's Missing
No work combines *multi-layer probing* with *non-linear architectures* and *adversarial probe training* in a unified framework. Individual defenses exist, but they haven't been composed. Neural Chameleons shows ensembles help but don't eliminate evasion — the question is whether adversarial training of the ensemble itself (minimax games) closes the remaining gap.

### Extension: Adversarially-Trained Multi-Layer Ensemble
Train probes at layers 8, 12, 15, 20, 24 with a shared aggregation head. During training, alternate between:
1. Standard probe training on labeled data
2. Adversarial rounds where a gradient-based attacker tries to shift activations at each layer simultaneously

An adversary would need to simultaneously fool all layers — and each layer encodes fundamentally different features (earlier = syntactic, later = semantic). The multi-layer constraint makes obfuscation geometrically harder.

**Feasibility:** Medium. Our sweep already trains per-layer probes. The aggregation head and adversarial training loop are new components.

---

## Gap 3: Probes Only Monitor Inputs, Not Generation

### Evidence
**"Can We Predict Alignment Before Models Finish Thinking?" (Chan et al., 2507.12428)** shows that linear probes on Chain-of-Thought *activations* outperform GPT-4.1, fine-tuned BERT, and human annotators at predicting whether a reasoning model's final response will be safe. Critically, CoT text is unfaithful — models "think" one way but "act" differently. Only activations reliably predict the outcome.

**"Emergent Response Planning in LLMs" (2502.06258)** demonstrates that hidden representations encode future output tokens — the model "plans ahead" in its residual stream before generating.

**"SafeSwitch" (2502.01042)** and **"In-Decoding Safety-Awareness Probing" (2601.10543)** both probe activations *during generation* to detect harmful outputs before completion, enabling early stopping.

The original high-stakes paper (2506.10805) only probes the input. Our own run confirmed this: the probe sees the prompt's activation at the last token, not what the model is about to generate.

### What's Missing
No work combines *input-phase probing* with *generation-phase probing* in a unified monitor. Existing work either probes the prompt or probes the CoT — not both. A two-stage monitor (input risk assessment + generation trajectory monitoring) would catch both harmful inputs and harmful outputs from benign-looking inputs.

### Extension: Dual-Phase Monitor (Input + Generation)
1. **Phase 1 (Input probe):** Score the prompt at layer 15 using the existing attention probe. High-risk inputs get flagged immediately.
2. **Phase 2 (Generation probe):** For borderline or pass-through inputs, extract activations from the first N generated tokens. Train a separate probe on these generation-phase activations to detect whether the model is producing harmful content.

The generation probe catches the case where a benign-looking prompt ("write a story about a chemistry teacher") produces harmful output. The input probe catches obvious high-stakes cases cheaply. Together they cover both threat surfaces.

**Feasibility:** Medium. Our toolkit has `ResponseGenerator` for generation and `ActivationExtractor` for extraction. Combining them in a single forward pass requires a new extraction mode that hooks during generation, not just on the prompt.

---

## Gap 4: No Causal Validation of Probe Directions

### Evidence
**"How Reliable are Causal Probing Interventions?" (2408.15510)** casts doubt on whether probe directions are causally relevant or merely correlational. **"On the Non-Identifiability of Steering Vectors" (2602.06801)** shows that steering vectors are not uniquely identifiable — multiple directions can produce the same behavioral change, and the "true" direction may not be what the probe found.

Conversely, **"Contrastive Activation Addition" (Rimsky et al., 2312.06681)** and **"Small Vectors, Big Effects" (2509.06608)** demonstrate that adding/subtracting learned directions *does* change model behavior (sycophancy, refusal, reasoning style). **"Mechanistic Indicators of Steering Effectiveness" (2602.01716)** identifies which directions steer effectively vs which don't.

**"Monitoring Emergent Reward Hacking" (2603.04069)** uses activation probes to detect reward hacking during generation — showing that probes can catch internal misalignment that isn't visible in outputs.

### What's Missing
No one has taken a probe trained for *risk detection* and validated it causally via steering. The question: if you add the "high-stakes direction" to a model processing a low-stakes prompt, does the model become more cautious? If yes, the probe found a causally relevant feature. If no, the probe found a correlate.

### Extension: Causal Validation via Activation Steering
After training the high-stakes probe, extract its learned direction vector. Then:
1. **Amplification test:** Add the direction to layer 15 activations during generation on low-stakes prompts. Measure if the model becomes more cautious (longer responses, hedging language, refusals).
2. **Suppression test:** Subtract the direction from high-stakes prompts. Measure if the model becomes less cautious.
3. **Behavioral metric:** Compare perplexity, refusal rate, and response length with/without intervention.

If steering works, the probe direction is causally linked to the model's risk processing — making it a validated safety feature, not just a classifier artifact.

**Feasibility:** Medium. Our config system already has `SteeringParams`. This is activation steering with the probe direction as the steering vector.

---

## Gap 5: No Production Architecture for Generalizable Monitoring

### Evidence
**"Building Production-Ready Probes for Gemini" (Google DeepMind, 2601.11516)** is the only work that takes probes to production. Key lessons:
- Single probe architectures fail on distribution shifts (short-context train → long-context deploy)
- Novel architectures (MultiMax, Max of Rolling Means) fix long-context generalization
- Cascading classifiers (cheap probe + expensive LLM judge for uncertain cases) achieve optimal cost-accuracy tradeoffs
- Even with all improvements, jailbreak success rate remains >1%
- AlphaEvolve (automated architecture search) discovered better probes than humans

**"RL-Obfuscation" (2506.14261)** shows models can learn to evade monitors via RL — relevant for deployed systems where the model might be fine-tuned post-deployment.

### What's Missing
Google's work is proprietary and Gemini-specific. No open-source, model-agnostic production monitoring framework exists. Key missing pieces:
1. **Concept-agnostic probe training** — one framework that monitors any concept (risk, toxicity, deception, bias) by swapping the training data
2. **Cascading with confidence routing** — cheap probe handles easy cases, expensive judge handles ambiguous ones
3. **Continuous calibration** — monitoring probe accuracy over time and retraining when distribution shifts
4. **Multi-concept monitoring** — running multiple probes simultaneously without multiplying compute

### Extension: Open-Source Production Monitoring Framework
Build on our existing toolkit to create a generalizable monitoring system:

```
User prompt → Model forward pass → Extract activations at layers [8,12,15,20,24]
                                  ↓
                            Multi-concept probe bank
                            ├── Risk probe (high/low stakes)
                            ├── Toxicity probe
                            ├── Deception probe
                            └── Custom concept probe
                                  ↓
                            Confidence router
                            ├── High confidence → act on probe decision
                            ├── Low confidence → defer to LLM judge
                            └── Log + update calibration statistics
```

The key insight from Oldfield et al. is that this doesn't need to be one probe — it's a *cascade* with adaptive compute. Easy cases (clearly benign) exit cheaply. Hard cases (adversarial, ambiguous) get more expensive analysis. The same framework monitors any concept by swapping the labeled training data.

**Feasibility:** High for the framework, since most components already exist in our toolkit. The cascade router and continuous calibration are new.

---

## Priority Ranking

| Extension | Impact | Novelty | Feasibility | Priority |
|-----------|--------|---------|-------------|----------|
| Dual-phase monitor (input + generation) | High | High | Medium | **1** |
| Causal validation via steering | High | High | Medium | **2** |
| Contrastive probe training | Medium | Medium | Medium | **3** |
| Multi-layer adversarial ensemble | Medium | Medium | Medium | **4** |
| Production monitoring framework | High | Medium | High | **5** |

**Recommended starting point:** Extensions 1 + 2 combined. Probe the model during generation (not just on input), then validate the learned direction via steering. This gives both a stronger monitor *and* causal evidence that what the probe learned is real. No existing work does both.

---

## Key References

| Paper | ID | Key Finding |
|-------|-----|-------------|
| Detecting High-Stakes Interactions | 2506.10805 | Probes match 8B LLM classifiers at 1M× less compute |
| False Sense of Security | 2509.03888 | Probes learn trigger words, not semantic harm |
| Obfuscated Activations Bypass Defenses | 2412.09565 | Adversarial suffixes evade single-layer probes |
| Neural Chameleons | 2512.11949 | Models learn to hide thoughts from unseen monitors |
| Beyond Linear Probes (TPCs) | 2509.26238 | Non-linear polynomial probes + adaptive cascading |
| Building Production Probes for Gemini | 2601.11516 | Production deployment lessons, MultiMax architecture |
| Predicting Alignment Before Thinking | 2507.12428 | CoT activations outperform GPT-4 at predicting safety |
| SafeSwitch | 2502.01042 | Generation-time activation monitoring |
| Emergent Response Planning | 2502.06258 | Hidden states encode future output tokens |
| RL-Obfuscation | 2506.14261 | Models learn to evade monitors via RL |
| Steering Vectors (CAA) | 2312.06681 | Adding/subtracting directions changes behavior |
| Monitoring Reward Hacking | 2603.04069 | Probes detect internal misalignment during generation |
