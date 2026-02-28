# Refusal Probing Example

End-to-end example: build a probing dataset for refusal behavior, extract activations, and train linear probes.

## Pipeline

```
Step 1: Generate prompts        (Oumi synth)
   ↓
Step 2: Generate responses      (Toolkit - target model)
   ↓
Step 3: Label responses         (Oumi synth - LLM-as-judge)
   ↓
Step 4: Extract activations     (Toolkit)
   ↓
Step 5: Train probes            (Toolkit)
```

## Running

### Step 1 — Generate diverse harmful + harmless prompts
```bash
oumi synth -c examples/refusal_probing/1_generate_prompts.yaml
```

### Step 2 — Generate target model responses
```bash
python -m cli.main run -c examples/refusal_probing/generate_responses.yaml
```

### Step 3 — Label responses as refusal / non-refusal
```bash
oumi synth -c examples/refusal_probing/2_label_responses.yaml
```
