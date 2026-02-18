import torch
from torch.utils.data import DataLoader

from activation import ActivationExtractor
from configs import ActivationConfig, ModelConfig
from dataset import ProbingDataset, ProbingSampleBuilder
from probes import BinaryLinearProbeTrainer, ProbeTrainConfig

TRUE_STATEMENTS = [
    "The capital of France is Paris.",
    "Water freezes at 0 degrees Celsius.",
    "The Earth orbits the Sun.",
    "Humans have 46 chromosomes.",
    "The speed of light is approximately 300,000 km/s.",
    "Mount Everest is the tallest mountain on Earth.",
    "The Great Wall of China is in Asia.",
    "Shakespeare wrote Hamlet.",
    "The Pacific Ocean is the largest ocean.",
    "DNA carries genetic information.",
    "2 + 2 equals 4.",
    "The chemical symbol for gold is Au.",
    "Mammals are warm-blooded.",
    "The Amazon is the longest river in South America.",
    "The moon causes tides on Earth.",
    "Penguins live in the Southern Hemisphere.",
    "Electrons have a negative charge.",
    "The human heart has four chambers.",
    "Python is a programming language.",
    "The Eiffel Tower is in Paris.",
    "Lions are carnivores.",
    "Photosynthesis produces oxygen.",
    "Rome is the capital of Italy.",
    "The sun is a star.",
    "Bees pollinate flowers.",
]

FALSE_STATEMENTS = [
    "The capital of Japan is London.",
    "Water freezes at 100 degrees Celsius.",
    "The Sun orbits the Earth.",
    "Humans have 23 chromosomes.",
    "The speed of light is approximately 100 km/s.",
    "Mount Kilimanjaro is the tallest mountain on Earth.",
    "The Great Wall of China is in Europe.",
    "Dickens wrote Hamlet.",
    "The Atlantic Ocean is the largest ocean.",
    "RNA carries all genetic information in humans.",
    "2 + 2 equals 5.",
    "The chemical symbol for gold is Gd.",
    "Reptiles are warm-blooded.",
    "The Nile is the longest river in South America.",
    "The sun causes tides on Earth.",
    "Penguins live in the Northern Hemisphere.",
    "Electrons have a positive charge.",
    "The human heart has two chambers.",
    "Java is not a programming language.",
    "The Eiffel Tower is in Berlin.",
    "Lions are herbivores.",
    "Photosynthesis consumes oxygen.",
    "Madrid is the capital of Italy.",
    "The moon is a star.",
    "Bees only eat honey.",
]


def main() -> None:
    records = [
        {"id": f"true-{i}", "text": text, "label": 1}
        for i, text in enumerate(TRUE_STATEMENTS)
    ] + [
        {"id": f"false-{i}", "text": text, "label": 0}
        for i, text in enumerate(FALSE_STATEMENTS)
    ]
    bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")

    model_config = ModelConfig(model_name="EleutherAI/pythia-70m")
    activation_config = ActivationConfig(
        model_config=model_config,
        save_path="artifacts/activations",
        activations=[f"layers_output:{i}" for i in range(6)],
        batch_size=8,
        token_index=-1,
        to_cpu=True,
    )
    extractor = ActivationExtractor(activation_config)
    result = extractor.extract(bundle)

    print("Model info:", extractor.info)
    for name, tensor in result["activations"].items():
        print(
            f"{name}: samples={int(tensor.shape[0])} tensor_shape={tuple(tensor.shape)}"
        )

    probing_dataset = ProbingDataset.from_extraction_path(
        "artifacts/activations_manifest.pt",
        activation_key="layers_output:5",
    )
    train_size = int(0.8 * len(probing_dataset))
    val_size = len(probing_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        probing_dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    trainer = BinaryLinearProbeTrainer(
        input_dim=probing_dataset[0][0].numel(),
        config=ProbeTrainConfig(epochs=30, learning_rate=0.01),
    )
    history = trainer.fit(train_loader, val_loader=val_loader)
    metrics = trainer.evaluate(val_loader)

    print(
        f"\nTraining complete ({len(probing_dataset)} samples, {train_size} train / {val_size} val)"
    )
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")
    print(f"Final val acc:    {history['val_accuracy'][-1]:.4f}")
    print(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
