from dataset import ProbingDataset, ProbingSampleBuilder
from models import ActivationExtractor


def main():
    extractor = ActivationExtractor(
        model_name="openai-community/gpt2",
        activations=[
            "layers_output:0",
            "attentions_output:3",
            "mlps_input:5",
            "logits",
        ],
    )
    result = extractor.extract(
        ProbingSampleBuilder(
            [
                {"text": "The capital of France is", "label": "1"},
                {"text": "The capital of Japan is", "label": "0"},
            ]
        ).to_samples(),
        token_index=-1,
    )

    print("Model info:", extractor.info)
    for name, tensors in result["activations"].items():
        print(f"{name}: samples={len(tensors)} first_shape={tuple(tensors[0].shape)}")

    print(result["labels"])
    probing_dataset = ProbingDataset.from_extraction_result(
        result, activation_key="layers_output:0"
    )

    print(probing_dataset)


if __name__ == "__main__":
    main()
