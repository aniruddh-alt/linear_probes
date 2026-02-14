from models import ActivationExtractor


def main():
    extractor = ActivationExtractor(
        model_name="openai-community/gpt2",
        activations=["layers_output:0", "mlps_input:5", "logits"],
    )
    result = extractor.extract(
        ["The capital of France is", "The capital of Japan is"],
        token_index=-1,
    )

    print("Model info:", extractor.info)
    for name, tensors in result["activations"].items():
        print(f"{name}: samples={len(tensors)} first_shape={tuple(tensors[0].shape)}")


if __name__ == "__main__":
    main()
