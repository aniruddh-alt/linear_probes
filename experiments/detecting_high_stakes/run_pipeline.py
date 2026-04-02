"""Run the high-stakes detection pipeline from config."""

from pathlib import Path

from runners.experiment_runner import run_experiment

if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    result = run_experiment(config_path=config_path)
    print(f"\nPipeline complete: {result.summary}")
