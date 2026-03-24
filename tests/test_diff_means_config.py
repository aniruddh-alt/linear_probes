from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

from core.configs.diff_means_config import DiffMeansConfig
from runners.experiment_runner import load_run_config, STAGE_CONFIG_MAP


class DiffMeansConfigTests(unittest.TestCase):
    def test_default_construction(self) -> None:
        cfg = DiffMeansConfig()
        self.assertEqual(cfg.action, "diff_means")
        self.assertEqual(cfg.seed, 0)

    def test_from_yaml(self) -> None:
        yaml_content = """\
action: diff_means
run_name: test-dm
seed: 42
split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
sweep:
  selection_metric: auroc
  activation_targets: [0, 1, 2]
io:
  input_path: data/activations_manifest.pt
  output_dir: artifacts
output:
  save_plots: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            cfg = load_run_config(f.name)
        self.assertIsInstance(cfg, DiffMeansConfig)
        assert isinstance(cfg, DiffMeansConfig)
        self.assertEqual(cfg.run_name, "test-dm")
        self.assertEqual(cfg.seed, 42)

    def test_registered_in_stage_config_map(self) -> None:
        self.assertIn("diff_means", STAGE_CONFIG_MAP)


if __name__ == "__main__":
    unittest.main()
