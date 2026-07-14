"""sonde — a mech-interp toolkit for activation extraction and linear probing.

A *sonde* is a slender instrument cast into an inaccessible medium to report
back what it finds. This package drops linear probes into the hidden layers of
transformer models: extract activations, train probes, sweep across layers to
find where a concept lives, and turn the result into a concept direction usable
for causal interventions.

Public API is re-exported here so the common path is ``from sonde import X``.
"""

from __future__ import annotations

__version__ = "0.1.0"

from sonde.activation import (
    ActivationExtractor,
    AllTokens,
    ExtractionResult,
    Index,
    IndexList,
    LastNonPad,
    Range,
    StringAnchor,
    TokenIdAnchor,
    TokenSelector,
)
from sonde.core.configs import (
    BaseConfig,
    DatasetParams,
    DiffMeansConfig,
    ExtractConfig,
    ExtractionParams,
    GenerateConfig,
    GenerationParams,
    IOParams,
    ModelParams,
    OutputParams,
    PipelineConfig,
    ProbeConfig,
    ProbeParams,
    SplitParams,
    SteeringParams,
    SweepParams,
    TokenSelectorParams,
)
from sonde.dataset import (
    ProbingDataset,
    ProbingSampleBuilder,
    SampleBundle,
    StringDataset,
    stratified_train_val_test_split,
)
from sonde.directions import (
    DiffMeansEstimator,
    DiffMeansLayerResult,
    DiffMeansSweepResult,
    DiffMeansSweepRunner,
    evaluate_projection,
)
from sonde.interventions import (
    InterventionContext,
    PendingSteer,
    load_vector,
)
from sonde.probes import (
    BaseProbe,
    BinaryProbeTrainer,
    LayerProbeSweepResult,
    LayerProbeSweepRunner,
    LinearProbe,
    ProbeAnalyzer,
    ProbeArtifact,
    TrainedLayerProbe,
    build_probe,
    run_probe_with_controls,
    save_probe_artifact,
)
from sonde.runners import RunResult, load_run_config, run_experiment

__all__ = [
    "ActivationExtractor",
    "AllTokens",
    "BaseConfig",
    "BaseProbe",
    "BinaryProbeTrainer",
    "DatasetParams",
    "DiffMeansConfig",
    "DiffMeansEstimator",
    "DiffMeansLayerResult",
    "DiffMeansSweepResult",
    "DiffMeansSweepRunner",
    "ExtractConfig",
    "ExtractionParams",
    "ExtractionResult",
    "GenerateConfig",
    "GenerationParams",
    "IOParams",
    "Index",
    "IndexList",
    "InterventionContext",
    "LastNonPad",
    "LayerProbeSweepResult",
    "LayerProbeSweepRunner",
    "LinearProbe",
    "ModelParams",
    "OutputParams",
    "PendingSteer",
    "PipelineConfig",
    "ProbeAnalyzer",
    "ProbeArtifact",
    "ProbeConfig",
    "ProbeParams",
    "ProbingDataset",
    "ProbingSampleBuilder",
    "Range",
    "RunResult",
    "SampleBundle",
    "SplitParams",
    "SteeringParams",
    "StringAnchor",
    "StringDataset",
    "SweepParams",
    "TokenIdAnchor",
    "TokenSelector",
    "TokenSelectorParams",
    "TrainedLayerProbe",
    "__version__",
    "build_probe",
    "evaluate_projection",
    "load_run_config",
    "load_vector",
    "run_experiment",
    "run_probe_with_controls",
    "save_probe_artifact",
    "stratified_train_val_test_split",
]
