"""Configuration dataclasses for RF-LEGO modules and training.

This module provides structured configuration management using Python dataclasses,
enabling type-safe hyperparameter specification and easy serialization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BaseModelConfig:
    """Base configuration for all RF-LEGO modules.

    Attributes:
        dropout: Dropout probability for regularization.
        device: Target device for computation ('cpu', 'cuda', 'mps').
    """

    dropout: float = 0.0
    device: str = "cpu"


@dataclass
class DetectorConfig(BaseModelConfig):
    """Configuration for the RF-LEGO Detector model.

    The Detector uses State Space Layers (SSL) with HiPPO-LegT transitions
    for signal peak detection.

    Attributes:
        num_layers: Number of stacked state space layers.
        hidden_dim: Hidden dimension (H) of the model.
        order: State space order (N), dimension of the state vector.
        dt_min: Minimum discretization step size.
        dt_max: Maximum discretization step size.
        channels: Number of output channels per state space layer.
    """

    num_layers: int = 1
    hidden_dim: int = 256
    order: int = 256
    dt_min: float = 8e-5
    dt_max: float = 1e-1
    channels: int = 1
    dropout: float = 0.2


@dataclass
class BeamformerConfig(BaseModelConfig):
    """Configuration for the RF-LEGO Beamformer model.

    The Beamformer implements unfolded ADMM with GRU-style gating
    for sparse Direction-of-Arrival (DoA) estimation.

    Attributes:
        dict_length: Dictionary length (number of DoA angles).
        num_layers: Number of unfolded ADMM iterations.
    """

    dict_length: int = 121
    num_layers: int = 10


@dataclass
class FrequencyTransformConfig(BaseModelConfig):
    """Configuration for the RF-LEGO Frequency Transform model.

    The FT model implements learnable Bluestein's FFT algorithm
    with complex convolutions for frequency-domain signal processing.

    Attributes:
        sequence_length: Input sequence length (N).
        num_conv_layers: Number of complex convolution layers.
    """

    sequence_length: int = 256
    num_conv_layers: int = 6


@dataclass
class TrainerConfig:
    """Configuration for model training.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Initial learning rate for optimizer.
        weight_decay: Weight decay (L2 regularization) coefficient.
        total_steps: Total number of training steps (for step-based training).
        epochs: Total number of epochs (for epoch-based training).
        num_workers: Number of data loading workers.
        device: Target device for training.
        save_dir: Directory for saving model checkpoints.
        log_dir: Directory for TensorBoard logs.
        seed: Random seed for reproducibility.
    """

    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    total_steps: int = 10000
    epochs: int = 100
    num_workers: int = 4
    device: str = "cpu"
    save_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    seed: int = 42

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)


@dataclass
class SchedulerConfig:
    """Configuration for learning rate schedulers.

    Attributes:
        scheduler_type: Type of scheduler ('step', 'cosine', 'linear').
        step_size: Step size for StepLR scheduler.
        gamma: Multiplicative factor for StepLR.
        t_max: Maximum number of iterations for CosineAnnealingLR.
    """

    scheduler_type: Literal["step", "cosine", "linear"] = "step"
    step_size: int = 1000
    gamma: float = 0.9
    t_max: int = 100


@dataclass
class DetectorTrainerConfig(TrainerConfig):
    """Extended trainer configuration for Detector model.

    Attributes:
        scheduler: Learning rate scheduler configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class BeamformerTrainerConfig(TrainerConfig):
    """Extended trainer configuration for Beamformer model.

    Uses cosine annealing scheduler by default.
    """

    scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(scheduler_type="cosine")
    )


@dataclass
class FrequencyTransformTrainerConfig(TrainerConfig):
    """Extended trainer configuration for Frequency Transform model.

    Attributes:
        scheduler: Learning rate scheduler configuration.
    """

    total_steps: int = 100000
    scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(step_size=8000, gamma=0.85)
    )
