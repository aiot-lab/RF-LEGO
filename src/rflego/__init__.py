"""RF-LEGO: Lightweight, Efficient, Generalizable, and Optimized neural operators for RF signal processing.

RF-LEGO provides production-grade implementations of neural network modules
for various RF signal processing tasks:

- **Detector**: Peak detection using State Space Layers (SSL) with HiPPO-LegT transitions
- **Beamformer**: Direction-of-Arrival (DoA) estimation using unfolded ADMM with GRU-style gating
- **Frequency Transform**: Learnable frequency-domain processing based on Bluestein's FFT

Example:
    >>> from rflego.config import DetectorConfig
    >>> from rflego.modules import DetectorModel
    >>>
    >>> config = DetectorConfig(num_layers=2, hidden_dim=256)
    >>> model = DetectorModel(config)
    >>> model.log_model_info()

For more information, see the documentation and examples.
"""

__version__ = "0.1.0"
__author__ = "Kong-To"

# Re-export main components for convenient access
from rflego.config import (
    BaseModelConfig,
    BeamformerConfig,
    BeamformerTrainerConfig,
    DetectorConfig,
    DetectorTrainerConfig,
    FrequencyTransformConfig,
    FrequencyTransformTrainerConfig,
    SchedulerConfig,
    TrainerConfig,
)
from rflego.data import (
    BaseDataset,
    BeamformerDataset,
    DetectorDataset,
    FrequencyTransformDataset,
    create_dataset,
)
from rflego.trainer import BaseTrainer
from rflego.modules import (
    AmplitudeThreshold,
    BaseModel,
    BeamformerModel,
    ComplexBatchNorm,
    DetectorModel,
    FrequencyTransformModel,
    create_beamformer,
    create_detector,
    create_frequency_transform,
)
from rflego.utils import (
    count_parameters,
    ensure_dir,
    format_params,
    get_device,
    logger,
    set_seed,
    setup_logger,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Config
    "BaseModelConfig",
    "DetectorConfig",
    "BeamformerConfig",
    "FrequencyTransformConfig",
    "TrainerConfig",
    "SchedulerConfig",
    "DetectorTrainerConfig",
    "BeamformerTrainerConfig",
    "FrequencyTransformTrainerConfig",
    # Modules
    "BaseModel",
    "DetectorModel",
    "BeamformerModel",
    "FrequencyTransformModel",
    "ComplexBatchNorm",
    "AmplitudeThreshold",
    # Factory functions
    "create_detector",
    "create_beamformer",
    "create_frequency_transform",
    # Data
    "BaseDataset",
    "DetectorDataset",
    "BeamformerDataset",
    "FrequencyTransformDataset",
    "create_dataset",
    # Trainer
    "BaseTrainer",
    # Utils
    "logger",
    "setup_logger",
    "get_device",
    "count_parameters",
    "format_params",
    "set_seed",
    "ensure_dir",
]
