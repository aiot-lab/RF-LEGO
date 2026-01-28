"""Neural network modules for RF-LEGO.

This module provides the core module implementations for RF signal processing:
- DetectorModel: Peak detection using State Space Layers
- BeamformerModel: Direction-of-Arrival estimation using unfolded ADMM
- FrequencyTransformModel: Frequency-domain processing using Bluestein's FFT
"""

from rflego.modules.base import AmplitudeThreshold, BaseModel, ComplexBatchNorm
from rflego.modules.beamformer import BeamformerLayer, BeamformerModel, create_beamformer
from rflego.modules.detector import (
    AdaptiveTransition,
    DetectorModel,
    LegTTransition,
    StateSpaceLayer,
    create_detector,
)
from rflego.modules.ft import (
    ComplexActivation,
    ComplexConv1d,
    ComplexLinear,
    FrequencyTransformModel,
    create_frequency_transform,
)

__all__ = [
    # Base classes
    "BaseModel",
    "ComplexBatchNorm",
    "AmplitudeThreshold",
    # Detector
    "DetectorModel",
    "StateSpaceLayer",
    "AdaptiveTransition",
    "LegTTransition",
    "create_detector",
    # Beamformer
    "BeamformerModel",
    "BeamformerLayer",
    "create_beamformer",
    # Frequency Transform
    "FrequencyTransformModel",
    "ComplexConv1d",
    "ComplexLinear",
    "ComplexActivation",
    "create_frequency_transform",
]
