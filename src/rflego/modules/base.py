"""Base classes for RF-LEGO neural network modules.

This module provides abstract base classes that define the common interface
and shared functionality for all RF-LEGO modules (Detector, Beamformer, FT).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rflego.config import BaseModelConfig
from rflego.utils import count_parameters, format_params, logger


class BaseModel(nn.Module, ABC):
    """Abstract base class for all RF-LEGO modules.

    Provides common functionality including:
    - Parameter counting and logging
    - Model saving and loading
    - Device management

    All RF-LEGO modules should inherit from this class and implement
    the abstract `forward` method.

    Attributes:
        config: Model configuration dataclass.
    """

    def __init__(self, config: BaseModelConfig) -> None:
        """Initialize the base model.

        Args:
            config: Model configuration dataclass.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the model.

        Must be implemented by all subclasses.

        Args:
            *args: Positional arguments specific to the model.
            **kwargs: Keyword arguments specific to the model.

        Returns:
            Model output, format depends on the specific model.
        """
        raise NotImplementedError

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        return count_parameters(self, trainable_only=trainable_only)

    def log_model_info(self) -> None:
        """Log model architecture and parameter information."""
        total_params = self.num_parameters(trainable_only=False)
        trainable_params = self.num_parameters(trainable_only=True)

        logger.info(f"Model: {self.__class__.__name__}")
        logger.info(f"  Total parameters: {format_params(total_params)} ({total_params:,})")
        logger.info(
            f"  Trainable parameters: {format_params(trainable_params)} ({trainable_params:,})"
        )

    def save(self, path: Path | str) -> None:
        """Save model state dict to file.

        Args:
            path: File path for saving the model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path | str, strict: bool = True) -> None:
        """Load model state dict from file.

        Args:
            path: File path to load the model from.
            strict: If True, strictly enforce that the keys in state_dict match.
        """
        path = Path(path)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict, strict=strict)
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(cls, path: Path | str, config: BaseModelConfig) -> "BaseModel":
        """Load a pretrained model from file.

        Args:
            path: File path to the pretrained model.
            config: Model configuration for initialization.

        Returns:
            Loaded model instance.
        """
        model = cls(config)
        model.load(path)
        return model


class ComplexBatchNorm(nn.Module):
    """Complex-valued batch normalization.

    Applies batch normalization separately to real and imaginary parts
    of complex-valued tensors.

    Args:
        num_features: Number of features (channels).
        momentum: Momentum for running statistics.
        eps: Small constant for numerical stability.
        affine: If True, use learnable affine parameters.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.bn_real = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps, affine=affine)
        self.bn_imag = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex batch normalization.
        Args:
            x: Complex tensor of shape (B, N) or (B, C, N).
        Returns:
            Normalized complex tensor with same shape.
        """
        if torch.is_complex(x):
            real_norm = self.bn_real(x.real)
            imag_norm = self.bn_imag(x.imag)
            return torch.complex(real_norm, imag_norm)
        else:
            # Assume input is (B, 2, N) with real/imag channels
            real_norm = self.bn_real(x[:, 0, :])
            imag_norm = self.bn_imag(x[:, 1, :])
            return torch.stack([real_norm, imag_norm], dim=1)


class AmplitudeThreshold(nn.Module):
    """Learnable amplitude-based soft thresholding layer.

    Applies a soft threshold based on signal amplitude, useful for
    sparsity-promoting operations in signal processing.

    Args:
        init_threshold: Initial threshold value.
        init_alpha: Initial sharpness parameter for sigmoid.
    """

    def __init__(self, init_threshold: float = 0.1, init_alpha: float = 2.0) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(init_threshold))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply amplitude-based soft thresholding.

        Args:
            x: Input tensor (real or complex).

        Returns:
            Thresholded tensor with same type and shape.
        """
        if torch.is_complex(x):
            amplitude = torch.abs(x)
            mask = torch.sigmoid(self.alpha * (amplitude - self.threshold))
            return x * mask
        else:
            # Assume separate real/imag channels
            real, imag = x[..., 0, :], x[..., 1, :]
            amplitude = torch.sqrt(real**2 + imag**2)
            mask = torch.sigmoid(self.alpha * (amplitude - self.threshold))
            return x * mask.unsqueeze(-2)
