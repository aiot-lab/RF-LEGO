"""RF-LEGO Frequency Transform model based on Bluestein's FFT algorithm.

This module implements the RF-LEGO Frequency Transform model using learnable
complex convolutions for frequency-domain signal processing and denoising.
"""

import numpy as np
import torch
import torch.nn as nn

from rflego.config import FrequencyTransformConfig
from rflego.modules.base import AmplitudeThreshold, BaseModel


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution layer using real and imaginary channels.

    Implements complex multiplication via real-valued operations.

    Args:
        size_in: Number of input channels.
        size_out: Number of output channels.
    """

    def __init__(self, size_in: int, size_out: int) -> None:
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        self.conv_real = nn.Conv1d(
            in_channels=size_in,
            out_channels=size_out,
            kernel_size=1,
            bias=False,
        )
        self.conv_imag = nn.Conv1d(
            in_channels=size_in,
            out_channels=size_out,
            kernel_size=1,
            bias=False,
        )
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Xavier initialization with gain=2
        nn.init.xavier_uniform_(self.conv_real.weight, gain=2)
        nn.init.xavier_uniform_(self.conv_imag.weight, gain=2)
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def _swap_real_imag(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [real, imag] to [-imag, real] for complex multiplication."""
        return torch.stack([-x[..., 1], x[..., 0]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape [*, 2, size_in] -> [*, 2, size_out]."""
        x_perm = x.transpose(-2, -1)
        h1 = self.conv_real(x_perm)
        h2 = self.conv_imag(x_perm)
        h2 = self._swap_real_imag(h2)
        h = h1 + h2
        h = h.transpose(-2, -1) + self.bias
        return h


class ComplexLinear(nn.Module):
    """Complex-valued linear layer using real and imaginary weight matrices.

    Args:
        size_in: Number of input features.
        size_out: Number of output features.
    """

    def __init__(self, size_in: int, size_out: int) -> None:
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out

        self.weights_real = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.weights_imag = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Xavier initialization with gain=2
        nn.init.xavier_uniform_(self.weights_real, gain=2)
        nn.init.xavier_uniform_(self.weights_imag, gain=2)
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def _swap_real_imag(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [real, imag] to [-imag, real] for complex multiplication."""
        h = x.flip(dims=[-2]).transpose(-2, -1)
        device = h.device
        h = h * torch.tensor([-1, 1], device=device)
        h = h.transpose(-2, -1)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape [*, 2, size_in] -> [*, 2, size_out]."""
        h1 = torch.matmul(x, self.weights_real)
        h2 = torch.matmul(x, self.weights_imag)
        h2 = self._swap_real_imag(h2)
        h = h1 + h2
        h = torch.add(h, self.bias)
        return h


class ComplexActivation(nn.Module):
    """Complex-valued activation function applied separately to real and imaginary parts.

    Args:
        activation_type: Type of activation ('relu', 'leaky_relu', 'tanh', 'sigmoid', 'prelu').
    """

    ACTIVATION_TYPES = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.2),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "prelu": nn.PReLU,
    }

    def __init__(self, activation_type: str = "relu") -> None:
        super().__init__()
        self.activation_type = activation_type

        if activation_type not in self.ACTIVATION_TYPES:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        self.act = self.ACTIVATION_TYPES[activation_type]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: apply activation to real and imaginary parts separately."""
        self.act = self.act.to(x.device)
        real_part = x[..., 0, :]
        imag_part = x[..., 1, :]

        real_activated = self.act(real_part)
        imag_activated = self.act(imag_part)
        output = torch.stack([real_activated, imag_activated], dim=-2)

        return output


class FrequencyTransformModel(BaseModel):
    """RF-LEGO Frequency Transform model based on Bluestein's FFT algorithm.

    Implements a learnable frequency-domain transformation with complex convolutions
    and amplitude-based masking for signal processing and denoising.

    Args:
        config: FrequencyTransformConfig with model hyperparameters.

    Example:
        >>> from rflego.config import FrequencyTransformConfig
        >>> config = FrequencyTransformConfig(sequence_length=256)
        >>> model = FrequencyTransformModel(config)
        >>> x_real = torch.randn(32, 256)
        >>> x_imag = torch.randn(32, 256)
        >>> y_real, y_imag = model(x_real, x_imag)
    """

    def __init__(self, config: FrequencyTransformConfig) -> None:
        super().__init__(config)
        self.N = config.sequence_length
        self.device = config.device

        # Precompute and initialize chirp factors for Bluestein's FFT
        n = np.arange(self.N)
        chirp = np.exp(1j * np.pi * (n**2) / self.N).astype(np.complex64)
        chirp_tensor = torch.tensor(chirp, dtype=torch.complex64)
        chirp_tensor = chirp_tensor / torch.abs(chirp_tensor).mean()
        self.register_buffer("chirp_tensor", chirp_tensor)

        # Complex convolution layers
        self.conv_layers = nn.ModuleList(
            [ComplexConv1d(size_in=self.N, size_out=self.N) for _ in range(config.num_conv_layers)]
        )

        # Initialize convolution weights with chirp factors
        self._initialize_chirp_weights()

        # Activation function
        self.activation = ComplexActivation(activation_type="tanh")

        # Learnable amplitude thresholding
        self.threshold = AmplitudeThreshold(init_threshold=0.1, init_alpha=2.0)

    def _initialize_chirp_weights(self) -> None:
        """Initialize convolution weights with chirp factors."""
        for conv_layer in self.conv_layers:
            conv_layer.conv_real.weight.data *= self.chirp_tensor.real.unsqueeze(1).clone().detach()
            conv_layer.conv_imag.weight.data *= self.chirp_tensor.imag.unsqueeze(1).clone().detach()

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementing learnable Bluestein's FFT with amplitude masking.

        Args:
            x_real: Real part of input signal (shape: [B, N] or [B, 1, N]).
            x_imag: Imaginary part of input signal (shape: [B, N] or [B, 1, N]).

        Returns:
            Tuple of (y_real, y_imag) output signals with shape [B, 1, N].
        """
        # Handle input shape normalization
        if x_real.dim() == 3:
            x_real = x_real.squeeze(1)
            x_imag = x_imag.squeeze(1)

        device = x_real.device

        # Create complex tensor and apply chirp multiplication
        x = torch.complex(x_real, x_imag)
        x_chirp = x * torch.conj(self.chirp_tensor.to(device))
        
        # Zero-padding for Bluestein's algorithm
        x_padded = torch.zeros(x.shape[0], self.N, dtype=torch.complex64, device=device)
        x_padded[:, : self.N] = x_chirp

        # Convert to real-imaginary format for complex convolutions
        x_real_pad = x_padded.real.unsqueeze(-2)
        x_imag_pad = x_padded.imag.unsqueeze(-2)
        y_ri = torch.cat([x_real_pad, x_imag_pad], dim=-2)

        # Pass through complex convolution layers with activation
        for i, conv_layer in enumerate(self.conv_layers):
            y_ri = conv_layer(y_ri)
            if i < len(self.conv_layers) - 1:  # No activation after last layer
                y_ri = self.activation(y_ri)

        # Convert back to complex and apply final chirp correction
        y = torch.complex(y_ri[:, 0, :], y_ri[:, 1, :])
        y_corrected = y * torch.conj(self.chirp_tensor.to(device))
        y_corrected = torch.flip(y_corrected, [1])

        # Prepare output with same shape format as input
        y_real = y_corrected.real.unsqueeze(1)
        y_imag = y_corrected.imag.unsqueeze(1)

        # Apply learnable amplitude-based masking
        amplitude = torch.sqrt(y_real**2 + y_imag**2)
        mask = torch.sigmoid(self.threshold.alpha * (amplitude - self.threshold.threshold))
        y_real = y_real * mask
        y_imag = y_imag * mask

        return y_real, y_imag


# Convenience function for backward compatibility
def create_frequency_transform(
    sequence_length: int = 256,
    num_conv_layers: int = 6,
    device: str = "cpu",
    **kwargs,
) -> FrequencyTransformModel:
    """Create a FrequencyTransformModel with the specified configuration.

    Args:
        sequence_length: Input sequence length.
        num_conv_layers: Number of complex convolution layers.
        device: Target device.
        **kwargs: Additional config parameters.

    Returns:
        Configured FrequencyTransformModel instance.
    """
    config = FrequencyTransformConfig(
        sequence_length=sequence_length,
        num_conv_layers=num_conv_layers,
        device=device,
        **kwargs,
    )
    return FrequencyTransformModel(config)
