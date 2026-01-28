"""RF-LEGO Beamformer model for Direction-of-Arrival estimation.

This module implements the RF-LEGO Beamformer based on unfolded ADMM
with GRU-style gating for sparse DoA recovery from antenna array measurements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rflego.config import BeamformerConfig
from rflego.modules.base import AmplitudeThreshold, BaseModel, ComplexBatchNorm


class BeamformerLayer(nn.Module):
    """Single layer of the RF-LEGO Beamformer implementing unfolded ADMM with GRU-style gates.

    This layer performs one iteration of the Alternating Direction Method of Multipliers (ADMM)
    for sparse recovery, with learnable parameters and GRU-inspired gating mechanisms.

    Args:
        dict_length: Dictionary length (number of DoA angles), e.g., 121.
    """

    def __init__(self, dict_length: int) -> None:
        super().__init__()
        self.dict_length = dict_length

        # Learnable diagonal approximation of ADMM weight matrix
        self.w_diag = nn.Parameter(0.01 * torch.randn(dict_length, dtype=torch.cfloat))

        # Learnable ADMM parameters
        self.beta = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.raw_eta = nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32)))

        # GRU-style update gate parameters for complex-valued gating
        self.Wg_real = nn.Linear(dict_length, dict_length)
        self.Ug_real = nn.Linear(dict_length, dict_length)
        self.Wg_imag = nn.Linear(dict_length, dict_length)
        self.Ug_imag = nn.Linear(dict_length, dict_length)

    def forward(
        self,
        A_H_y: torch.Tensor,
        z_prev: torch.Tensor,
        v_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass implementing one ADMM iteration with learnable parameters.

        Args:
            A_H_y: Matched filter output A^H y (shape: [B, dict_length]).
            z_prev: Previous sparse DoA estimate (shape: [B, dict_length]).
            v_prev: Previous dual variable (shape: [B, dict_length]).

        Returns:
            Tuple of (x, z, v) - Updated primal, sparse, and dual variables.
        """
        B, N = A_H_y.shape

        # Compute learnable penalty parameter with stability
        eta = F.softplus(self.raw_eta)

        # Construct batched diagonal weight matrix
        w_diag_batch = self.w_diag.unsqueeze(0).expand(B, N)
        W = torch.diag_embed(w_diag_batch)
        I = torch.eye(N, device=W.device, dtype=W.dtype).unsqueeze(0).expand(B, N, N)

        # ADMM x-update: solve (W + eta*I)x = A^H y + eta*(z - v)
        W_eta_inv = torch.inverse(W + eta * I)
        B_term = A_H_y + eta * (z_prev - v_prev)
        x = torch.matmul(W_eta_inv, B_term.unsqueeze(-1)).squeeze(-1)

        # Compute gating input
        u = x + v_prev

        # GRU-style update gate for complex values
        g_real = torch.sigmoid(self.Wg_real(z_prev.real) + self.Ug_real(v_prev.real))
        g_imag = torch.sigmoid(self.Wg_imag(z_prev.imag) + self.Ug_imag(v_prev.imag))
        g = torch.complex(g_real, g_imag)

        # Gated sparse update
        z = g * u + (1 - g) * z_prev

        # Dual variable update
        v = v_prev + x - z

        return x, z, v


class BeamformerModel(BaseModel):
    """RF-LEGO Beamformer model for Direction-of-Arrival (DoA) estimation.

    Implements an unfolded ADMM algorithm with GRU-style gating for sparse DoA recovery.
    The model takes antenna array measurements and a steering dictionary to estimate
    the angle-of-arrival spectrum.

    Args:
        config: BeamformerConfig with model hyperparameters.

    Example:
        >>> from rflego.config import BeamformerConfig
        >>> config = BeamformerConfig(dict_length=121, num_layers=10)
        >>> model = BeamformerModel(config)
        >>> y = torch.randn(32, 8, dtype=torch.complex64)  # Measurements
        >>> A = torch.randn(32, 8, 121, dtype=torch.complex64)  # Steering matrix
        >>> spectrum = model(y, A)  # DoA spectrum
    """

    def __init__(self, config: BeamformerConfig) -> None:
        super().__init__(config)
        self.dict_length = config.dict_length
        self.num_layers = config.num_layers

        # Stack of unfolded ADMM layers
        self.layers = nn.ModuleList(
            [BeamformerLayer(config.dict_length) for _ in range(config.num_layers)]
        )

        # Complex-valued batch normalization
        self.bn = ComplexBatchNorm(config.dict_length, affine=False)

        # Learnable amplitude thresholding
        self.threshold = AmplitudeThreshold(init_threshold=0.1, init_alpha=2.0)

    def forward(self, y: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Forward pass estimating DoA spectrum from measurements.

        Args:
            y: Measurement vector from antenna array (shape: [B, num_elements]).
            A: Steering dictionary/matrix (shape: [B, num_elements, dict_length]).

        Returns:
            z: Estimated sparse DoA spectrum (shape: [B, dict_length]).
        """
        B, num_elems = y.shape
        _, _, N = A.shape

        if N != self.dict_length:
            raise ValueError(
                f"Dictionary length {N} does not match model's dict_length {self.dict_length}"
            )

        # Compute matched filter output: A^H y
        A_H = A.conj().transpose(-2, -1)  # [B, dict_length, num_elements]
        y_3d = y.unsqueeze(-1)  # [B, num_elements, 1]
        A_H_y = torch.matmul(A_H, y_3d).squeeze(-1)  # [B, dict_length]

        # Apply complex batch normalization
        A_H_y = self.bn(A_H_y)

        # Initialize ADMM variables
        z = torch.zeros_like(A_H_y)
        v = torch.zeros_like(A_H_y)

        # Unfolded ADMM iterations
        for layer in self.layers:
            x, z, v = layer(A_H_y, z, v)

        # Apply learnable amplitude-based thresholding
        z = self.threshold(z)

        return z


# Convenience function for backward compatibility
def create_beamformer(
    dict_length: int = 121,
    num_layers: int = 10,
    **kwargs,
) -> BeamformerModel:
    """Create a BeamformerModel with the specified configuration.

    Args:
        dict_length: Dictionary length (number of DoA angles).
        num_layers: Number of unfolded ADMM iterations.
        **kwargs: Additional config parameters.

    Returns:
        Configured BeamformerModel instance.
    """
    config = BeamformerConfig(
        dict_length=dict_length,
        num_layers=num_layers,
        **kwargs,
    )
    return BeamformerModel(config)
