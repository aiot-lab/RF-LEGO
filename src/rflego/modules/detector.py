"""RF-LEGO Detector model based on State Space Layers.

This module implements the RF-LEGO Detector using continuous-time linear
dynamical systems with HiPPO-LegT state matrices for signal peak detection.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rflego.config import DetectorConfig
from rflego.modules.base import BaseModel


def triangular_toeplitz_multiply(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Efficient triangular Toeplitz matrix multiplication using FFT.

    Args:
        u: Input tensor.
        v: Input tensor.

    Returns:
        Output tensor after Toeplitz multiplication.
    """
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2 * n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2 * n, dim=-1)
    uv_f = u_f * v_f
    output = torch.fft.irfft(uv_f, n=2 * n, dim=-1)[..., :n]
    return output


def compute_krylov(L: int, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.

    Args:
        L: Length of the Krylov matrix.
        A: State transition matrix.
        b: Input vector.

    Returns:
        Krylov matrix of shape (..., N, L).
    """
    x = b.unsqueeze(-1)
    A_ = A

    done = L == 1
    while not done:
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., : L - l]
        else:
            _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1)
        if not done:
            A_ = A_ @ A_

    assert x.shape[-1] == L
    x = x.contiguous()
    return x


def hippo_legt_matrices(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate HiPPO-LegT state matrices for state space models.

    Args:
        N: State space dimension.

    Returns:
        Tuple of (A, B) state matrices as numpy arrays.
    """
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1) ** 0.5
    j, i = np.meshgrid(Q, Q)
    A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
    B = R[:, None]
    A = -A
    return A, B


class AdaptiveTransition(nn.Module):
    """Base class for discretizing state space equations x' = Ax + Bu.

    Supports different discretization methods (forward/backward Euler, bilinear transform)
    specialized for HiPPO-LegT transitions.

    Args:
        N: State space order (dimension of HiPPO matrix).
    """

    def __init__(self, N: int) -> None:
        super().__init__()
        self.N = N

        # Initialize HiPPO-LegT state matrices
        A, B = hippo_legt_matrices(N)
        A = torch.as_tensor(A, dtype=torch.float)
        B = torch.as_tensor(B, dtype=torch.float)[:, 0]

        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("I", torch.eye(N))

    def forward_mult(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Compute (I + delta A) u.

        Args:
            u: Input tensor (..., n).
            delta: Discretization step (...) or scalar.

        Returns:
            Output tensor (..., n).
        """
        raise NotImplementedError

    def inverse_mult(self, u: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Compute (I - delta A)^-1 u.

        Args:
            u: Input tensor (..., n).
            delta: Discretization step (...) or scalar.

        Returns:
            Output tensor (..., n).
        """
        raise NotImplementedError

    def forward_diff(
        self,
        d: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward Euler update: (I + d A) u + d B v."""
        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = self.forward_mult(u, d)
        x = x + v
        return x

    def backward_diff(
        self,
        d: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute backward Euler update: (I - d A)^-1 (u + d B v)."""
        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = u + v
        x = self.inverse_mult(x, d)
        return x

    def bilinear(
        self,
        dt: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Compute bilinear (Tustin's) update rule.

        (I - alpha*dt A)^-1 [(I + (1-alpha)*dt A) u + dt B v]
        """
        x = self.forward_mult(u, (1 - alpha) * dt)
        v = dt * v
        v = v.unsqueeze(-1) * self.B
        x = x + v
        x = self.inverse_mult(x, alpha * dt)
        return x

    def gbt_A(self, dt: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Compute transition matrix A using generalized bilinear transform.

        Args:
            dt: Discretization step.
            alpha: Bilinear transform parameter (0.5 for Tustin's method).

        Returns:
            Transition matrix (..., N, N).
        """
        dims = len(dt.shape)
        I = self.I.view([self.N] + [1] * dims + [self.N])
        A = self.bilinear(dt, I, dt.new_zeros(*dt.shape), alpha=alpha)
        A = rearrange(A, "n ... m -> ... m n", n=self.N, m=self.N)
        return A

    def gbt_B(self, dt: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
        """Compute input matrix B using generalized bilinear transform."""
        B = self.bilinear(dt, dt.new_zeros(*dt.shape, self.N), dt.new_ones(1), alpha=alpha)
        return B


class LegTTransition(AdaptiveTransition):
    """Dense matrix implementation of HiPPO-LegT transition.

    Uses explicit matrix multiplication and inversion. More memory-intensive
    but numerically stable implementation for smaller state spaces.
    """

    def forward_mult(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        transpose: bool = False,
    ) -> torch.Tensor:
        """Compute (I + delta A) u using dense matrix multiplication."""
        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)
        A_ = self.A.transpose(-1, -2) if transpose else self.A
        x = (A_ @ u.unsqueeze(-1)).squeeze(-1)
        x = u + delta * x
        return x

    def inverse_mult(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        transpose: bool = False,
    ) -> torch.Tensor:
        """Compute (I - delta A)^-1 u using matrix inversion.

        Uses batched linear solve to avoid memory issues.
        """
        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1).unsqueeze(-1)
        _A = self.I - delta * self.A
        if transpose:
            _A = _A.transpose(-1, -2)

        # Batched solve to avoid memory issues
        xs = []
        for _A_, u_ in zip(*torch.broadcast_tensors(_A, u.unsqueeze(-1))):
            x_ = torch.linalg.solve(_A_, u_[..., :1]).squeeze(-1)
            xs.append(x_)
        x = torch.stack(xs, dim=0)
        return x


class StateSpaceLayer(nn.Module):
    """State space layer implementing continuous-time linear dynamical system.

    Simulates the state space ODE:
        x'(t) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)

    Each feature in the input is processed through an independent state space
    with learnable timescales and discretization parameters.

    Args:
        d: Hidden dimension (H).
        order: State space order (N), defaults to d if <= 0.
        dt_min: Minimum discretization step size.
        dt_max: Maximum discretization step size.
        channels: Number of output channels per feature (M).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d: int,
        order: int = -1,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        channels: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.H = d
        self.N = order if order > 0 else d
        self.M = channels

        # Initialize HiPPO-LegT transition
        self.transition = LegTTransition(self.N)

        # Learnable output and feedthrough matrices
        self.C = nn.Parameter(torch.randn(self.H, self.M, self.N))
        self.D = nn.Parameter(torch.randn(self.H, self.M))

        # Initialize learnable timescales (log-uniformly distributed)
        log_dt = torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register_buffer("dt", torch.exp(log_dt))

        # Cache for Krylov matrix (convolution filter)
        self.k: torch.Tensor | None = None

        # Activation and regularization
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(self.M * self.H, self.H)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass through state space layer.

        Args:
            u: Input tensor of shape (L, B, H) where
               L = sequence length, B = batch size, H = hidden dimension.

        Returns:
            Output tensor of shape (L, B, H).
        """
        # Compute or update Krylov matrix if needed
        if self.k is None or u.shape[0] > self.k.shape[-1]:
            A = self.transition.gbt_A(self.dt)
            B = self.transition.gbt_B(self.dt)
            self.k = compute_krylov(u.shape[0], A, B)

        # Apply state space convolution
        y = self._linear_system_from_krylov(u, self.k[..., : u.shape[0]])

        # Apply activation and dropout
        y = self.dropout(self.activation_fn(y))

        # Project back to hidden dimension
        y = rearrange(y, "l b h m -> l b (h m)")
        y = self.output_linear(y)
        return y

    def _linear_system_from_krylov(
        self,
        u: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state space output y = Cx + Du using Krylov representation.

        Args:
            u: Input tensor (L, B, H).
            k: Krylov matrix (H, N, L) representing [b, Ab, A^2b, ...].

        Returns:
            Output tensor (L, B, H, M).
        """
        # Apply output matrix C
        k = self.C @ k

        # Rearrange for efficient convolution
        k = rearrange(k, "... m l -> m ... l")
        k = k.to(u)
        k = k.unsqueeze(1)

        # Compute convolution using Toeplitz multiplication
        v = u.unsqueeze(-1).transpose(0, -1)
        y = triangular_toeplitz_multiply(k, v)
        y = y.transpose(0, -1)

        # Add feedthrough term Du
        y = y + u.unsqueeze(-1) * self.D.to(device=y.device)
        return y


class DetectorModel(BaseModel):
    """RF-LEGO Detector model based on State Space Layers (SSL).

    Implements a deep state space model for signal peak detection with
    learnable thresholding and residual connections.

    Args:
        config: DetectorConfig with model hyperparameters.

    Example:
        >>> from rflego.config import DetectorConfig
        >>> config = DetectorConfig(num_layers=2, hidden_dim=256, order=256)
        >>> model = DetectorModel(config)
        >>> x = torch.randn(1024, 32)  # (sequence_length, batch_size)
        >>> logits = model(x)  # Peak detection logits
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self.d = config.hidden_dim

        # Stack of state space layers with layer normalization
        self.layers = nn.ModuleList(
            [
                StateSpaceLayer(
                    config.hidden_dim,
                    config.order,
                    config.dt_min,
                    config.dt_max,
                    config.channels,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]
        )

        # Output projection
        self.fc = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the detector model.

        Args:
            x: Input tensor of shape (L, B) where L = sequence length, B = batch size.

        Returns:
            logits: Raw logits before sigmoid (L, B).
        """
        # Expand input to hidden dimension
        x = x.unsqueeze(-1).expand(-1, -1, self.d)

        # Pass through state space layers with residual connections
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        # Project to detection logits
        logits = self.fc(x).squeeze(-1)

        return logits


# Convenience function for backward compatibility
def create_detector(
    num_layers: int = 1,
    hidden_dim: int = 256,
    order: int = 256,
    dt_min: float = 8e-5,
    dt_max: float = 1e-1,
    channels: int = 1,
    dropout: float = 0.2,
    **kwargs,
) -> DetectorModel:
    """Create a DetectorModel with the specified configuration.

    Args:
        num_layers: Number of state space layers.
        hidden_dim: Hidden dimension of the model.
        order: State space order.
        dt_min: Minimum discretization step.
        dt_max: Maximum discretization step.
        channels: Number of output channels.
        dropout: Dropout probability.
        **kwargs: Additional config parameters.

    Returns:
        Configured DetectorModel instance.
    """
    config = DetectorConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        order=order,
        dt_min=dt_min,
        dt_max=dt_max,
        channels=channels,
        dropout=dropout,
        **kwargs,
    )
    return DetectorModel(config)
