import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


def rflego_detector_triangular_toeplitz_multiply(u, v):
    """Efficient triangular Toeplitz matrix multiplication using FFT.
    
    Args:
        u: Input tensor
        v: Input tensor
        
    Returns:
        Output tensor after Toeplitz multiplication
    """
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)
    uv_f = u_f * v_f
    output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
    return output


def rflego_detector_krylov(L, A, b):
    """Compute the Krylov matrix (b, Ab, A^2b, ...) using the squaring trick.
    
    Args:
        L: Length of the Krylov matrix
        A: State transition matrix
        b: Input vector
        
    Returns:
        Krylov matrix of shape (..., N, L)
    """
    x = b.unsqueeze(-1)
    A_ = A

    done = L == 1
    while not done:
        l = x.shape[-1]
        if L - l <= l:
            done = True
            _x = x[..., :L-l]
        else:
            _x = x

        _x = A_ @ _x
        x = torch.cat([x, _x], dim=-1)
        if not done:
            A_ = A_ @ A_

    assert x.shape[-1] == L
    x = x.contiguous()
    return x


def rflego_detector_hippo(N):
    """Generate HiPPO-LegT state matrices for state space models.
    
    Args:
        N: State space dimension
        
    Returns:
        tuple: (A, B) state matrices
    """
    Q = np.arange(N, dtype=np.float64)
    R = (2*Q + 1) ** 0.5
    j, i = np.meshgrid(Q, Q)
    A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
    B = R[:, None]
    A = -A
    return A, B


class RfLegoDetectorAdaptiveTransition(nn.Module):
    """Base class for discretizing state space equations x' = Ax + Bu.
    
    Supports different discretization methods (forward/backward Euler, bilinear transform)
    specialized for HiPPO-LegT transitions.
    """

    def __init__(self, N):
        """Initialize adaptive transition with HiPPO matrices.
        
        Args:
            N: State space order (dimension of HiPPO matrix)
        """
        super().__init__()
        self.N = N

        # Initialize HiPPO-LegT state matrices
        A, B = rflego_detector_hippo(N)
        A = torch.as_tensor(A, dtype=torch.float)
        B = torch.as_tensor(B, dtype=torch.float)[:, 0]

        self.register_buffer('A', A)
        self.register_buffer('B', B)
        self.register_buffer('I', torch.eye(N))


    def forward_mult(self, u, delta):
        """Compute (I + delta A) u.
        
        Args:
            u: Input tensor (..., n)
            delta: Discretization step (...) or scalar
            
        Returns:
            Output tensor (..., n)
        """
        raise NotImplementedError

    def inverse_mult(self, u, delta):
        """Compute (I - delta A)^-1 u.
        
        Args:
            u: Input tensor (..., n)
            delta: Discretization step (...) or scalar
            
        Returns:
            Output tensor (..., n)
        """
        raise NotImplementedError

    def forward_diff(self, d, u, v):
        """Compute forward Euler update: (I + d A) u + d B v."""
        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = self.forward_mult(u, d)
        x = x + v
        return x

    def backward_diff(self, d, u, v):
        """Compute backward Euler update: (I - d A)^-1 (u + d B v)."""
        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = u + v
        x = self.inverse_mult(x, d)
        return x

    def bilinear(self, dt, u, v, alpha=0.5):
        """Compute bilinear (Tustin's) update rule.
        
        (I - alpha*dt A)^-1 [(I + (1-alpha)*dt A) u + dt B v]
        """
        x = self.forward_mult(u, (1-alpha)*dt)
        v = dt * v
        v = v.unsqueeze(-1) * self.B
        x = x + v
        x = self.inverse_mult(x, alpha*dt)
        return x

    def gbt_A(self, dt, alpha=0.5):
        """Compute transition matrix A using generalized bilinear transform.
        
        Args:
            dt: Discretization step
            alpha: Bilinear transform parameter (0.5 for Tustin's method)
            
        Returns:
            Transition matrix (..., N, N)
        """
        dims = len(dt.shape)
        I = self.I.view([self.N] + [1]*dims + [self.N])
        A = self.bilinear(dt, I, dt.new_zeros(*dt.shape), alpha=alpha)
        A = rearrange(A, 'n ... m -> ... m n', n=self.N, m=self.N)
        return A

    def gbt_B(self, dt, alpha=0.5):
        """Compute input matrix B using generalized bilinear transform."""
        B = self.bilinear(dt, dt.new_zeros(*dt.shape, self.N), dt.new_ones(1), alpha=alpha)
        return B


class RfLegoDetectorLegTTransition(RfLegoDetectorAdaptiveTransition):
    """Dense matrix implementation of HiPPO-LegT transition.
    
    Uses explicit matrix multiplication and inversion. More memory-intensive
    but numerically stable implementation for smaller state spaces.
    """

    def forward_mult(self, u, delta, transpose=False):
        """Compute (I + delta A) u using dense matrix multiplication."""
        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)
        A_ = self.A.transpose(-1, -2) if transpose else self.A
        x = (A_ @ u.unsqueeze(-1)).squeeze(-1)
        x = u + delta * x
        return x

    def inverse_mult(self, u, delta, transpose=False):
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


class RfLegoDetectorStateSpace(nn.Module):
    """State space layer implementing continuous-time linear dynamical system.
    
    Simulates the state space ODE:
        x'(t) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)
    
    Each feature in the input is processed through an independent state space
    with learnable timescales and discretization parameters.
    
    Args:
        d: Hidden dimension (H)
        order: State space order (N), defaults to d if <= 0
        dt_min: Minimum discretization step size
        dt_max: Maximum discretization step size
        channels: Number of output channels per feature (M)
        dropout: Dropout probability
    """
    
    def __init__(
            self,
            d,
            order=-1,
            dt_min=1e-3,
            dt_max=1e-1,
            channels=1,
            dropout=0.0,
        ):
        super().__init__()
        self.H = d
        self.N = order if order > 0 else d
        self.M = channels

        # Initialize HiPPO-LegT transition
        self.transition = RfLegoDetectorLegTTransition(self.N)

        # Learnable output and feedthrough matrices
        self.C = nn.Parameter(torch.randn(self.H, self.M, self.N))
        self.D = nn.Parameter(torch.randn(self.H, self.M))

        # Initialize learnable timescales (log-uniformly distributed)
        log_dt = torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.register_buffer('dt', torch.exp(log_dt))

        # Cache for Krylov matrix (convolution filter)
        self.k = None

        # Activation and regularization
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(self.M * self.H, self.H)


    def forward(self, u):
        """Forward pass through state space layer.
        
        Args:
            u: Input tensor of shape (L, B, H) where
               L = sequence length, B = batch size, H = hidden dimension
               
        Returns:
            Output tensor of shape (L, B, H)
        """
        # Compute or update Krylov matrix if needed
        if self.k is None or u.shape[0] > self.k.shape[-1]:
            A = self.transition.gbt_A(self.dt)
            B = self.transition.gbt_B(self.dt)
            self.k = rflego_detector_krylov(u.shape[0], A, B)

        # Apply state space convolution
        y = self.linear_system_from_krylov(u, self.k[..., :u.shape[0]])

        # Apply activation and dropout
        y = self.dropout(self.activation_fn(y))

        # Project back to hidden dimension
        y = rearrange(y, 'l b h m -> l b (h m)')
        y = self.output_linear(y)
        return y

    def linear_system_from_krylov(self, u, k):
        """Compute state space output y = Cx + Du using Krylov representation.
        
        Args:
            u: Input tensor (L, B, H)
            k: Krylov matrix (H, N, L) representing [b, Ab, A^2b, ...]
            
        Returns:
            Output tensor (L, B, H, M)
        """
        # Apply output matrix C
        k = self.C @ k

        # Rearrange for efficient convolution
        k = rearrange(k, '... m l -> m ... l')
        k = k.to(u)
        k = k.unsqueeze(1)

        # Compute convolution using Toeplitz multiplication
        v = u.unsqueeze(-1).transpose(0, -1)
        y = rflego_detector_triangular_toeplitz_multiply(k, v)
        y = y.transpose(0, -1)
        
        # Add feedthrough term Du
        y = y + u.unsqueeze(-1) * self.D.to(device=y.device)
        return y

class RfLegoDetectorModel(nn.Module):
    """RF-LEGO Detector model based on State Space Layers (SSL).
    
    Implements a deep state space model for signal peak detection with
    learnable thresholding and residual connections.
    
    Args:
        num_layers: Number of state space layers
        d: Hidden dimension
        order: State space order (dimension of state vector)
        dt_min: Minimum discretization step size
        dt_max: Maximum discretization step size
        channels: Number of output channels per state space layer
        dropout: Dropout probability
    """
    
    def __init__(self, num_layers, d, order, dt_min, dt_max, channels, dropout):
        super().__init__()
        self.d = d
        
        # Stack of state space layers with layer normalization
        self.layers = nn.ModuleList([
            RfLegoDetectorStateSpace(d, order, dt_min, dt_max, channels, dropout) 
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(num_layers)])
        
        # Output projection
        self.fc = nn.Linear(d, 1)
        
    def forward(self, x):
        """Forward pass through the detector model.
        
        Args:
            x: Input tensor of shape (L, B) where L = sequence length, B = batch size
            
        Returns:
            tuple: (logits)
                - logits: Raw logits before sigmoid (L, B)
        """
        # Expand input to hidden dimension
        x = x.unsqueeze(-1).expand(-1, -1, self.d)
        
        # Pass through state space layers with residual connections
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        
        # Project to detection logits
        logits = self.fc(x).squeeze(-1)
        
        return logits


def rflego_detector_count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    logger.info("RF-LEGO Detector model module loaded successfully!")