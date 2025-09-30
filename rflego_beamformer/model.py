import torch
import torch.nn as nn
import torch.nn.functional as F


class RfLegoBeamformerLayer(nn.Module):
    """Single layer of the RF-LEGO Beamformer implementing unfolded ADMM with GRU-style gates.
    
    This layer performs one iteration of the Alternating Direction Method of Multipliers (ADMM)
    for sparse recovery, with learnable parameters and GRU-inspired gating mechanisms.
    
    Args:
        dict_length (int): Dictionary length (number of DoA angles), e.g., 121
    """
    
    def __init__(self, dict_length):
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


    def forward(self, A_H_y, z_prev, v_prev):
        """Forward pass implementing one ADMM iteration with learnable parameters.
        
        Args:
            A_H_y: Matched filter output A^H y (shape: [B, dict_length])
            z_prev: Previous sparse DoA estimate (shape: [B, dict_length])
            v_prev: Previous dual variable (shape: [B, dict_length])
            
        Returns:
            tuple: (x, z, v) - Updated primal, sparse, and dual variables
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


class RfLegoBeamformerModel(nn.Module):
    """RF-LEGO Beamformer model for Direction-of-Arrival (DoA) estimation.
    
    Implements an unfolded ADMM algorithm with GRU-style gating for sparse DoA recovery.
    The model takes antenna array measurements and a steering dictionary to estimate
    the angle-of-arrival spectrum.
    
    Args:
        dict_length (int): Dictionary length (number of DoA angles), e.g., 121
        num_layers (int): Number of unfolded ADMM iterations
    """
    
    def __init__(self, dict_length, num_layers):
        super().__init__()
        self.dict_length = dict_length
        self.num_layers = num_layers
        
        # Stack of unfolded ADMM layers
        self.layers = nn.ModuleList([
            RfLegoBeamformerLayer(dict_length) 
            for _ in range(num_layers)
        ])
        
        # Complex-valued batch normalization (without learnable affine parameters)
        self.bn_real = nn.BatchNorm1d(dict_length, affine=False)
        self.bn_imag = nn.BatchNorm1d(dict_length, affine=False)

        # Learnable amplitude thresholding parameters
        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(2.0))

    def forward(self, y, A):
        """Forward pass estimating DoA spectrum from measurements.
        
        Args:
            y: Measurement vector from antenna array (shape: [B, num_elements])
            A: Steering dictionary/matrix (shape: [B, num_elements, dict_length])
            
        Returns:
            z: Estimated sparse DoA spectrum (shape: [B, dict_length])
        """
        B, num_elems = y.shape
        _, _, N = A.shape
        assert N == self.dict_length, f"Dictionary length {N} does not match model's dict_length {self.dict_length}"

        # Compute matched filter output: A^H y
        A_H = A.conj().transpose(-2, -1)
        y_3d = y.unsqueeze(-1)
        A_H_y = torch.matmul(A_H, y_3d).squeeze(-1)

        # Apply complex batch normalization
        A_H_y_real = self.bn_real(A_H_y.real)
        A_H_y_imag = self.bn_imag(A_H_y.imag)
        A_H_y = torch.complex(A_H_y_real, A_H_y_imag)
        
        # Initialize ADMM variables
        z = torch.zeros_like(A_H_y)
        v = torch.zeros_like(A_H_y)

        # Unfolded ADMM iterations
        for layer in self.layers:
            x, z, v = layer(A_H_y, z, v)

        # Apply learnable amplitude-based thresholding
        z_real = z.real
        z_imag = z.imag
        amplitude = torch.sqrt(z_real ** 2 + z_imag ** 2)
        mask = torch.sigmoid(self.alpha * (amplitude - self.threshold))
        z_real = z_real * mask
        z_imag = z_imag * mask
        z = torch.complex(z_real, z_imag)

        return z


if __name__ == "__main__":
    from loguru import logger
    
    logger.info("RF-LEGO Beamformer model module loaded successfully!")