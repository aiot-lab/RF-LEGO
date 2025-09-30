import torch
import torch.nn as nn
import numpy as np
from loguru import logger


class RfLegoFtSoftMaskLayer(nn.Module):
    """Learnable soft mask layer for amplitude-based signal filtering."""
    
    def __init__(self, size):
        super(RfLegoFtSoftMaskLayer, self).__init__()
        self.size = size
        self.mask_params = nn.Parameter(torch.randn(1, 1, size) * 0.1)

    def forward(self, y_real, y_imag):
        device = y_real.device
        mask = torch.sigmoid(self.mask_params.to(device))
        
        amplitude = torch.sqrt(y_real ** 2 + y_imag ** 2)
        masked_amplitude = amplitude * mask
        
        # Avoid division by zero
        y_real_new = masked_amplitude * (y_real / (amplitude + 1e-8))
        y_imag_new = masked_amplitude * (y_imag / (amplitude + 1e-8))

        return y_real_new, y_imag_new

class RfLegoFtComplexConv(nn.Module):
    """Complex-valued 1D convolution layer using real and imaginary channels."""
    
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        
        self.conv_real = nn.Conv1d(in_channels=size_in, out_channels=size_out, kernel_size=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=size_in, out_channels=size_out, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Xavier initialization with gain=2
        nn.init.xavier_uniform_(self.conv_real.weight, gain=2)
        nn.init.xavier_uniform_(self.conv_imag.weight, gain=2)
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def swap_real_imag(self, x):
        """Convert [real, imag] to [-imag, real] for complex multiplication."""
        return torch.stack([-x[..., 1], x[..., 0]], dim=-1)

    def forward(self, x):
        """Forward pass: x shape [*, 2, size_in] -> [*, 2, size_out]"""
        x_perm = x.transpose(-2, -1)
        h1 = self.conv_real(x_perm)
        h2 = self.conv_imag(x_perm)
        h2 = self.swap_real_imag(h2)
        h = h1 + h2
        h = h.transpose(-2, -1) + self.bias
        return h


class RfLegoFtComplexLinear(nn.Module):
    """Complex-valued linear layer using real and imaginary weight matrices."""
    
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out

        self.weights_real = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.weights_imag = nn.Parameter(torch.randn(size_in, size_out, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(2, size_out, dtype=torch.float32))

        # Xavier initialization with gain=2
        nn.init.xavier_uniform_(self.weights_real, gain=2)
        nn.init.xavier_uniform_(self.weights_imag, gain=2)
        nn.init.uniform_(self.bias, -0.1, 0.1)
    
    def swap_real_imag(self, x):
        """Convert [real, imag] to [-imag, real] for complex multiplication."""
        h = x.flip(dims=[-2]).transpose(-2, -1)
        device = h.device
        h = h * torch.tensor([-1, 1], device=device)
        h = h.transpose(-2, -1)
        return h

    def forward(self, x):
        """Forward pass: x shape [*, 2, size_in] -> [*, 2, size_out]"""
        h1 = torch.matmul(x, self.weights_real)
        h2 = torch.matmul(x, self.weights_imag)
        h2 = self.swap_real_imag(h2)
        h = h1 + h2
        h = torch.add(h, self.bias)
        return h

class RfLegoFtComplexBatchNorm(nn.Module):
    """Complex-valued batch normalization applied separately to real and imaginary parts."""
    
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        self.bn_real = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)
        self.bn_imag = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)

    def forward(self, x):
        """Forward pass: x shape [batch, 2, N] -> [batch, 2, N]"""
        device = x.device
        self.bn_real = self.bn_real.to(device)
        self.bn_imag = self.bn_imag.to(device)
        
        real, imag = x[:, 0, :], x[:, 1, :]
        real_bn = self.bn_real(real)
        imag_bn = self.bn_imag(imag)

        return torch.stack([real_bn, imag_bn], dim=1)


class RfLegoFtComplexActivation(nn.Module):
    """Complex-valued activation function applied separately to real and imaginary parts."""
    
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.activation_type = activation_type
        
        if activation_type == 'relu':
            self.act = nn.ReLU()
        elif activation_type == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2)
        elif activation_type == 'tanh':
            self.act = nn.Tanh()
        elif activation_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif activation_type == 'prelu':
            self.act = nn.PReLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")
    
    def forward(self, x):
        """Forward pass: apply activation to real and imaginary parts separately."""
        self.act = self.act.to(x.device)
        real_part = x[..., 0, :]
        imag_part = x[..., 1, :]
        
        real_activated = self.act(real_part)
        imag_activated = self.act(imag_part)
        output = torch.stack([real_activated, imag_activated], dim=-2)
        
        return output

class RfLegoFtModel(nn.Module):
    """RF-LEGO Frequency Transform model based on Bluestein's FFT algorithm with learnable components."""
    
    def __init__(self, N=256, device="cpu"):
        super(RfLegoFtModel, self).__init__()
        self.N = N
        self.device = device

        # Precompute and initialize chirp factors for Bluestein's FFT
        n = np.arange(self.N)
        self.chirp = np.exp(1j * np.pi * (n ** 2) / self.N).astype(np.complex64)
        self.chirp_tensor = nn.Parameter(torch.tensor(self.chirp, dtype=torch.complex64).to(device), requires_grad=False)
        self.chirp_tensor /= torch.abs(self.chirp_tensor).mean()

        # Complex convolution layers
        self.conv_layers = nn.ModuleList([
            RfLegoFtComplexConv(size_in=self.N, size_out=self.N).to(self.device)
            for _ in range(6)
        ])

        # Initialize convolution weights with chirp factors
        self._initialize_chirp_weights()

        # Activation function
        self.activation = RfLegoFtComplexActivation(activation_type='tanh')

        # Learnable amplitude thresholding parameters
        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(2.0))
    
    def _initialize_chirp_weights(self):
        """Initialize convolution weights with chirp factors."""
        for conv_layer in self.conv_layers:
            conv_layer.conv_real.weight.data *= self.chirp_tensor.real.unsqueeze(1).clone().detach()
            conv_layer.conv_imag.weight.data *= self.chirp_tensor.imag.unsqueeze(1).clone().detach()
    
    def forward(self, x_real, x_imag):
        """Forward pass implementing learnable Bluestein's FFT with amplitude masking."""
        # Handle input shape normalization
        if x_real.dim() == 3:
            x_real = x_real.squeeze(1)
            x_imag = x_imag.squeeze(1)
        
        # Create complex tensor and apply chirp multiplication
        x = torch.complex(x_real, x_imag)
        x_chirp = x * self.chirp_tensor

        # Zero-padding for Bluestein's algorithm
        x_padded = torch.zeros(x.shape[0], self.N, dtype=torch.complex64, device=self.device)
        x_padded[:, :self.N] = x_chirp

        # Convert to real-imaginary format for complex convolutions
        x_real = x_padded.real.unsqueeze(-2)
        x_imag = x_padded.imag.unsqueeze(-2)
        y_ri = torch.cat([x_real, x_imag], dim=-2)

        # Pass through complex convolution layers with activation
        for i, conv_layer in enumerate(self.conv_layers):
            y_ri = conv_layer(y_ri)
            if i < len(self.conv_layers) - 1:  # No activation after last layer
                y_ri = self.activation(y_ri)

        # Convert back to complex and apply final chirp correction
        y = torch.complex(y_ri[:, 0, :], y_ri[:, 1, :])
        y_corrected = y * torch.conj(self.chirp_tensor)
        y_corrected = torch.flip(y_corrected, [1])

        # Prepare output with same shape format as input
        y_real = y_corrected.real.unsqueeze(1)
        y_imag = y_corrected.imag.unsqueeze(1)

        # Apply learnable amplitude-based masking
        amplitude = torch.sqrt(y_real ** 2 + y_imag ** 2)
        mask = torch.sigmoid(self.alpha * (amplitude - self.threshold))
        y_real = y_real * mask
        y_imag = y_imag * mask

        return y_real, y_imag
    

# def test_rflego_ft_model():
#     """Test function for the RF-LEGO FT model with different input shapes."""
#     model = RfLegoFtModel(N=256, device="cpu")
    
#     # Test with 2D input [batch, N]
#     x_real_2d = torch.randn(5, 256)
#     x_imag_2d = torch.randn(5, 256)
#     y_real_2d, y_imag_2d = model(x_real_2d, x_imag_2d)
#     logger.info(f"2D input test - Output shape: {y_real_2d.shape}, {y_imag_2d.shape}")
    
#     # Test with 3D input [batch, 1, N]
#     x_real_3d = torch.randn(5, 1, 256)
#     x_imag_3d = torch.randn(5, 1, 256)
#     y_real_3d, y_imag_3d = model(x_real_3d, x_imag_3d)
#     logger.info(f"3D input test - Output shape: {y_real_3d.shape}, {y_imag_3d.shape}")
    
#     logger.info("RF-LEGO FT model test completed successfully!")


# if __name__ == "__main__":
#     test_rflego_ft_model()

