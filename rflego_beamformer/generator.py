"""Data generator for RF-LEGO Beamformer training datasets.

This module simulates a uniform linear array with a precomputed steering dictionary.
For each sample, we draw a random set of sources with random directions and complex
amplitudes to form a clean array snapshot, then add white Gaussian noise. The ground
truth is a sparse vector on a discrete angle grid whose nonzero entries correspond to
the dictionary indices nearest the true directions.

By default, the array has 8 antennas, and the steering grid spans from -60 to 60 degrees
at 1-degree resolution (121 grid points).

Usage:
    python generator.py --num_samples 10000 --snr_range 5 40 --save_dir ./data
"""

import os
import pickle
import random
import torch
from loguru import logger


class BeamformerDataGenerator:
    """Data generator for RF-LEGO Beamformer training samples.
    
    Simulates a uniform linear array with a precomputed steering dictionary. For each sample,
    draws a random set of sources with random directions and complex amplitudes to form a
    clean array snapshot, then adds white Gaussian noise. The ground truth is a sparse vector
    on a discrete angle grid whose nonzero entries correspond to the dictionary indices
    nearest the true directions.
    
    Args:
        num_elements (int): Number of antenna array elements. Defaults to 8.
        dictionary_length (int): Number of discrete angles in the dictionary (1-degree resolution). Defaults to 121.
        angle_range (tuple): Angular range in degrees as (min_angle, max_angle). Defaults to (-60, 60).
        gamma (float): Wavelength-to-spacing ratio for the antenna array. Defaults to 0.5.
        save_dir (str): Root directory for saving generated datasets. Defaults to "data".
    """
    
    def __init__(self, 
                 num_elements=8, 
                 dictionary_length=121, 
                 angle_range=(-60, 60), 
                 gamma=0.5,
                 save_dir="data"):
        
        self.save_dir = save_dir
        self.num_elements = num_elements
        self.dictionary_length = dictionary_length
        self.angle_range = angle_range
        self.gamma = gamma
        self.array = self._generate_array()
        self.dictionary = self.generate_dictionary()
        
        logger.info(f"BeamformerDataGenerator initialized:")
        logger.info(f"  Num elements: {num_elements}")
        logger.info(f"  Dictionary length: {dictionary_length}")
        logger.info(f"  Angle range: {angle_range}°")
        logger.info(f"  Save directory: {save_dir}")


    def _generate_array(self):
        """Generate uniform linear array element positions.
        
        Returns:
            torch.Tensor: Array element indices from 0 to num_elements-1
        """
        return torch.arange(self.num_elements)

    def _f2angle(self, f):
        """Convert normalized spatial frequency to angle in degrees.
        
        Args:
            f: Normalized spatial frequency in range [-0.5, 0.5]
            
        Returns:
            Angle in degrees
        """
        f = f - 1 / 2
        theta = torch.asin(2 * f) + torch.pi / 2
        return torch.rad2deg(theta)

    def _db(self, x: torch.Tensor):
        """Convert magnitude to decibels.
        
        Args:
            x: Input tensor
            
        Returns:
            Value in dB scale
        """
        return 10 * torch.log10(torch.abs(x))

    def _randu(self, shape, a, b):
        """Generate uniform random values in range [a, b].
        
        Args:
            shape: Output tensor shape
            a: Lower bound
            b: Upper bound
            
        Returns:
            Uniformly distributed random tensor
        """
        return a + (b - a) * torch.rand(shape)

    def _randn(self, shape, mean, std_dev):
        """Generate Gaussian random values.
        
        Args:
            shape: Output tensor shape
            mean: Mean of the distribution
            std_dev: Standard deviation of the distribution
            
        Returns:
            Normally distributed random tensor
        """
        return mean + std_dev * torch.randn(shape)

    def _randi(self, low, high, k):
        """Generate k unique random integers from [low, high].
        
        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
            k: Number of integers to sample
            
        Returns:
            Tensor of k unique random integers
        """
        assert k <= (high - low + 1), "k cannot be larger than the number of available integers."
        all_integers = torch.arange(low, high + 1)
        random_selection = all_integers[torch.randperm(all_integers.size(0))][:k]
        return random_selection

    def _idxf(self, freqs, frequency_grid):
        """Find closest indices in frequency grid for given frequencies.
        
        Args:
            freqs: Target frequencies to map
            frequency_grid: Discrete frequency grid
            
        Returns:
            Sorted indices of closest grid points
        """
        K = freqs.size(0)
        N = frequency_grid.size(0)
        closest_indices = torch.zeros(K, dtype=torch.long)
        
        for i in range(K):
            pos = torch.nonzero(frequency_grid >= freqs[i], as_tuple=True)[0]
            if pos.numel() == 0:
                closest_indices[i] = N - 1
            elif pos[0] == 0:
                closest_indices[i] = 0
            else:
                pos = pos[0]
                if torch.abs(frequency_grid[pos] - freqs[i]) < torch.abs(frequency_grid[pos - 1] - freqs[i]):
                    closest_indices[i] = pos
                else:
                    closest_indices[i] = pos - 1
        return torch.sort(closest_indices)[0]

    def _generate_random_angles(self, theta_min_deg=-60, theta_max_deg=60, min_sep_deg=2, gamma=0.5, k=None, return_freqs=True):
        """Generate random angles with minimum separation constraint.
        
        Creates k random angles within specified range, ensuring each angle is separated
        by at least min_sep_deg from its neighbors. Optionally converts to spatial frequencies.
        
        Args:
            theta_min_deg: Minimum angle in degrees
            theta_max_deg: Maximum angle in degrees
            min_sep_deg: Minimum angular separation between sources in degrees
            gamma: Wavelength-to-spacing ratio for frequency conversion
            k: Number of angles to generate (random if None)
            return_freqs: If True, return spatial frequencies; otherwise return angles
            
        Returns:
            Spatial frequencies or angles (depending on return_freqs)
        """
        if k is None:
            k = random.randint(1, 5)
        if (theta_max_deg - theta_min_deg) < (k - 1) * min_sep_deg:
            raise ValueError("Angle range too small for requested separation.")
        
        angles = torch.zeros(k)
        angles[0] = theta_min_deg + random.random() * (theta_max_deg - (k - 1) * min_sep_deg)
        for i in range(1, k):
            min_angle = angles[i - 1] + min_sep_deg
            remaining_range = theta_max_deg - (k - i) * min_sep_deg
            angles[i] = min_angle + random.random() * (remaining_range - min_angle)
        
        if return_freqs:
            freqs = -gamma * torch.sin(torch.deg2rad(angles))
            return freqs
        else:
            return angles

    def generate_dictionary(self):
        """Generate steering dictionary for DoA estimation.
        
        Creates a steering matrix (dictionary) for a uniform linear array across the
        specified angle range. Each column corresponds to the array response for a
        specific angle of arrival.
        
        Returns:
            dict: Dictionary containing:
                - 'metadata': Configuration information (array type, dimensions, etc.)
                - 'array': Array element positions
                - 'angle_grid': Angular grid in degrees
                - 'frequency_grid': Spatial frequency grid
                - 'dictionary': Steering matrix [num_elements x dictionary_length]
        """
        # Generate uniform angular grid
        angles = torch.linspace(self.angle_range[0], self.angle_range[1], self.dictionary_length)
        
        # Convert angles to spatial frequencies
        freqs = -self.gamma * torch.sin(torch.deg2rad(angles))
        
        # Compute steering vectors for each angle
        A = torch.exp(
            1j * 2 * torch.pi * self.array.unsqueeze(-1) * freqs.unsqueeze(0)
        ).to(torch.complex64)
        
        metadata = {
            'array_type': 'ULA',
            'num_elements': self.num_elements,
            'dictionary_length': self.dictionary_length,
            'angle_grid': angles,
            'frequency_grid': freqs
        }
        
        dictionary = {
            'metadata': metadata,
            'array': self.array,
            'angle_grid': angles,
            'frequency_grid': freqs,
            'dictionary': A
        }
        
        return dictionary

    def single_measurement_vector(self, snr, max_number_sources, min_freq_separation):
        """Generate a single measurement vector with multiple signal sources.
        
        Simulates antenna array measurements by creating random signal sources at different
        angles, computing array responses, and adding noise based on specified SNR.
        
        Args:
            snr: Signal-to-noise ratio in dB (can be float or [min, max] range)
            max_number_sources: Maximum number of simultaneous signal sources
            min_freq_separation: Minimum angular separation between sources in degrees
            
        Returns:
            tuple: (y, x) where:
                - y: Measurement vector [1, num_elements]
                - x: Ground truth sparse angle spectrum [1, dictionary_length]
        """
        # Randomly select number of sources
        num_signals = self._randi(1, max_number_sources, 1).item()
        
        # Generate random source angles/frequencies
        freqs = self._generate_random_angles(
            theta_min_deg=self.angle_range[0],
            theta_max_deg=self.angle_range[1],
            min_sep_deg=min_freq_separation,
            gamma=self.gamma,
            k=num_signals
        )
        
        # Generate complex amplitudes with random phases
        phas = self._randu((num_signals, 1), -torch.pi, torch.pi)
        amps = torch.exp(1j * phas).to(torch.complex64)
        
        # Compute noiseless array response
        y = torch.matmul(
            torch.exp(1j * 2 * torch.pi * self.array.unsqueeze(-1) * freqs.unsqueeze(0)).to(torch.complex64),
            amps
        )
        
        # Add noise based on SNR
        sig_power = torch.mean(torch.abs(y) ** 2)
        if isinstance(snr, (list, tuple)):
            snr = self._randu((1,), snr[0], snr[1]).item()
        sigma = torch.sqrt(sig_power / 2) * 10 ** (-snr / 20)
        noise = self._randn((len(y), 1), 0, sigma) + 1j * self._randn((len(y), 1), 0, sigma)
        y = y + noise
        
        # Create sparse ground truth vector
        x = torch.zeros((self.dictionary_length, 1), dtype=torch.complex64)
        indices = self._idxf(freqs, self.dictionary['frequency_grid'])
        x[indices] = amps
        
        return y.T, x.T

    def save_data(self, num_samples, snr_range, max_number_sources, min_freq_separation):
        """Generate and save dataset samples to pickle files.
        
        Creates training and test datasets by generating synthetic measurement vectors
        and ground truth angle spectra. Data is automatically split 80/20 for train/test.
        
        Args:
            num_samples (int): Total number of samples to generate
            snr_range (list or tuple): SNR range in dB as [min_snr, max_snr]
            max_number_sources (int): Maximum number of simultaneous signal sources
            min_freq_separation (float): Minimum angular separation between sources in degrees
        """
        # Create output directories
        train_dir = os.path.join(self.save_dir, "train")
        test_dir = os.path.join(self.save_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        logger.info(f"Generating {num_samples} samples...")
        logger.info(f"  SNR range: {snr_range} dB")
        logger.info(f"  Max sources: {max_number_sources}")
        logger.info(f"  Min separation: {min_freq_separation}°")
        
        num_train = int(num_samples * 0.8)
        
        for i in range(num_samples):
            is_train = i < num_train
            
            # Generate single sample
            y, x = self.single_measurement_vector(
                snr=snr_range, 
                max_number_sources=max_number_sources, 
                min_freq_separation=min_freq_separation
            )
            
            # Package sample data
            data_dict = {
                "signal": y,
                "angle_spectrum": x,
                "dictionary": self.dictionary
            }
            
            # Save to appropriate folder
            folder = "train" if is_train else "test"
            file_path = os.path.join(self.save_dir, folder, f"sample_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(data_dict, f)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{num_samples} samples saved")
        
        logger.info(f"Dataset generation complete! Saved to {self.save_dir}")
        logger.info(f"  Train samples: {num_train}")
        logger.info(f"  Test samples: {num_samples - num_train}")


if __name__ == "__main__":
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RF-LEGO Beamformer Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Generator configuration
    parser.add_argument("--num_elements", type=int, default=8,
                       help="Number of antenna array elements")
    parser.add_argument("--dictionary_length", type=int, default=121,
                       help="Dictionary length (number of DoA angles)")
    parser.add_argument("--angle_range", type=float, nargs=2, default=[-60, 60],
                       help="Angular range in degrees [min, max]")
    parser.add_argument("--gamma", type=float, default=0.5,
                       help="Wavelength-to-spacing ratio")
    parser.add_argument("--save_dir", type=str, default="./data",
                       help="Directory to save generated dataset")
    
    # Data generation parameters
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--snr_range", type=float, nargs=2, default=[5, 40],
                       help="SNR range in dB [min, max]")
    parser.add_argument("--max_sources", type=int, default=1,
                       help="Maximum number of simultaneous signal sources")
    parser.add_argument("--min_separation", type=float, default=10,
                       help="Minimum angular separation between sources in degrees")
    
    args = parser.parse_args()
    
    logger.info("RF-LEGO Beamformer Data Generator")
    logger.info("=" * 50)
    
    # Initialize generator
    generator = BeamformerDataGenerator(
        num_elements=args.num_elements,
        dictionary_length=args.dictionary_length,
        angle_range=tuple(args.angle_range),
        gamma=args.gamma,
        save_dir=args.save_dir
    )
    
    # Generate and save dataset
    generator.save_data(
        num_samples=args.num_samples,
        snr_range=args.snr_range,
        max_number_sources=args.max_sources,
        min_freq_separation=args.min_separation
    )
    
    logger.info("=" * 50)
    logger.info("Data generation completed successfully!")