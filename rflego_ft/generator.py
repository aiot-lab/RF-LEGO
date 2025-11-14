"""Data generator for RF-LEGO Frequency Transform training datasets.

This module provides tools for generating synthetic temporal signal sequences and their
corresponding clean frequency spectra. Following the RF-LEGO methodology:
  1. Construct clean frequency-domain spectra with multiple spectral peaks
  2. Inject typical artifacts (e.g., spectral leakage)
  3. Add white Gaussian noise across 5-40 dB SNR
  4. Apply inverse Fourier transform to obtain noisy time-domain signals

Usage:
    python generator.py --num_samples 10000 --seq_len 256 --save_dir ./data
"""

import os
import pickle
import time
import numpy as np
from loguru import logger


class FtDataGenerator:
    """Data generator for RF-LEGO Frequency Transform training samples.
    
    Generates synthetic temporal signals by first constructing clean frequency-domain spectra,
    then injecting artifacts and noise before inverse transforming to time domain. This approach
    ensures realistic spectral characteristics for training frequency transform denoising models.
    
    Args:
        seq_len (int): Sequence length (number of samples). Defaults to 256.
        max_peaks (int): Maximum number of spectral peaks per sample. Defaults to 3.
        snr_range (tuple): SNR range in dB as (min, max). Defaults to (5, 40).
        leakage_probability (float): Probability of adding spectral leakage. Defaults to 0.5.
        save_dir (str): Root directory for saving datasets. Defaults to "./data".
        total_num (int): Total number of samples to generate. Defaults to 10000.
    """
    
    def __init__(self,
                 seq_len=256,
                 max_peaks=3,
                 snr_range=(5, 40),
                 leakage_probability=0.5,
                 save_dir="./data",
                 total_num=10000):

        # Signal parameters
        self.seq_len = seq_len
        self.n_fft = seq_len
        
        # Data generation parameters
        self.max_peaks = max_peaks
        self.snr_range = snr_range
        self.leakage_probability = leakage_probability
        
        # Save configuration
        self.save_dir = save_dir
        self.total_num = total_num
        
        # Create output directories
        os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)
        
        logger.info(f"FtDataGenerator initialized:")
        logger.info(f"  Sequence length: {self.seq_len}")
        logger.info(f"  Max spectral peaks: {self.max_peaks}")
        logger.info(f"  SNR range: {self.snr_range} dB")
        logger.info(f"  Spectral leakage probability: {self.leakage_probability}")
        logger.info(f"  Save directory: {save_dir}")

    def generate_one_sample(self):
        """Generate a single sample following RF-LEGO methodology.
        
        Constructs clean frequency-domain spectra, injects artifacts (spectral leakage),
        adds AWGN noise (5-40 dB SNR), then applies inverse FFT to obtain noisy
        time-domain signals for training.
        
        Returns:
            tuple: (noisy_signal, clean_spectrum) where:
                - noisy_signal: Noisy time-domain signal (model input) [seq_len]
                - clean_spectrum: Clean frequency spectrum (learning target) [seq_len]
        """
        # Step 1: Construct clean frequency-domain spectrum with multiple peaks
        num_peaks = np.random.randint(1, self.max_peaks + 1)
        clean_spectrum = np.zeros(self.seq_len, dtype=np.complex64)
        
        # Generate random peak locations in frequency domain
        peak_indices = np.random.choice(self.seq_len, size=num_peaks, replace=False)
        
        for peak_idx in peak_indices:
            # Random amplitude and phase for each peak
            amplitude = 0.5 + 0.5 * np.random.rand()
            phase = 2 * np.pi * np.random.rand()
            clean_spectrum[peak_idx] = amplitude * np.exp(1j * phase)
        
        # Step 2: Inject typical artifacts - spectral leakage
        if np.random.rand() < self.leakage_probability:
            # Add spectral leakage around peaks (energy spreading to adjacent bins)
            leaky_spectrum = clean_spectrum.copy()
            for peak_idx in peak_indices:
                leakage_strength = 0.1 + 0.2 * np.random.rand()  # 10-30% leakage
                
                # Leak to adjacent frequency bins
                if peak_idx > 0:
                    leaky_spectrum[peak_idx - 1] += leakage_strength * clean_spectrum[peak_idx]
                if peak_idx < self.seq_len - 1:
                    leaky_spectrum[peak_idx + 1] += leakage_strength * clean_spectrum[peak_idx]
            
            # Use leaky spectrum for generating noisy signal
            working_spectrum = leaky_spectrum
        else:
            working_spectrum = clean_spectrum
        
        # Step 3: Inverse FFT to get clean time-domain signal
        clean_signal = np.fft.ifft(working_spectrum, self.n_fft)
        
        # Step 4: Add white Gaussian noise with random SNR (5-40 dB)
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Calculate signal power
        signal_power = np.mean(np.abs(clean_signal) ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate complex AWGN
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.seq_len) + 1j * np.random.randn(self.seq_len)
        )
        
        # Add noise to clean signal to get noisy time-domain signal
        noisy_signal = clean_signal + noise
        
        # Return noisy time-domain signal (input) and clean frequency spectrum (target)
        return noisy_signal.astype(np.complex64), clean_spectrum

    def save_data(self, data_dict, is_train=True):
        """Save a single data sample to pickle file.
        
        Args:
            data_dict: Dictionary containing signal and spectrum data
            is_train: Whether this is a training sample
        """
        timestamp = int(time.time() * 1000000)
        filename = f"sample_{timestamp}.pkl"
        folder = "train" if is_train else "test"
        file_path = os.path.join(self.save_dir, folder, filename)
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

    def generate_and_save_data(self):
        """Generate all samples and save to train/test directories.
        
        Follows RF-LEGO methodology: constructs clean frequency spectra, injects artifacts
        (spectral leakage), adds AWGN (5-40 dB SNR), then applies inverse FFT to obtain
        noisy time-domain signals. Data is automatically split 80/20 for train/test.
        """
        num_train = int(self.total_num * 0.8)
        
        logger.info(f"Generating {self.total_num} samples...")
        logger.info(f"  Training samples: {num_train}")
        logger.info(f"  Test samples: {self.total_num - num_train}")
        
        start_time = time.time()
        
        for i in range(self.total_num):
            is_train = i < num_train
            
            # Generate single sample following RF-LEGO methodology
            noisy_signal, clean_spectrum = self.generate_one_sample()
            
            # Package sample data
            data_dict = {
                "clean_spectrum": clean_spectrum,
                "noisy_signal": noisy_signal,
                "noisy_spectrum": np.fft.fft(noisy_signal),
                "clean_signal": np.fft.ifft(clean_spectrum),
            }
            
            self.save_data(data_dict, is_train)
            
            # Progress logging
            if (i + 1) % 500 == 0 or (i + 1) == self.total_num:
                logger.info(f"Progress: {i + 1}/{self.total_num} samples saved")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Dataset generation complete! Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Saved to {self.save_dir}")



if __name__ == "__main__":
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RF-LEGO Frequency Transform Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Signal parameters
    parser.add_argument("--seq_len", type=int, default=256,
                       help="Sequence length (number of samples)")
    
    # Spectrum parameters
    parser.add_argument("--max_peaks", type=int, default=3,
                       help="Maximum number of spectral peaks per sample")
    parser.add_argument("--snr_min", type=float, default=5,
                       help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=float, default=40,
                       help="Maximum SNR in dB")
    parser.add_argument("--leakage_prob", type=float, default=0.5,
                       help="Probability of adding spectral leakage (0.0 to 1.0)")
    
    # Data generation parameters
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--save_dir", type=str, default="./data",
                       help="Directory to save generated dataset")
    
    args = parser.parse_args()
    
    logger.info("RF-LEGO Frequency Transform Data Generator")
    logger.info("=" * 50)
    
    # Initialize generator
    generator = FtDataGenerator(
        seq_len=args.seq_len,
        max_peaks=args.max_peaks,
        snr_range=(args.snr_min, args.snr_max),
        leakage_probability=args.leakage_prob,
        save_dir=args.save_dir,
        total_num=args.num_samples
    )
    
    # Generate and save dataset following RF-LEGO methodology
    generator.generate_and_save_data()
    
    logger.info("=" * 50)
    logger.info("Data generation completed successfully!")