"""Data generator for RF-LEGO Detector training datasets.

This module synthesizes one-dimensional signals by superimposing 1-5 target peaks on
unit-variance white Gaussian noise. Peaks use Hann or Hamming shapes with varying widths
and are scaled to achieve 5-40 dB SNR. Labels are binary masks marking the peak locations.

By default, each sample has a length of 128.

Usage:
    python generator.py --num_samples 10000 --seq_len 128 --save_dir ./data
"""

import os
import pickle
import time
import numpy as np
from loguru import logger


def generate_sequence(seq_len: int = 128,
                      num_targets_range=(1, 6),
                      snr_db_range=(5, 40),
                      window_size_range=(20, 30)) -> tuple[np.ndarray, np.ndarray]:
    """Generate a noisy signal sequence with random windowed peaks and detection labels.
    
    Synthesizes one-dimensional signals by superimposing 1-5 target peaks on unit-variance
    white Gaussian noise. Peaks use Hann or Hamming shapes with varying widths and are
    scaled to achieve 5-40 dB SNR. Labels are binary masks marking the peak locations.
    
    Args:
        seq_len: Length of the generated sequence. Defaults to 128.
        num_targets_range: Range for random number of peaks as (min, max). Defaults to (1, 6) for 1-5 peaks.
        snr_db_range: SNR range in dB as (min, max). Defaults to (5, 40).
        window_size_range: Range for random window size as (min, max). Defaults to (20, 30).
        
    Returns:
        tuple: (sequence, labels) where:
            - sequence: Normalized signal with peaks [seq_len]
            - labels: Binary peak detection labels [seq_len]
    """
    # Initialize noise baseline
    noise_power = 1.0
    sequence = np.random.normal(scale=np.sqrt(noise_power / 2), size=seq_len)
    labels = np.zeros(seq_len, dtype=np.float32)

    # Randomly select number of peaks and their locations
    num_targets = np.random.randint(*num_targets_range)
    peak_indices = np.random.choice(seq_len, size=num_targets, replace=False)
    labels[peak_indices] = 1.0

    # Add windowed peaks to signal
    for idx in peak_indices:
        win_size = np.random.randint(*window_size_range)
        win_type = np.random.choice(["hann", "hamming"])
        snr_db = np.random.uniform(*snr_db_range)
        amp = np.sqrt(noise_power * 10 ** (snr_db / 10))

        # Select window type
        if win_type == "hann":
            window = np.hanning(win_size)
        elif win_type == "hamming":
            window = np.hamming(win_size)
        else:
            window = np.ones(win_size)

        # Apply windowed peak centered at peak location
        start = idx - win_size // 2
        for i, w in enumerate(window):
            j = start + i
            if 0 <= j < seq_len:
                sequence[j] += amp * w

    # Normalize to [0, 1] range
    sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
    return sequence.astype(np.float32), labels

class DetectorDataGenerator:
    """Data generator for RF-LEGO Detector training samples.
    
    Synthesizes one-dimensional signals by superimposing 1-5 target peaks on unit-variance
    white Gaussian noise. Peaks use Hann or Hamming shapes with varying widths and are
    scaled to achieve 5-40 dB SNR. Labels are binary masks marking the peak locations.
    Data is automatically split into training and test sets (80/20).
    
    Args:
        total_num (int): Total number of samples to generate. Defaults to 10000.
        seq_len (int): Length of each sequence. Defaults to 128.
        num_targets_range (tuple): Range for number of peaks as (min, max). Defaults to (1, 6) for 1-5 peaks.
        snr_db_range (tuple): SNR range in dB as (min, max). Defaults to (5, 40).
        window_size_range (tuple): Range for window size as (min, max). Defaults to (20, 30).
        save_dir (str): Root directory for saving datasets. Defaults to "./data".
        train_ratio (float): Ratio of training samples. Defaults to 0.8.
    """
    
    def __init__(self,
                 total_num: int = 10000,
                 seq_len: int = 128,
                 num_targets_range: tuple = (1, 6),
                 snr_db_range: tuple = (5, 40),
                 window_size_range: tuple = (20, 30),
                 save_dir: str = "./data",
                 train_ratio: float = 0.8):
        
        self.total_num = total_num
        self.seq_len = seq_len
        self.num_targets_range = num_targets_range
        self.snr_db_range = snr_db_range
        self.window_size_range = window_size_range
        self.save_dir = save_dir
        self.train_ratio = train_ratio

        # Create output directories
        os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)
        
        logger.info(f"DetectorDataGenerator initialized:")
        logger.info(f"  Total samples: {total_num}")
        logger.info(f"  Sequence length: {seq_len}")
        logger.info(f"  Targets range: {num_targets_range}")
        logger.info(f"  SNR range: {snr_db_range} dB")
        logger.info(f"  Window size range: {window_size_range}")
        logger.info(f"  Save directory: {save_dir}")

    def save_data(self, data_dict: dict, is_train: bool):
        """Save a single data sample to pickle file.
        
        Args:
            data_dict: Dictionary containing signal and labels
            is_train: Whether this is a training sample
        """
        folder = "train" if is_train else "test"
        fname = f"sample_{int(time.time()*1e6)}.pkl"
        path = os.path.join(self.save_dir, folder, fname)
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)

    def generate_and_save(self):
        """Generate all samples and save to train/test directories.
        
        Generates synthetic signal sequences with peaks and saves them as pickle files.
        Data is split according to train_ratio.
        """
        logger.info(f"Generating {self.total_num} samples...")
        logger.info(f"  Training samples: {int(self.total_num * self.train_ratio)}")
        logger.info(f"  Test samples: {int(self.total_num * (1 - self.train_ratio))}")
        
        n_train = int(self.total_num * self.train_ratio)
        for i in range(self.total_num):
            is_train = i < n_train
            
            # Generate single sample
            noisy_signal, labels = generate_sequence(
                seq_len=self.seq_len,
                num_targets_range=self.num_targets_range,
                snr_db_range=self.snr_db_range,
                window_size_range=self.window_size_range
            )
            
            # Package sample data
            data_dict = {
                "noisy_signal": noisy_signal,
                "labels": labels
            }
            
            self.save_data(data_dict, is_train)
            
            # Progress logging
            if (i + 1) % 100 == 0 or (i + 1) == self.total_num:
                logger.info(f"Progress: {i + 1}/{self.total_num} samples saved")
        
        logger.info(f"Dataset generation complete! Saved to {self.save_dir}")


if __name__ == "__main__":
    """Main entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RF-LEGO Detector Data Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data generation parameters
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Total number of samples to generate")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Length of each sequence")
    parser.add_argument("--num_targets_min", type=int, default=1,
                       help="Minimum number of peaks per sequence")
    parser.add_argument("--num_targets_max", type=int, default=5,
                       help="Maximum number of peaks per sequence (1-5 peaks)")
    parser.add_argument("--snr_min", type=float, default=5,
                       help="Minimum SNR in dB")
    parser.add_argument("--snr_max", type=float, default=40,
                       help="Maximum SNR in dB")
    parser.add_argument("--window_size_min", type=int, default=20,
                       help="Minimum window size for peaks")
    parser.add_argument("--window_size_max", type=int, default=30,
                       help="Maximum window size for peaks")
    parser.add_argument("--save_dir", type=str, default="./data",
                       help="Directory to save generated dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of training samples (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    logger.info("RF-LEGO Detector Data Generator")
    logger.info("=" * 50)
    
    # Initialize generator
    generator = DetectorDataGenerator(
        total_num=args.num_samples,
        seq_len=args.seq_len,
        num_targets_range=(args.num_targets_min, args.num_targets_max),
        snr_db_range=(args.snr_min, args.snr_max),
        window_size_range=(args.window_size_min, args.window_size_max),
        save_dir=args.save_dir,
        train_ratio=args.train_ratio
    )
    
    # Generate and save dataset
    generator.generate_and_save()
    
    logger.info("=" * 50)
    logger.info("Data generation completed successfully!")
