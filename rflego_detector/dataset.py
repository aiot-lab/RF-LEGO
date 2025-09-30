import os
import pickle
import torch
from torch.utils.data import Dataset


class RfLegoDetectorDataset(Dataset):
    """Dataset class for RF-LEGO Detector training data.
    
    Loads pickled data containing noisy signals and their corresponding peak detection labels
    for training the RF-LEGO Detector model.
    
    Each sample contains:
        - 'noisy_signal': 1D signal with noise (shape: [L])
        - 'labels': Binary labels indicating peak locations (shape: [L])
    
    Returns:
        dict: Dictionary containing:
            - 'input': Input signal tensor (shape: [1, L])
            - 'labels': Peak detection labels (shape: [L])
    """
    
    def __init__(self, data_dir):
        """Initialize dataset with data directory path.
        
        Args:
            data_dir (str): Path to directory containing pickle files with training data
        """
        self.data_dir = data_dir
        self.file_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith('.pkl')
        ]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load and return a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to load
            
        Returns:
            dict: Dictionary containing input signal and labels
        """
        file_path = self.file_paths[idx]
        
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)

        # Extract noisy signal and peak labels
        noisy_signal = sample['noisy_signal']
        peak_labels = sample['labels']

        # Convert to tensors with appropriate shapes
        input_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0)  # [1, L]
        label_tensor = torch.tensor(peak_labels, dtype=torch.float32)  # [L]

        return {
            'input': input_tensor,
            'labels': label_tensor
        }
