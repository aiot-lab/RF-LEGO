import os
import pickle
import torch
from torch.utils.data import Dataset


class RfLegoFtDataset(Dataset):
    """Dataset class for RF-LEGO Frequency Transform training data.
    
    Loads pickled data containing noisy signals and their clean frequency domain representations
    for training the RF-LEGO FT denoising model.
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
            dict: Dictionary containing:
                - 'input': Noisy signal (complex tensor)
                - 'label': Clean frequency spectrum (complex tensor)
                - 'label_signal': Clean time-domain signal
                - 'input_fft': Noisy frequency spectrum
        """
        file_path = self.file_paths[idx]
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract signals and spectra
        noise_signal = data['noisy_signal']
        clean_spectrum = data['clean_spectrum']
        clean_signal = data['clean_signal']
        noisy_spectrum = data['noisy_spectrum']
        
        # Convert to tensors with proper shape [1, N]
        noise_signal = torch.tensor(noise_signal, dtype=torch.cfloat).unsqueeze(0)
        clean_spectrum = torch.tensor(clean_spectrum, dtype=torch.cfloat).unsqueeze(0)
        
        return {
            'input': noise_signal,
            'label': clean_spectrum,
            'label_signal': clean_signal,
            'input_fft': noisy_spectrum
        }