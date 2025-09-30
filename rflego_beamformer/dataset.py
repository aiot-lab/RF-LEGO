import os
import pickle
import torch
from torch.utils.data import Dataset
from loguru import logger


class RfLegoBeamformerDataset(Dataset):
    """Dataset class for RF-LEGO Beamformer training data.
    
    Loads pickled data containing measurement signals, angle spectra, and dictionary matrices
    for training the RF-LEGO Beamformer model for Direction-of-Arrival (DoA) estimation.
    
    Each sample contains:
        - 'signal': Measurement vector from antenna array (shape: [1, num_elements])
        - 'angle_spectrum': Ground truth DoA spectrum (shape: [1, dictionary_length])
        - 'dictionary': Steering matrix/dictionary (shape: [num_elements, dictionary_length])
    
    Returns:
        dict: Dictionary containing:
            - 'input': Measurement signal (complex tensor)
            - 'label': Ground truth angle spectrum (complex tensor)
            - 'dictionary': Steering matrix for beamforming
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

        if not self.file_paths:
            logger.warning(f"No .pkl files found in {data_dir}")

        logger.info(f"RfLegoBeamformerDataset initialized with {len(self.file_paths)} samples")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load and return a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to load
            
        Returns:
            dict: Dictionary containing input signal, label spectrum, and dictionary matrix
        """
        file_path = self.file_paths[idx]
        
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Extract measurement signal and ground-truth angle spectrum
        signal = data_dict["signal"]
        angle_spectrum = data_dict["angle_spectrum"]
        dictionary_info = data_dict["dictionary"]

        # Convert to PyTorch tensors (handle both numpy arrays and existing tensors)
        if isinstance(signal, torch.Tensor):
            signal_tensor = signal.clone().detach().to(torch.complex64)
        else:
            signal_tensor = torch.tensor(signal, dtype=torch.complex64)

        if isinstance(angle_spectrum, torch.Tensor):
            angle_spectrum_tensor = angle_spectrum.clone().detach().to(torch.complex64)
        else:
            angle_spectrum_tensor = torch.tensor(angle_spectrum, dtype=torch.complex64)

        # Remove leading batch dimension if present
        signal_tensor = signal_tensor.squeeze(0)
        angle_spectrum_tensor = angle_spectrum_tensor.squeeze(0)

        return {
            "input": signal_tensor,
            "label": angle_spectrum_tensor,
            "dictionary": dictionary_info
        }
