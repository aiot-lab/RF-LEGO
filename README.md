<div align="center">

# RF-LEGO: Modularized Signal Processing-Deep Learning Co-Design for RF Sensing via Deep Unrolling


[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](https://www.gnu.org/licenses/lgpl-2.1) [![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
</div>

<div align="center">
    <a href=https://1ucayu.github.io>
        Luca Jiang-Tao Yu
    </a>
    ,
    <a href=https://cswu.me>
        Chenshu Wu
    </a>
</div>

<div align="center">

The University of Hong Kong
</div>

The official implementation of RF-LEGO: Modularized Signal Processing-Deep Learning Co-Design for RF Sensing via Deep Unrolling, accepted by ACM Mobicom 2026, Austin, TX, USA.

---

## Abstract
Wireless sensing, traditionally relying on signal processing (SP) techniques, has recently shifted toward data-driven deep learning (DL) to achieve performance breakthroughs. However, existing deep wireless sensing models are typically end-to-end and task-specific, lacking reusability and interpretability. We propose RF-LEGO, a modular co-design framework that transforms interpretable SP algorithms into trainable, physics-grounded DL modules through deep unrolling. By replacing hand-tuned parameters with learnable ones while preserving core processing structures and mathematical operators, RF-LEGO ensures modularity, cascadability, and structure-aligned interpretability. Specifically, we introduce three deep-unrolled modules for critical RF sensing tasks: frequency transform, spatial angle estimation, and signal detection. Extensive experiments using real-world data for Wi-Fi, millimeter-wave, UWB, and 6G sensing demonstrate that RF-LEGO significantly outperforms existing SP and DL baselines, both standalone and when integrated into downstream tasks such as tracking and vital sign monitoring. RF-LEGO pioneers a novel SP-DL co-design paradigm for wireless sensing via deep unrolling, shedding light on efficient and interpretable deep wireless sensing solutions.

<p align="center"> 
  <img src='src/figures/thumbnail.png' align="center">
</p>

---

## Installation

```bash
$ git clone https://github.com/1ucayu/RF-LEGO.git
$ cd RF-LEGO
$ uv sync
```

---

## Project Structure

```
RF-LEGO/
├── src/rflego/
│   ├── __init__.py          # Main exports
│   ├── config.py            # Dataclass configurations
│   ├── utils.py             # Logging & utilities
│   ├── data/                # Dataset modules
│   │   ├── base.py          # Base dataset class
│   │   └── datasets.py      # Dataset implementations
│   ├── trainer/             # Trainer
│   │   └── trainer.py       # BaseTrainer class
│   └── modules/             # Deep unrolling modules
│       ├── base.py          # Base model and blocks
│       ├── ft.py            # RF-LEGO FT
│       ├── beamformer.py    # RF-LEGO Beamformer
│       └── detector.py      # RF-LEGO Detector
├── pyproject.toml           # Project configuration
└── README.md
```

---

## Quick Start

### RF-LEGO FT

```python
import torch
from rflego import FrequencyTransformConfig, FrequencyTransformModel

# Configure and create model
config = FrequencyTransformConfig(
    sequence_length=256,
    num_conv_layers=6
)
model = FrequencyTransformModel(config)

# Forward pass (separate real/imag inputs)
x_real = torch.randn(32, 256)
x_imag = torch.randn(32, 256)
y_real, y_imag = model(x_real, x_imag)
```

### RF-LEGO Beamformer

```python
import torch
from rflego import BeamformerConfig, BeamformerModel

# Configure and create model
config = BeamformerConfig(
    dict_length=121,  # 1-degree resolution over 120 degrees
    num_layers=10     # ADMM iterations
)
model = BeamformerModel(config)

# Forward pass
y = torch.randn(32, 8, dtype=torch.complex64)      # Measurements: (batch, num_antennas)
A = torch.randn(32, 8, 121, dtype=torch.complex64) # Steering matrix
spectrum = model(y, A)  # DoA spectrum: (batch, dict_length)
```

### RF-LEGO Detector

```python
import torch
from rflego import DetectorConfig, DetectorModel, setup_logger

# Configure and create model
config = DetectorConfig(
    num_layers=2,
    hidden_dim=256,
    order=256,
    dropout=0.2
)
model = DetectorModel(config)

# Forward pass: (sequence_length, batch_size)
x = torch.randn(1024, 32)
logits = model(x)  # Output: (1024, 32)
```

---

## Training

RF-LEGO provides a flexible `BaseTrainer` class for training:

```python
from rflego import BaseTrainer, TrainerConfig
from torch.utils.data import DataLoader

class DetectorTrainer(BaseTrainer):
    def compute_loss(self, batch):
        inputs = batch['input'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Reshape and forward
        x = inputs.squeeze(1).permute(1, 0)
        logits = self.model(x)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.permute(1, 0), labels
        )
        return loss, {"loss": loss.item()}

# Setup and train
config = TrainerConfig(
    batch_size=512,
    learning_rate=1e-3,
    epochs=100
)
trainer = DetectorTrainer(model, config, train_loader, val_loader)
trainer.fit()
```

---

## Citation

```bibtex
@INPROCEEDINGS{luca2026mobicom_rflego,
  author={Luca Jiang-Tao Yu and Chenshu Wu},
  booktitle={ACM International Conference on Mobile Computing and Networking},
  title={RF-LEGO: Modularized Signal Processing-Deep Learning Co-Design for RF Sensing via Deep Unrolling},
  pages={},
  month={Oct},
  year={2026},
}
```
