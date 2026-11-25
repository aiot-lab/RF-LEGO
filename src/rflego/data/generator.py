"""Synthetic training data generators for RF-LEGO modules.

Provides both:
1. Standalone dataset generator classes (FtDataGenerator, BeamformerDataGenerator,
   DetectorDataGenerator) that synthesize training/test splits to .pkl files
   following Section 4 of the paper.
2. Deterministic in-memory NumPy batch builders (generate_frequency_transform_batch,
   generate_beamformer_batch, generate_detector_batch) seeded via make_rng.
"""

from __future__ import annotations

import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from rflego.utils import logger

__all__ = [
    "MODULE_ID",
    "make_rng",
    "generate_frequency_transform_batch",
    "generate_beamformer_batch",
    "generate_detector_batch",
    "generate_sequence",
    "FtDataGenerator",
    "BeamformerDataGenerator",
    "DetectorDataGenerator",
]

# --------------------------------------------------------------------------- #
# Reproducible in-memory RNG streams & batch builders
# --------------------------------------------------------------------------- #
MODULE_ID = {"ft": 0, "beamformer": 1, "detector": 2}
TRAIN_STREAM_ID = 0


def make_rng(module: str, seed: int = 42) -> np.random.Generator:
    """Return an independent NumPy RNG for synthesized training samples."""
    if module not in MODULE_ID:
        raise ValueError(f"Unknown module {module!r}; expected one of {list(MODULE_ID)}")
    ss = np.random.SeedSequence(entropy=seed, spawn_key=(TRAIN_STREAM_ID, MODULE_ID[module]))
    return np.random.default_rng(ss)


def _case_layout(n: int, n_cases: int, rng: np.random.Generator, balanced: bool) -> np.ndarray:
    """Build the internal per-sample condition vector."""
    if balanced:
        if n % n_cases != 0:
            raise ValueError(f"n={n} must be divisible by n_cases={n_cases} for a balanced layout")
        return np.repeat(np.arange(n_cases), n // n_cases).astype(np.int64)
    return rng.integers(0, n_cases, size=n).astype(np.int64)


def _cnoise(rng: np.random.Generator, shape, sigma) -> np.ndarray:
    """Circularly-symmetric complex Gaussian noise with E[|n|^2] = sigma^2."""
    sigma = np.asarray(sigma)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (sigma / np.sqrt(2.0)) * (real + 1j * imag)


def _gaussian_bump(centers, length: int, sigma: float = 1.0) -> np.ndarray:
    """Peak-normalized Gaussian bump(s) on a circular grid."""
    idx = np.arange(length)
    out = np.zeros(length, dtype=np.float64)
    for c in np.atleast_1d(centers):
        if c < 0:
            continue
        d = np.minimum(np.abs(idx - c), length - np.abs(idx - c))
        out = np.maximum(out, np.exp(-0.5 * (d / sigma) ** 2))
    return out


def generate_frequency_transform_batch(
    n: int,
    rng: np.random.Generator,
    N: int = 256,
    balanced: bool = False,
    target_sigma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Build synthesized frequency-transform training samples."""
    case = _case_layout(n, 5, rng, balanced)
    x = np.zeros((n, N), dtype=np.complex128)
    gt_peak_bin = np.zeros(n, dtype=np.int64)
    gt_freq = np.zeros(n, dtype=np.float64)
    gt_tone_bins = np.full((n, 3), -1, dtype=np.int64)
    n_tones = np.ones(n, dtype=np.int64)
    snr_db = np.zeros(n, dtype=np.float64)
    nn = np.arange(N)

    snr_by_case = {0: 20.0, 1: 3.0, 2: 10.0, 3: 20.0, 4: 3.0}
    band_by_case = {
        0: (0.10, 0.42),
        1: (0.10, 0.42),
        2: (0.10, 0.42),
        3: (0.55, 0.88),
        4: (0.55, 0.88),
    }

    for i in range(n):
        c = int(case[i])
        snr = snr_by_case[c] + rng.uniform(-1.0, 1.0)
        snr_db[i] = snr
        lo, hi = band_by_case[c]
        f0 = rng.uniform(lo * N, hi * N) + rng.uniform(-0.5, 0.5)
        phi = rng.uniform(0.0, 2 * np.pi)
        a = 1.0
        sig = a * np.exp(1j * (2 * np.pi * f0 * nn / N + phi))
        peak_bin = int(round(f0)) % N
        gt_freq[i] = f0
        gt_peak_bin[i] = peak_bin
        gt_tone_bins[i, 0] = peak_bin

        if c == 2:
            f1 = rng.uniform(lo * N, hi * N) + rng.uniform(-0.5, 0.5)
            while abs(f1 - f0) < 6:
                f1 = rng.uniform(lo * N, hi * N) + rng.uniform(-0.5, 0.5)
            a1 = rng.uniform(0.3, 0.5)
            sig = sig + a1 * np.exp(1j * (2 * np.pi * f1 * nn / N + rng.uniform(0, 2 * np.pi)))
            gt_tone_bins[i, 1] = int(round(f1)) % N
            n_tones[i] = 2

        sigma = a / np.sqrt(10 ** (snr / 10.0))
        x[i] = sig + _cnoise(rng, N, sigma)

    gt_spectrum = np.stack(
        [_gaussian_bump(gt_peak_bin[i], N, sigma=target_sigma) for i in range(n)]
    ).astype(np.float32)

    return {
        "x_real": x.real.astype(np.float32),
        "x_imag": x.imag.astype(np.float32),
        "gt_peak_bin": gt_peak_bin,
        "gt_freq": gt_freq.astype(np.float32),
        "gt_spectrum": gt_spectrum,
        "gt_tone_bins": gt_tone_bins,
        "n_tones": n_tones,
        "snr_db": snr_db.astype(np.float32),
    }


def _steering_matrix(angles_deg: np.ndarray, M: int) -> np.ndarray:
    """Half-wavelength ULA steering matrix A[m, d] = exp(-j*pi*m*sin(theta_d))."""
    m = np.arange(M)[:, None]
    theta = np.deg2rad(angles_deg)[None, :]
    return np.exp(-1j * np.pi * m * np.sin(theta)).astype(np.complex128)


def generate_beamformer_batch(
    n: int,
    rng: np.random.Generator,
    M: int = 8,
    D: int = 121,
    angle_min: float = -60.0,
    angle_max: float = 60.0,
    balanced: bool = False,
    target_sigma: float = 1.0,
    profile: dict | None = None,
) -> dict[str, np.ndarray]:
    """Build synthesized ULA training samples for sparse-DoA recovery."""
    case = _case_layout(n, 5, rng, balanced)
    angle_grid = np.linspace(angle_min, angle_max, D).astype(np.float64)
    A_dict = _steering_matrix(angle_grid, M)

    y_meas = np.zeros((n, M), dtype=np.complex128)
    A_out = np.zeros((n, M, D), dtype=np.complex128)
    gt_angle_idx = np.full((n, 2), -1, dtype=np.int64)
    gt_angles_deg = np.full((n, 2), np.nan, dtype=np.float64)
    n_targets = np.zeros(n, dtype=np.int64)
    snr_db = np.zeros(n, dtype=np.float64)

    profile = profile or {}
    snr_by_case = profile.get("snr", {0: 4.0, 1: 3.0, 2: 3.0, 3: 2.0, 4: 2.0})
    n_clutter_by_case = profile.get("n_clutter", {0: 3, 1: 4, 2: 4, 3: 4, 4: 5})
    clutter_amp_by_case = profile.get(
        "clutter_amp",
        {0: (0.22, 0.36), 1: (0.24, 0.38), 2: (0.24, 0.38), 3: (0.26, 0.40), 4: (0.28, 0.42)},
    )

    def steer(theta_deg: float) -> np.ndarray:
        return np.exp(-1j * np.pi * np.arange(M) * np.sin(np.deg2rad(theta_deg)))

    for i in range(n):
        c = int(case[i])
        snr = snr_by_case[c] + rng.uniform(-1.0, 1.0)
        snr_db[i] = snr
        sigma = 1.0 / np.sqrt(10 ** (snr / 10.0))

        d = int(rng.integers(5, D - 5))
        theta = angle_grid[d] + rng.uniform(-0.4, 0.4)
        y = np.exp(1j * rng.uniform(0, 2 * np.pi)) * steer(theta)
        gt_angle_idx[i, 0] = d
        gt_angles_deg[i, 0] = theta
        n_targets[i] = 1

        lo, hi = clutter_amp_by_case.get(c, (0.0, 0.0))
        for _ in range(n_clutter_by_case[c]):
            ct = rng.uniform(angle_min, angle_max)
            y = y + rng.uniform(lo, hi) * np.exp(1j * rng.uniform(0, 2 * np.pi)) * steer(ct)

        y_meas[i] = y + _cnoise(rng, M, sigma)
        A_out[i] = A_dict

    gt_spectrum = np.zeros((n, D), dtype=np.float32)
    for i in range(n):
        gt_spectrum[i] = _gaussian_bump(gt_angle_idx[i], D, sigma=target_sigma)

    return {
        "y_meas": y_meas.astype(np.complex64),
        "A_dict": A_out.astype(np.complex64),
        "gt_angle_idx": gt_angle_idx,
        "gt_angles_deg": gt_angles_deg.astype(np.float32),
        "gt_spectrum": gt_spectrum,
        "angle_grid_deg": angle_grid.astype(np.float32),
        "snr_db": snr_db.astype(np.float32),
        "num_elements": np.int64(M),
    }


def generate_detector_batch(
    n: int,
    rng: np.random.Generator,
    L: int = 128,
    balanced: bool = False,
) -> dict[str, np.ndarray]:
    """Build synthesized range-profile peak-detection training samples."""
    case = _case_layout(n, 5, rng, balanced)
    x = np.zeros((L, n), dtype=np.float64)
    gt_mask = np.zeros((L, n), dtype=np.float64)
    target_bins = np.full((n, 5), -1, dtype=np.int64)
    clutter_bounds = np.full((n, 2), -1, dtype=np.int64)
    n_targets = np.zeros(n, dtype=np.int64)
    snr_db = np.zeros(n, dtype=np.float64)

    snr_by_case = {0: 18.0, 1: 12.0, 2: 15.0, 3: 15.0, 4: 15.0}

    def add_target(prof: np.ndarray, b: int, snr_lin: float, sigma_local: float) -> None:
        amp = np.sqrt(snr_lin) * sigma_local
        for off, scale in ((0, 1.0), (-1, 0.7), (1, 0.7)):
            bl = b + off
            if 0 <= bl < L:
                prof[bl] += amp * scale * np.exp(1j * rng.uniform(0, 2 * np.pi))

    for i in range(n):
        c = int(case[i])
        snr = snr_by_case[c] + rng.uniform(-1.0, 1.0)
        snr_db[i] = snr
        snr_lin = 10 ** (snr / 10.0)

        sigma_l = np.ones(L)
        if c == 2:
            l0 = int(rng.integers(int(0.45 * L), int(0.65 * L)))
            sigma_l[l0:] = rng.uniform(2.0, 3.0)
            clutter_bounds[i] = (l0, L)

        prof = _cnoise(rng, L, sigma_l)

        if c == 3:
            b0 = int(rng.integers(20, L - 30))
            sep = int(rng.integers(3, 6))
            centers = [b0, min(b0 + sep, L - 2)]
        elif c == 2:
            centers = [min(clutter_bounds[i, 0] + int(rng.integers(2, 6)), L - 2)]
        else:
            k = int(rng.integers(1, 6)) if c in (0, 1) else 1
            centers = sorted(int(rng.integers(8, L - 8)) for _ in range(k))

        for j, b in enumerate(centers[:5]):
            local_sigma = sigma_l[b]
            add_target(prof, b, snr_lin, local_sigma)
            target_bins[i, j] = b
            for off in (-1, 0, 1):
                if 0 <= b + off < L:
                    gt_mask[b + off, i] = 1.0
        n_targets[i] = len(centers)

        if c == 4:
            for _ in range(int(rng.integers(2, 4))):
                bs = int(rng.integers(5, L - 5))
                if gt_mask[bs, i] == 0:
                    prof[bs] += rng.uniform(3.0, 5.0) * np.exp(1j * rng.uniform(0, 2 * np.pi))

        env = np.abs(prof)
        env_min = env.min()
        env_span = env.max() - env_min
        x[:, i] = (env - env_min) / env_span if env_span > 0.0 else 0.0

    return {
        "x": x.astype(np.float32),
        "gt_mask": gt_mask.astype(np.float32),
        "target_bins": target_bins,
        "snr_db": snr_db.astype(np.float32),
        "seq_len": np.int64(L),
    }


# --------------------------------------------------------------------------- #
# Paper Section 4 disk-backed synthetic dataset generators
# --------------------------------------------------------------------------- #
class FtDataGenerator:
    """Data generator for RF-LEGO Frequency Transform training samples.

    Constructs clean frequency-domain spectra, injects spectral leakage artifacts,
    adds AWGN across 5-40 dB SNR, and inverse-transforms to noisy time-domain signals.
    """

    def __init__(
        self,
        seq_len: int = 256,
        max_peaks: int = 3,
        snr_range: tuple[float, float] = (5.0, 40.0),
        leakage_probability: float = 0.5,
        save_dir: str | Path = "./data",
        total_num: int = 10000,
    ) -> None:
        self.seq_len = seq_len
        self.n_fft = seq_len
        self.max_peaks = max_peaks
        self.snr_range = snr_range
        self.leakage_probability = leakage_probability
        self.save_dir = str(save_dir)
        self.total_num = total_num

        os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)

    def generate_one_sample(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate one (noisy_signal, clean_spectrum) pair."""
        num_peaks = np.random.randint(1, self.max_peaks + 1)
        clean_spectrum = np.zeros(self.seq_len, dtype=np.complex64)
        peak_indices = np.random.choice(self.seq_len, size=num_peaks, replace=False)

        for peak_idx in peak_indices:
            amplitude = 0.5 + 0.5 * np.random.rand()
            phase = 2 * np.pi * np.random.rand()
            clean_spectrum[peak_idx] = amplitude * np.exp(1j * phase)

        if np.random.rand() < self.leakage_probability:
            leaky_spectrum = clean_spectrum.copy()
            for peak_idx in peak_indices:
                leakage_strength = 0.1 + 0.2 * np.random.rand()
                if peak_idx > 0:
                    leaky_spectrum[peak_idx - 1] += leakage_strength * clean_spectrum[peak_idx]
                if peak_idx < self.seq_len - 1:
                    leaky_spectrum[peak_idx + 1] += leakage_strength * clean_spectrum[peak_idx]
            working_spectrum = leaky_spectrum
        else:
            working_spectrum = clean_spectrum

        clean_signal = np.fft.ifft(working_spectrum, self.n_fft)
        snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        signal_power = np.mean(np.abs(clean_signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(self.seq_len) + 1j * np.random.randn(self.seq_len)
        )
        noisy_signal = clean_signal + noise
        return noisy_signal.astype(np.complex64), clean_spectrum

    def save_data(self, data_dict: dict, is_train: bool = True, idx: int | None = None) -> None:
        """Save a single sample dictionary as a .pkl file."""
        suffix = idx if idx is not None else int(time.time() * 1_000_000)
        folder = "train" if is_train else "test"
        file_path = os.path.join(self.save_dir, folder, f"sample_{suffix}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

    def generate_and_save_data(self) -> None:
        """Generate all samples and write an 80/20 train/test split to disk."""
        num_train = int(self.total_num * 0.8)
        logger.info(f"Generating {self.total_num} FT samples ({num_train} train)...")
        start_time = time.time()
        for i in range(self.total_num):
            noisy_signal, clean_spectrum = self.generate_one_sample()
            data_dict = {
                "clean_spectrum": clean_spectrum,
                "noisy_signal": noisy_signal,
                "noisy_spectrum": np.fft.fft(noisy_signal).astype(np.complex64),
                "clean_signal": np.fft.ifft(clean_spectrum).astype(np.complex64),
            }
            self.save_data(data_dict, is_train=(i < num_train), idx=i)
        logger.info(
            f"FT dataset saved to {self.save_dir} in {time.time() - start_time:.2f}s"
        )


class BeamformerDataGenerator:
    """Data generator for RF-LEGO Beamformer training samples.

    Simulates a uniform linear array with a steering dictionary (default 8 antennas,
    -60 to 60 degrees at 1-degree resolution) and sparse ground-truth DoA spectra.
    """

    def __init__(
        self,
        num_elements: int = 8,
        dictionary_length: int = 121,
        angle_range: tuple[float, float] = (-60.0, 60.0),
        gamma: float = 0.5,
        save_dir: str | Path = "./data",
    ) -> None:
        self.save_dir = str(save_dir)
        self.num_elements = num_elements
        self.dictionary_length = dictionary_length
        self.angle_range = angle_range
        self.gamma = gamma
        self.array = torch.arange(self.num_elements)
        self.dictionary = self.generate_dictionary()

    def _randu(self, shape: tuple[int, ...], a: float, b: float) -> torch.Tensor:
        return a + (b - a) * torch.rand(shape)

    def _randn(self, shape: tuple[int, ...], mean: float, std_dev: float) -> torch.Tensor:
        return mean + std_dev * torch.randn(shape)

    def _randi(self, low: int, high: int, k: int) -> torch.Tensor:
        if k > (high - low + 1):
            raise ValueError("k cannot be larger than the number of available integers.")
        all_integers = torch.arange(low, high + 1)
        return all_integers[torch.randperm(all_integers.size(0))][:k]

    def _idxf(self, freqs: torch.Tensor, frequency_grid: torch.Tensor) -> torch.Tensor:
        k_len = freqs.size(0)
        n_len = frequency_grid.size(0)
        closest_indices = torch.zeros(k_len, dtype=torch.long)
        for i in range(k_len):
            pos = torch.nonzero(frequency_grid >= freqs[i], as_tuple=True)[0]
            if pos.numel() == 0:
                closest_indices[i] = n_len - 1
            elif pos[0] == 0:
                closest_indices[i] = 0
            else:
                idx = pos[0]
                if torch.abs(frequency_grid[idx] - freqs[i]) < torch.abs(
                    frequency_grid[idx - 1] - freqs[i]
                ):
                    closest_indices[i] = idx
                else:
                    closest_indices[i] = idx - 1
        return torch.sort(closest_indices)[0]

    def _generate_random_angles(
        self,
        theta_min_deg: float = -60.0,
        theta_max_deg: float = 60.0,
        min_sep_deg: float = 2.0,
        gamma: float = 0.5,
        k: int | None = None,
        return_freqs: bool = True,
    ) -> torch.Tensor:
        if k is None:
            k = random.randint(1, 5)
        if (theta_max_deg - theta_min_deg) < (k - 1) * min_sep_deg:
            raise ValueError("Angle range too small for requested separation.")
        angles = torch.zeros(k)
        angles[0] = theta_min_deg + random.random() * (
            theta_max_deg - theta_min_deg - (k - 1) * min_sep_deg
        )
        for i in range(1, k):
            min_angle = angles[i - 1] + min_sep_deg
            remaining_range = theta_max_deg - (k - 1 - i) * min_sep_deg
            angles[i] = min_angle + random.random() * (remaining_range - min_angle)
        if return_freqs:
            return -gamma * torch.sin(torch.deg2rad(angles))
        return angles

    def generate_dictionary(self) -> dict[str, object]:
        """Create the ULA steering dictionary across angle_range."""
        angles = torch.linspace(self.angle_range[0], self.angle_range[1], self.dictionary_length)
        freqs = -self.gamma * torch.sin(torch.deg2rad(angles))
        steering = torch.exp(
            1j * 2 * torch.pi * self.array.unsqueeze(-1) * freqs.unsqueeze(0)
        ).to(torch.complex64)
        metadata = {
            "array_type": "ULA",
            "num_elements": self.num_elements,
            "dictionary_length": self.dictionary_length,
            "angle_grid": angles,
            "frequency_grid": freqs,
        }
        return {
            "metadata": metadata,
            "array": self.array,
            "angle_grid": angles,
            "frequency_grid": freqs,
            "dictionary": steering,
        }

    def single_measurement_vector(
        self,
        snr: float | list[float] | tuple[float, float],
        max_number_sources: int = 1,
        min_freq_separation: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate one noisy array snapshot y and sparse DoA label x."""
        num_signals = int(self._randi(1, max_number_sources, 1).item())
        freqs = self._generate_random_angles(
            theta_min_deg=self.angle_range[0],
            theta_max_deg=self.angle_range[1],
            min_sep_deg=min_freq_separation,
            gamma=self.gamma,
            k=num_signals,
        )
        phas = self._randu((num_signals, 1), -torch.pi, torch.pi)
        amps = torch.exp(1j * phas).to(torch.complex64)
        y = torch.matmul(
            torch.exp(1j * 2 * torch.pi * self.array.unsqueeze(-1) * freqs.unsqueeze(0)).to(
                torch.complex64
            ),
            amps,
        )
        sig_power = torch.mean(torch.abs(y) ** 2)
        snr_val = (
            float(self._randu((1,), snr[0], snr[1]).item())
            if isinstance(snr, (list, tuple))
            else float(snr)
        )
        sigma = float(torch.sqrt(sig_power / 2).item()) * (10 ** (-snr_val / 20))
        noise = self._randn((len(y), 1), 0.0, sigma) + 1j * self._randn((len(y), 1), 0.0, sigma)
        y = y + noise

        x = torch.zeros((self.dictionary_length, 1), dtype=torch.complex64)
        freq_grid = self.dictionary["frequency_grid"]
        assert isinstance(freq_grid, torch.Tensor)
        indices = self._idxf(freqs, freq_grid)
        x[indices] = amps
        return y.T, x.T

    def save_data(
        self,
        num_samples: int = 10000,
        snr_range: list[float] | tuple[float, float] = (5.0, 40.0),
        max_number_sources: int = 1,
        min_freq_separation: float = 10.0,
    ) -> None:
        """Generate num_samples snapshots and write an 80/20 train/test split."""
        os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)
        num_train = int(num_samples * 0.8)
        logger.info(f"Generating {num_samples} Beamformer samples ({num_train} train)...")
        for i in range(num_samples):
            y, x = self.single_measurement_vector(
                snr=snr_range,
                max_number_sources=max_number_sources,
                min_freq_separation=min_freq_separation,
            )
            data_dict = {"signal": y, "angle_spectrum": x, "dictionary": self.dictionary}
            folder = "train" if i < num_train else "test"
            file_path = os.path.join(self.save_dir, folder, f"sample_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(data_dict, f)
        logger.info(f"Beamformer dataset saved to {self.save_dir}")


def generate_sequence(
    seq_len: int = 128,
    num_targets_range: tuple[int, int] = (1, 6),
    snr_db_range: tuple[float, float] = (5.0, 40.0),
    window_size_range: tuple[int, int] = (20, 30),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 1D noisy signal sequence with Hann/Hamming peaks and binary peak labels."""
    noise_power = 1.0
    sequence = np.random.normal(scale=np.sqrt(noise_power / 2), size=seq_len)
    labels = np.zeros(seq_len, dtype=np.float32)

    num_targets = np.random.randint(*num_targets_range)
    peak_indices = np.random.choice(seq_len, size=num_targets, replace=False)
    labels[peak_indices] = 1.0

    for idx in peak_indices:
        win_size = np.random.randint(*window_size_range)
        win_type = np.random.choice(["hann", "hamming"])
        snr_db = np.random.uniform(*snr_db_range)
        amp = np.sqrt(noise_power * 10 ** (snr_db / 10))
        window = np.hanning(win_size) if win_type == "hann" else np.hamming(win_size)
        start = idx - win_size // 2
        for i, w in enumerate(window):
            j = start + i
            if 0 <= j < seq_len:
                sequence[j] += amp * w

    span = sequence.max() - sequence.min()
    sequence = (sequence - sequence.min()) / span if span > 0 else np.zeros_like(sequence)
    return sequence.astype(np.float32), labels


class DetectorDataGenerator:
    """Data generator for RF-LEGO Detector training samples.

    Synthesizes 1D signals of length seq_len (default 128) by superimposing
    1-5 Hann/Hamming peaks on unit-variance AWGN across 5-40 dB SNR.
    """

    def __init__(
        self,
        total_num: int = 10000,
        seq_len: int = 128,
        num_targets_range: tuple[int, int] = (1, 6),
        snr_db_range: tuple[float, float] = (5.0, 40.0),
        window_size_range: tuple[int, int] = (20, 30),
        save_dir: str | Path = "./data",
        train_ratio: float = 0.8,
    ) -> None:
        self.total_num = total_num
        self.seq_len = seq_len
        self.num_targets_range = num_targets_range
        self.snr_db_range = snr_db_range
        self.window_size_range = window_size_range
        self.save_dir = str(save_dir)
        self.train_ratio = train_ratio

        os.makedirs(os.path.join(self.save_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)

    def save_data(self, data_dict: dict, is_train: bool = True, idx: int | None = None) -> None:
        """Save a single sample dictionary as a .pkl file."""
        suffix = idx if idx is not None else int(time.time() * 1_000_000)
        folder = "train" if is_train else "test"
        path = os.path.join(self.save_dir, folder, f"sample_{suffix}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data_dict, f)

    def generate_and_save(self) -> None:
        """Generate all samples and write the train/test split to disk."""
        n_train = int(self.total_num * self.train_ratio)
        logger.info(f"Generating {self.total_num} Detector samples ({n_train} train)...")
        for i in range(self.total_num):
            noisy_signal, labels = generate_sequence(
                seq_len=self.seq_len,
                num_targets_range=self.num_targets_range,
                snr_db_range=self.snr_db_range,
                window_size_range=self.window_size_range,
            )
            self.save_data(
                {"noisy_signal": noisy_signal, "labels": labels},
                is_train=(i < n_train),
                idx=i,
            )
        logger.info(f"Detector dataset saved to {self.save_dir}")
