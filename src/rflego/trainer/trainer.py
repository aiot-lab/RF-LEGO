"""Base trainer class and training utilities for RF-LEGO modules.

This module provides a flexible training framework with support for:
- Step-based and epoch-based training
- Configurable learning rate schedulers
- Checkpoint saving and resumption
- TensorBoard logging
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rflego.config import SchedulerConfig, TrainerConfig
from rflego.modules.base import BaseModel
from rflego.utils import ensure_dir, get_device, logger, set_seed


class BaseTrainer(ABC):
    """Abstract base class for training RF-LEGO modules.

    Provides common training infrastructure including:
    - Optimizer and scheduler setup
    - Checkpoint management
    - Logging and metrics tracking

    Subclasses should implement:
    - `compute_loss`: Define the loss computation for specific model
    - `train_step`: Single training step logic (optional)
    - `validate_step`: Single validation step logic (optional)

    Attributes:
        model: The RF-LEGO model to train.
        config: Training configuration.
        device: Target device for training.
        optimizer: Training optimizer.
        scheduler: Learning rate scheduler.
        writer: TensorBoard writer for logging.
    """

    def __init__(
        self,
        model: BaseModel,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: Model to train.
            config: Training configuration.
            train_loader: DataLoader for training data.
            val_loader: Optional DataLoader for validation data.
        """
        self.config = config
        self.device = get_device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set random seed
        set_seed(config.seed)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler (will be set by subclass or default)
        self.scheduler: LRScheduler | None = None

        # Setup directories and logging
        self.checkpoint_name = self._generate_checkpoint_name()
        self.checkpoint_dir = ensure_dir(config.save_dir / self.checkpoint_name)
        self.writer = SummaryWriter(log_dir=str(config.log_dir / self.checkpoint_name))

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        # Log model info
        self.model.log_model_info()
        logger.info(f"Training on device: {self.device}")

    def _create_optimizer(self) -> Optimizer:
        """Create the optimizer based on configuration.

        Returns:
            Configured optimizer instance.
        """
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self, scheduler_config: SchedulerConfig) -> LRScheduler:
        """Create learning rate scheduler based on configuration.

        Args:
            scheduler_config: Scheduler configuration.

        Returns:
            Configured scheduler instance.
        """
        if scheduler_config.scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma,
            )
        elif scheduler_config.scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.t_max,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_config.scheduler_type}")

    def _generate_checkpoint_name(self) -> str:
        """Generate a descriptive checkpoint directory name.

        Returns:
            Checkpoint name string with timestamp and key hyperparameters.
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return (
            f"{timestamp}_"
            f"bs{self.config.batch_size}_"
            f"lr{self.config.learning_rate}_"
            f"wd{self.config.weight_decay}"
        )

    @abstractmethod
    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss for a single batch.

        Must be implemented by subclasses.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        raise NotImplementedError

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform a single training step.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.optimizer.zero_grad()

        loss, metrics = self.compute_loss(batch)

        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return metrics

    @torch.no_grad()
    def validate_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Perform a single validation step.

        Args:
            batch: Dictionary containing batch data.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        _, metrics = self.compute_loss(batch)
        return metrics

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of aggregated training metrics.
        """
        self.model.train()
        epoch_metrics: dict[str, list[float]] = {}

        for batch in self.train_loader:
            metrics = self.train_step(batch)

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)

            # Log step metrics
            for key, value in metrics.items():
                self.writer.add_scalar(f"Train/{key}", value, self.global_step)

            self.global_step += 1

            if self.global_step % 100 == 0:
                logger.info(f"Step {self.global_step}, Loss: {metrics.get('loss', 0.0):.6f}")

        # Average metrics
        return {key: sum(values) / len(values) for key, values in epoch_metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation on the validation set.

        Returns:
            Dictionary of aggregated validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_metrics: dict[str, list[float]] = {}

        for batch in self.val_loader:
            metrics = self.validate_step(batch)

            for key, value in metrics.items():
                if key not in val_metrics:
                    val_metrics[key] = []
                val_metrics[key].append(value)

        # Average metrics
        avg_metrics = {key: sum(values) / len(values) for key, values in val_metrics.items()}

        # Log validation metrics
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f"Val/{key}", value, self.global_step)

        return avg_metrics

    def save_checkpoint(self, filename: str = "last.pt", is_best: bool = False) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename.
            is_best: If True, also save as 'best.pt'.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        logger.debug(f"Checkpoint saved to {save_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved with val_loss: {self.best_val_loss:.6f}")

    def load_checkpoint(self, path: Path | str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {path}, resuming from epoch {self.current_epoch}")

    def fit(self) -> None:
        """Main training loop.

        Runs epoch-based training with validation and checkpointing.
        """
        logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self.train_epoch()
            logger.info(
                f"Epoch {epoch}/{self.config.epochs}, "
                f"Train Loss: {train_metrics.get('loss', 0.0):.6f}"
            )

            # Validation
            val_metrics = self.validate()
            if val_metrics:
                val_loss = val_metrics.get("loss", float("inf"))
                logger.info(f"Validation Loss: {val_loss:.6f}")

                # Save checkpoints
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                self.save_checkpoint("last.pt", is_best=is_best)
            else:
                self.save_checkpoint("last.pt")

        self.writer.close()
        logger.info("Training completed!")

    def fit_steps(self) -> None:
        """Step-based training loop.

        Runs training for a fixed number of steps with periodic validation.
        """
        logger.info(f"Starting training for {self.config.total_steps} steps")

        while self.global_step < self.config.total_steps:
            self.model.train()

            for batch in self.train_loader:
                if self.global_step >= self.config.total_steps:
                    break

                metrics = self.train_step(batch)

                # Log metrics
                for key, value in metrics.items():
                    self.writer.add_scalar(f"Train/{key}", value, self.global_step)

                if self.global_step % 100 == 0:
                    logger.info(
                        f"Step {self.global_step}/{self.config.total_steps}, "
                        f"Loss: {metrics.get('loss', 0.0):.6f}"
                    )

                self.global_step += 1

            # Validation after each epoch through data
            val_metrics = self.validate()
            if val_metrics:
                val_loss = val_metrics.get("loss", float("inf"))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                self.save_checkpoint("last.pt", is_best=is_best)

        self.writer.close()
        logger.info("Training completed!")
