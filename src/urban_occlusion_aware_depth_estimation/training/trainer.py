"""Training loop and utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..evaluation.metrics import (
    CombinedLoss,
    compute_depth_metrics,
    compute_edge_metrics,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for edge-guided depth estimation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        mlflow_logger: Optional[Any] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            config: Configuration dictionary.
            device: Device to use.
            mlflow_logger: Optional MLflow logger.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.mlflow = mlflow_logger

        # Training config
        train_config = config['training']
        self.num_epochs = train_config['num_epochs']
        self.gradient_clip = train_config.get('gradient_clip', 1.0)
        self.mixed_precision = train_config.get('mixed_precision', True)

        # Loss function
        self.criterion = CombinedLoss(
            depth_weight=train_config.get('depth_loss_weight', 1.0),
            edge_weight=train_config.get('edge_loss_weight', 0.5),
            gradient_weight=train_config.get('gradient_loss_weight', 0.1),
            ssim_weight=train_config.get('ssim_loss_weight', 0.2),
        ).to(device)

        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None

        # Early stopping
        self.patience = train_config.get('patience', 15)
        self.min_delta = train_config.get('min_delta', 1e-4)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Checkpointing
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = train_config.get('save_frequency', 5)

        logger.info("Trainer initialized successfully")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()

        total_loss = 0.0
        total_depth_loss = 0.0
        total_edge_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            edge_gt = batch['edge'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.mixed_precision:
                with autocast():
                    predictions = self.model(images)
                    loss, loss_dict = self.criterion(
                        predictions,
                        {'depth': depth_gt, 'edge': edge_gt}
                    )

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss, loss_dict = self.criterion(
                    predictions,
                    {'depth': depth_gt, 'edge': edge_gt}
                )

                loss.backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss_dict['total_loss']
            total_depth_loss += loss_dict['depth_loss']
            total_edge_loss += loss_dict['edge_loss']

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'depth': f"{loss_dict['depth_loss']:.4f}",
                'edge': f"{loss_dict['edge_loss']:.4f}",
            })

        # Average metrics
        num_batches = len(self.train_loader)
        metrics = {
            'train_loss': total_loss / num_batches,
            'train_depth_loss': total_depth_loss / num_batches,
            'train_edge_loss': total_edge_loss / num_batches,
        }

        return metrics

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_depth_loss = 0.0
        total_edge_loss = 0.0

        all_depth_metrics = {
            'abs_rel': 0.0,
            'sq_rel': 0.0,
            'rmse': 0.0,
            'rmse_log': 0.0,
            'delta1': 0.0,
            'delta2': 0.0,
            'delta3': 0.0,
        }

        all_edge_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'iou': 0.0,
            'occlusion_recall': 0.0,
        }

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.num_epochs} [Val]")

        for batch in pbar:
            images = batch['image'].to(self.device)
            depth_gt = batch['depth'].to(self.device)
            edge_gt = batch['edge'].to(self.device)

            # Forward pass
            predictions = self.model(images)
            loss, loss_dict = self.criterion(
                predictions,
                {'depth': depth_gt, 'edge': edge_gt}
            )

            # Update losses
            total_loss += loss_dict['total_loss']
            total_depth_loss += loss_dict['depth_loss']
            total_edge_loss += loss_dict['edge_loss']

            # Compute metrics
            depth_metrics = compute_depth_metrics(
                predictions['depth'],
                depth_gt
            )
            edge_metrics = compute_edge_metrics(
                predictions['edge'],
                edge_gt
            )

            # Accumulate metrics
            for key in all_depth_metrics:
                all_depth_metrics[key] += depth_metrics[key]

            for key in all_edge_metrics:
                all_edge_metrics[key] += edge_metrics[key]

        # Average metrics
        num_batches = len(self.val_loader)
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_depth_loss': total_depth_loss / num_batches,
            'val_edge_loss': total_edge_loss / num_batches,
        }

        # Add averaged depth and edge metrics
        for key in all_depth_metrics:
            metrics[f'val_{key}'] = all_depth_metrics[key] / num_batches

        for key in all_edge_metrics:
            metrics[f'val_edge_{key}'] = all_edge_metrics[key] / num_batches

        return metrics

    def train(self) -> Dict[str, float]:
        """Train the model for all epochs.

        Returns:
            Dictionary of final metrics.
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Combine metrics
            all_metrics = {**train_metrics, **val_metrics}

            # Log metrics
            self._log_metrics(all_metrics, epoch)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                all_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Log epoch summary
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Abs Rel: {val_metrics['val_abs_rel']:.4f}, "
                f"Val Delta1: {val_metrics['val_delta1']:.4f}, "
                f"Val Edge F1: {val_metrics['val_edge_f1']:.4f}"
            )

            # Early stopping check
            if self._check_early_stopping(val_metrics['val_loss']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # Save checkpoint
            if epoch % self.save_frequency == 0 or epoch == self.num_epochs:
                self._save_checkpoint(epoch, all_metrics)

        logger.info("Training completed")
        return all_metrics

    def _log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics.
            epoch: Current epoch number.
        """
        if self.mlflow is not None:
            try:
                for key, value in metrics.items():
                    self.mlflow.log_metric(key, value, step=epoch)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered.

        Args:
            val_loss: Current validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return True
            return False

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if metrics.get('val_loss', float('inf')) <= self.best_val_loss:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")


# Alias for backwards compatibility
MultiTaskTrainer = Trainer
