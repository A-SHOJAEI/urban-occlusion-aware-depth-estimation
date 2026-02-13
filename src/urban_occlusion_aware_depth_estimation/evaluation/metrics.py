"""Evaluation metrics for depth and edge prediction."""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
) -> Dict[str, float]:
    """Compute depth estimation metrics.

    Args:
        pred: Predicted depth (B, 1, H, W) or (B, H, W).
        target: Ground truth depth (B, 1, H, W) or (B, H, W).
        mask: Optional valid pixel mask (B, 1, H, W) or (B, H, W).

    Returns:
        Dictionary of metric values.
    """
    # Ensure same shape
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Create mask if not provided
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        mask = mask.bool()

    # Apply mask
    pred_masked = pred[mask]
    target_masked = target[mask]

    # Avoid division by zero
    eps = 1e-6
    target_masked = torch.clamp(target_masked, min=eps)
    pred_masked = torch.clamp(pred_masked, min=eps)

    # Absolute relative error
    abs_rel = torch.mean(torch.abs(pred_masked - target_masked) / target_masked)

    # Squared relative error
    sq_rel = torch.mean(((pred_masked - target_masked) ** 2) / target_masked)

    # RMSE
    rmse = torch.sqrt(torch.mean((pred_masked - target_masked) ** 2))

    # RMSE log
    rmse_log = torch.sqrt(torch.mean((torch.log(pred_masked) - torch.log(target_masked)) ** 2))

    # Threshold accuracy
    thresh = torch.max(pred_masked / target_masked, target_masked / pred_masked)
    delta1 = torch.mean((thresh < 1.25).float())
    delta2 = torch.mean((thresh < 1.25 ** 2).float())
    delta3 = torch.mean((thresh < 1.25 ** 3).float())

    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'rmse_log': rmse_log.item(),
        'delta1': delta1.item(),
        'delta2': delta2.item(),
        'delta3': delta3.item(),
    }


def compute_edge_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute edge/boundary detection metrics.

    Args:
        pred: Predicted edge probabilities (B, 1, H, W) or (B, H, W).
        target: Ground truth edges (B, 1, H, W) or (B, H, W).
        threshold: Threshold for binary classification.

    Returns:
        Dictionary of metric values.
    """
    # Ensure same shape
    if pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)

    # Binarize predictions
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Flatten
    pred_flat = pred_binary.reshape(-1)
    target_flat = target_binary.reshape(-1)

    # Compute confusion matrix elements
    tp = torch.sum((pred_flat == 1) & (target_flat == 1)).float()
    fp = torch.sum((pred_flat == 1) & (target_flat == 0)).float()
    fn = torch.sum((pred_flat == 0) & (target_flat == 1)).float()
    tn = torch.sum((pred_flat == 0) & (target_flat == 0)).float()

    # Precision, recall, F1
    eps = 1e-6
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    # IoU
    iou = tp / (tp + fp + fn + eps)

    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item(),
        'occlusion_recall': recall.item(),  # Alias for occlusion edge recall
    }


class DepthLoss(torch.nn.Module):
    """Combined depth loss with multiple components."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        gradient_weight: float = 0.1,
        ssim_weight: float = 0.2,
    ):
        """Initialize depth loss.

        Args:
            l1_weight: Weight for L1 loss.
            gradient_weight: Weight for gradient matching loss.
            ssim_weight: Weight for SSIM loss.
        """
        super().__init__()
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute depth loss.

        Args:
            pred: Predicted depth (B, 1, H, W).
            target: Ground truth depth (B, 1, H, W).

        Returns:
            Combined loss value.
        """
        # Ensure both tensors are 4D (B, 1, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # Gradient matching loss
        grad_loss = self._gradient_loss(pred, target)

        # SSIM loss
        ssim_loss = 1 - self._ssim(pred, target)

        # Combined loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.gradient_weight * grad_loss +
            self.ssim_weight * ssim_loss
        )

        return total_loss

    def _gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute gradient matching loss.

        Args:
            pred: Predicted depth.
            target: Ground truth depth.

        Returns:
            Gradient loss value.
        """
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        # L1 loss on gradients
        loss_dx = F.l1_loss(pred_dx, target_dx)
        loss_dy = F.l1_loss(pred_dy, target_dy)

        return loss_dx + loss_dy

    def _ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Compute SSIM.

        Args:
            pred: Predicted depth.
            target: Ground truth depth.
            window_size: Size of sliding window.

        Returns:
            SSIM value.
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        window = gauss / gauss.sum()
        window = window.unsqueeze(0)
        window = window.mm(window.t()).unsqueeze(0).unsqueeze(0)
        window = window.to(pred.device)

        # Compute means
        mu1 = F.conv2d(pred, window, padding=window_size // 2)
        mu2 = F.conv2d(target, window, padding=window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size // 2) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()


class EdgeLoss(torch.nn.Module):
    """Edge detection loss with class balancing."""

    def __init__(self, pos_weight: float = 2.0):
        """Initialize edge loss.

        Args:
            pos_weight: Weight for positive class (edges).
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge loss.

        Args:
            pred: Predicted edge probabilities (B, 1, H, W).
            target: Ground truth edges (B, 1, H, W).

        Returns:
            Loss value.
        """
        # Ensure both tensors are 4D (B, 1, H, W)
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Weighted binary cross entropy
        pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight
        )

        return loss


class CombinedLoss(torch.nn.Module):
    """Combined loss for multi-task learning."""

    def __init__(
        self,
        depth_weight: float = 1.0,
        edge_weight: float = 0.5,
        gradient_weight: float = 0.1,
        ssim_weight: float = 0.2,
    ):
        """Initialize combined loss.

        Args:
            depth_weight: Weight for depth loss.
            edge_weight: Weight for edge loss.
            gradient_weight: Weight for gradient matching.
            ssim_weight: Weight for SSIM.
        """
        super().__init__()
        self.depth_loss = DepthLoss(
            l1_weight=1.0,
            gradient_weight=gradient_weight,
            ssim_weight=ssim_weight,
        )
        self.edge_loss = EdgeLoss(pos_weight=2.0)
        self.depth_weight = depth_weight
        self.edge_weight = edge_weight

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            pred: Dictionary with 'depth' and 'edge' predictions.
            target: Dictionary with 'depth' and 'edge' ground truth.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Individual losses
        depth_loss = self.depth_loss(pred['depth'], target['depth'])
        edge_loss = self.edge_loss(pred['edge'], target['edge'])

        # Combined loss
        total_loss = (
            self.depth_weight * depth_loss +
            self.edge_weight * edge_loss
        )

        loss_dict = {
            'depth_loss': depth_loss.item(),
            'edge_loss': edge_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict


class DepthMetrics:
    """Depth evaluation metrics aggregator."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            'abs_rel': [],
            'sq_rel': [],
            'rmse': [],
            'rmse_log': [],
            'delta1': [],
            'delta2': [],
            'delta3': [],
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> None:
        """Update metrics with new predictions.

        Args:
            pred: Predicted depth.
            target: Ground truth depth.
            mask: Optional valid pixel mask.
        """
        batch_metrics = compute_depth_metrics(pred, target, mask)
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)

    def compute(self) -> Dict[str, float]:
        """Compute average metrics.

        Returns:
            Dictionary of averaged metrics.
        """
        return {key: np.mean(values) for key, values in self.metrics.items()}


class BoundaryMetrics:
    """Boundary/edge evaluation metrics aggregator."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'iou': [],
            'occlusion_recall': [],
        }

    def update(self, pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        """Update metrics with new predictions.

        Args:
            pred: Predicted edge probabilities.
            target: Ground truth edges.
            threshold: Classification threshold.
        """
        batch_metrics = compute_edge_metrics(pred, target, threshold)
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)

    def compute(self) -> Dict[str, float]:
        """Compute average metrics.

        Returns:
            Dictionary of averaged metrics.
        """
        return {key: np.mean(values) for key, values in self.metrics.items()}


class MultiTaskMetrics:
    """Combined metrics for multi-task learning."""

    def __init__(self):
        """Initialize multi-task metrics."""
        self.depth_metrics = DepthMetrics()
        self.boundary_metrics = BoundaryMetrics()

    def reset(self) -> None:
        """Reset all metrics."""
        self.depth_metrics.reset()
        self.boundary_metrics.reset()

    def update(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
    ) -> None:
        """Update all metrics.

        Args:
            pred: Dictionary with 'depth' and 'edge' predictions.
            target: Dictionary with 'depth' and 'edge' ground truth.
            mask: Optional valid pixel mask for depth.
        """
        self.depth_metrics.update(pred['depth'], target['depth'], mask)
        self.boundary_metrics.update(pred['edge'], target['edge'])

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary with all averaged metrics.
        """
        metrics = {}
        metrics.update(self.depth_metrics.compute())
        metrics.update(self.boundary_metrics.compute())

        # Add combined metrics with specific names for target metrics
        metrics['abs_rel_error'] = metrics['abs_rel']
        metrics['delta_1_accuracy'] = metrics['delta1']
        metrics['boundary_f1_score'] = metrics['f1']
        metrics['occlusion_edge_recall'] = metrics['occlusion_recall']

        return metrics
