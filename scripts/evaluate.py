#!/usr/bin/env python
"""Evaluation script for edge-guided depth estimation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from tqdm import tqdm

from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
from urban_occlusion_aware_depth_estimation.models.model import create_model
from urban_occlusion_aware_depth_estimation.evaluation.metrics import MultiTaskMetrics
from urban_occlusion_aware_depth_estimation.utils.config import (
    load_config,
    setup_logging,
    get_device,
)

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to load weights into.
        device: Device to load model on.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_predictions: bool = False,
    output_dir: Path = None,
) -> dict:
    """Evaluate model on dataset.
    
    Args:
        model: Model to evaluate.
        data_loader: Data loader for evaluation.
        device: Device to run evaluation on.
        save_predictions: Whether to save predictions.
        output_dir: Directory to save predictions.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    metrics_aggregator = MultiTaskMetrics()
    
    logger.info("Running evaluation...")
    pbar = tqdm(data_loader, desc="Evaluating")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        edge_gt = batch['edge'].to(device)
        
        # Forward pass
        predictions = model(images)
        
        # Update metrics
        metrics_aggregator.update(
            pred=predictions,
            target={'depth': depth_gt, 'edge': edge_gt}
        )
        
        # Save predictions if requested
        if save_predictions and output_dir is not None and batch_idx < 10:
            save_batch_predictions(
                images, predictions, batch, batch_idx, output_dir
            )
    
    # Compute final metrics
    final_metrics = metrics_aggregator.compute()
    
    return final_metrics


def save_batch_predictions(
    images: torch.Tensor,
    predictions: dict,
    batch: dict,
    batch_idx: int,
    output_dir: Path,
):
    """Save predictions for a batch.
    
    Args:
        images: Input images.
        predictions: Model predictions.
        batch: Ground truth batch.
        batch_idx: Batch index.
        output_dir: Output directory.
    """
    try:
        import matplotlib.pyplot as plt
        
        pred_dir = output_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Save only first sample in batch
        pred_depth = predictions['depth'][0, 0].cpu().numpy()
        pred_edge = predictions['edge'][0, 0].cpu().numpy()
        gt_depth = batch['depth'][0, 0].cpu().numpy()
        gt_edge = batch['edge'][0, 0].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(pred_depth, cmap='plasma')
        axes[0, 0].set_title('Predicted Depth')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_depth, cmap='plasma')
        axes[0, 1].set_title('Ground Truth Depth')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pred_edge, cmap='gray')
        axes[1, 0].set_title('Predicted Edges')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(gt_edge, cmap='gray')
        axes[1, 1].set_title('Ground Truth Edges')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(pred_dir / f'sample_{batch_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
    except Exception as e:
        logger.warning(f"Failed to save predictions: {e}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate edge-guided depth estimation model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save prediction visualizations'
    )
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging()
        logger.info("Starting evaluation pipeline")
        
        # Load configuration
        config = load_config(args.config)
        
        # Get device
        device = get_device(args.device)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        use_synthetic = config['data'].get('use_synthetic', True)
        _, val_loader = create_dataloaders(config, use_synthetic=use_synthetic)
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        
        # Load checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        load_checkpoint(str(checkpoint_path), model, device)
        
        # Evaluate
        results_dir = Path(config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            save_predictions=args.save_predictions,
            output_dir=results_dir,
        )
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        
        # Print key metrics
        key_metrics = [
            'abs_rel_error',
            'delta_1_accuracy',
            'boundary_f1_score',
            'occlusion_edge_recall',
        ]
        
        for metric in key_metrics:
            if metric in metrics:
                logger.info(f"{metric}: {metrics[metric]:.4f}")
        
        logger.info("=" * 50)
        logger.info("All metrics:")
        for key, value in sorted(metrics.items()):
            logger.info(f"  {key}: {value:.4f}")
        
        # Save metrics
        metrics_file = results_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
