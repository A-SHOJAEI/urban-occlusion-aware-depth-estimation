#!/usr/bin/env python
"""Training script for edge-guided depth estimation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim

from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
from urban_occlusion_aware_depth_estimation.models.model import create_model
from urban_occlusion_aware_depth_estimation.training.trainer import Trainer
from urban_occlusion_aware_depth_estimation.utils.config import (
    load_config,
    save_config,
    set_seed,
    setup_logging,
    get_device,
    count_parameters,
)

logger = logging.getLogger(__name__)


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration.
    
    Args:
        model: Model to optimize.
        config: Configuration dictionary.
        
    Returns:
        Optimizer instance.
    """
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', [0.9, 0.999]),
            eps=optimizer_config.get('eps', 1e-8),
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=optimizer_config.get('momentum', 0.9),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer instance.
        config: Configuration dictionary.
        
    Returns:
        Scheduler instance or None.
    """
    if 'scheduler' not in config:
        return None
    
    scheduler_config = config['scheduler']
    scheduler_type = scheduler_config.get('type', 'none').lower()
    
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 50),
            eta_min=scheduler_config.get('eta_min', 1e-6),
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1),
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 5),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def setup_mlflow(config: dict):
    """Setup MLflow tracking.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        MLflow client or None.
    """
    mlflow_config = config.get('mlflow', {})
    
    if not mlflow_config.get('enabled', False):
        logger.info("MLflow tracking disabled")
        return None
    
    try:
        import mlflow
        
        # Set tracking URI
        tracking_uri = mlflow_config.get('tracking_uri', './mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = mlflow_config.get('experiment_name', 'depth-estimation')
        mlflow.set_experiment(experiment_name)
        
        # Start run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            'model_encoder': config['model']['encoder'],
            'learning_rate': config['training']['learning_rate'],
            'batch_size': config['data']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
        })
        
        logger.info(f"MLflow tracking enabled: {experiment_name}")
        return mlflow
        
    except ImportError:
        logger.warning("MLflow not installed. Tracking disabled.")
        return None
    except Exception as e:
        logger.warning(f"Failed to setup MLflow: {e}")
        return None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train edge-guided depth estimation model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        log_dir = config['paths'].get('log_dir', './logs')
        setup_logging(log_dir=log_dir)
        logger.info("Starting training pipeline")
        
        # Set random seed
        seed = config.get('seed', 42)
        set_seed(seed)
        
        # Get device
        device = get_device(args.device)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        use_synthetic = config['data'].get('use_synthetic', True)
        train_loader, val_loader = create_dataloaders(config, use_synthetic=use_synthetic)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        model = model.to(device)
        
        num_params = count_parameters(model)
        logger.info(f"Model parameters: {num_params:,}")
        
        # Create optimizer
        logger.info("Creating optimizer...")
        optimizer = create_optimizer(model, config)
        
        # Create scheduler
        scheduler = create_scheduler(optimizer, config)
        if scheduler is not None:
            logger.info(f"Using scheduler: {config['scheduler']['type']}")
        
        # Setup MLflow
        mlflow = setup_mlflow(config)
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            mlflow_logger=mlflow,
        )
        
        # Train
        logger.info("Starting training...")
        final_metrics = trainer.train()
        
        # Log final metrics
        logger.info("Training completed!")
        logger.info("Final metrics:")
        for key, value in final_metrics.items():
            if 'val' in key:
                logger.info(f"  {key}: {value:.4f}")
        
        # Save final config
        results_dir = Path(config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, str(results_dir / 'config.yaml'))
        
        # Save metrics
        import json
        with open(results_dir / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        logger.info(f"Results saved to {results_dir}")
        
        # End MLflow run
        if mlflow is not None:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
