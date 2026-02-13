"""Tests for training functionality."""

import pytest
import torch
import tempfile
from pathlib import Path

from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
from urban_occlusion_aware_depth_estimation.models.model import create_model
from urban_occlusion_aware_depth_estimation.training.trainer import Trainer
from urban_occlusion_aware_depth_estimation.evaluation.metrics import (
    DepthLoss,
    EdgeLoss,
    CombinedLoss,
    compute_depth_metrics,
    compute_edge_metrics,
    DepthMetrics,
    BoundaryMetrics,
    MultiTaskMetrics,
)


class TestDepthLoss:
    """Tests for DepthLoss."""
    
    def test_forward_pass(self, sample_depth, device):
        """Test forward pass through depth loss."""
        loss_fn = DepthLoss().to(device)
        
        pred = sample_depth.to(device)
        target = sample_depth.to(device)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0.0
    
    def test_loss_decreases_with_better_prediction(self, device):
        """Test that loss is lower for better predictions."""
        loss_fn = DepthLoss().to(device)
        
        target = torch.rand(2, 1, 64, 64).to(device)
        good_pred = target + 0.01
        bad_pred = target + 0.5
        
        loss_good = loss_fn(good_pred, target)
        loss_bad = loss_fn(bad_pred, target)
        
        assert loss_good < loss_bad


class TestEdgeLoss:
    """Tests for EdgeLoss."""
    
    def test_forward_pass(self, sample_edge, device):
        """Test forward pass through edge loss."""
        loss_fn = EdgeLoss().to(device)
        
        # Note: EdgeLoss expects logits, not probabilities
        pred_logits = torch.randn(2, 1, 64, 64).to(device)
        target = sample_edge[:, :, :64, :64].to(device)
        
        loss = loss_fn(pred_logits, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0.0


class TestCombinedLoss:
    """Tests for CombinedLoss."""
    
    def test_forward_pass(self, device):
        """Test forward pass through combined loss."""
        loss_fn = CombinedLoss().to(device)
        
        pred = {
            'depth': torch.rand(2, 1, 64, 64).to(device),
            'edge': torch.rand(2, 1, 64, 64).to(device),
        }
        target = {
            'depth': torch.rand(2, 1, 64, 64).to(device),
            'edge': torch.rand(2, 1, 64, 64).to(device),
        }
        
        total_loss, loss_dict = loss_fn(pred, target)
        
        assert isinstance(total_loss, torch.Tensor)
        assert 'depth_loss' in loss_dict
        assert 'edge_loss' in loss_dict
        assert 'total_loss' in loss_dict


class TestMetrics:
    """Tests for metric computation."""
    
    def test_compute_depth_metrics(self, sample_depth, device):
        """Test depth metrics computation."""
        pred = sample_depth.to(device)
        target = sample_depth.to(device)
        
        metrics = compute_depth_metrics(pred, target)
        
        assert 'abs_rel' in metrics
        assert 'delta1' in metrics
        assert 'rmse' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_compute_edge_metrics(self, sample_edge, device):
        """Test edge metrics computation."""
        pred = sample_edge.to(device)
        target = sample_edge.to(device)
        
        metrics = compute_edge_metrics(pred, target)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_depth_metrics_aggregator(self, sample_depth, device):
        """Test DepthMetrics aggregator."""
        aggregator = DepthMetrics()
        
        pred = sample_depth.to(device)
        target = sample_depth.to(device)
        
        # Update multiple times
        aggregator.update(pred, target)
        aggregator.update(pred, target)
        
        metrics = aggregator.compute()
        
        assert 'abs_rel' in metrics
        assert 'delta1' in metrics
    
    def test_boundary_metrics_aggregator(self, sample_edge, device):
        """Test BoundaryMetrics aggregator."""
        aggregator = BoundaryMetrics()
        
        pred = sample_edge.to(device)
        target = sample_edge.to(device)
        
        aggregator.update(pred, target)
        aggregator.update(pred, target)
        
        metrics = aggregator.compute()
        
        assert 'precision' in metrics
        assert 'f1' in metrics
    
    def test_multitask_metrics(self, device):
        """Test MultiTaskMetrics."""
        aggregator = MultiTaskMetrics()
        
        pred = {
            'depth': torch.rand(2, 1, 64, 64).to(device),
            'edge': torch.rand(2, 1, 64, 64).to(device),
        }
        target = {
            'depth': torch.rand(2, 1, 64, 64).to(device),
            'edge': torch.rand(2, 1, 64, 64).to(device),
        }
        
        aggregator.update(pred, target)
        metrics = aggregator.compute()
        
        # Check target metrics are present
        assert 'abs_rel_error' in metrics
        assert 'delta_1_accuracy' in metrics
        assert 'boundary_f1_score' in metrics
        assert 'occlusion_edge_recall' in metrics


class TestTrainer:
    """Tests for Trainer class."""
    
    def test_trainer_initialization(self, test_config, device):
        """Test trainer can be initialized."""
        # Create components
        train_loader, val_loader = create_dataloaders(test_config, use_synthetic=True)
        model = create_model(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Update config for test
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config['paths']['checkpoint_dir'] = tmpdir
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                config=test_config,
                device=device,
                mlflow_logger=None,
            )
            
            assert trainer is not None
    
    def test_train_epoch(self, test_config, device):
        """Test training for one epoch."""
        # Create components
        train_loader, val_loader = create_dataloaders(test_config, use_synthetic=True)
        model = create_model(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config['paths']['checkpoint_dir'] = tmpdir
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                config=test_config,
                device=device,
                mlflow_logger=None,
            )
            
            metrics = trainer.train_epoch(epoch=1)
            
            assert 'train_loss' in metrics
            assert 'train_depth_loss' in metrics
            assert 'train_edge_loss' in metrics
    
    def test_validate(self, test_config, device):
        """Test validation."""
        # Create components
        train_loader, val_loader = create_dataloaders(test_config, use_synthetic=True)
        model = create_model(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config['paths']['checkpoint_dir'] = tmpdir
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                config=test_config,
                device=device,
                mlflow_logger=None,
            )
            
            metrics = trainer.validate(epoch=1)
            
            assert 'val_loss' in metrics
            assert 'val_abs_rel' in metrics
            assert 'val_delta1' in metrics
    
    def test_checkpoint_saving(self, test_config, device):
        """Test checkpoint saving."""
        train_loader, val_loader = create_dataloaders(test_config, use_synthetic=True)
        model = create_model(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_config['paths']['checkpoint_dir'] = tmpdir
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=None,
                config=test_config,
                device=device,
                mlflow_logger=None,
            )
            
            metrics = {'val_loss': 0.5}
            trainer._save_checkpoint(epoch=1, metrics=metrics)
            
            checkpoint_path = Path(tmpdir) / "checkpoint_epoch_1.pth"
            assert checkpoint_path.exists()
