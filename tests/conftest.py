"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_image():
    """Create sample image tensor."""
    return torch.randn(2, 3, 256, 512)


@pytest.fixture
def sample_depth():
    """Create sample depth tensor."""
    return torch.rand(2, 1, 256, 512)


@pytest.fixture
def sample_edge():
    """Create sample edge tensor."""
    return torch.rand(2, 1, 256, 512)


@pytest.fixture
def sample_batch():
    """Create sample batch."""
    return {
        'image': torch.randn(2, 3, 256, 512),
        'depth': torch.rand(2, 1, 256, 512),
        'edge': torch.rand(2, 1, 256, 512),
    }


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'seed': 42,
        'data': {
            'image_height': 256,
            'image_width': 512,
            'batch_size': 2,
            'num_workers': 0,
            'min_depth': 0.1,
            'max_depth': 80.0,
            'augmentation_prob': 0.5,
            'use_synthetic': True,
        },
        'model': {
            'encoder': 'resnet18',
            'pretrained': False,
            'decoder_channels': [128, 64, 32, 16],
            'use_edge_attention': True,
            'dropout': 0.1,
        },
        'training': {
            'num_epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'gradient_clip': 1.0,
            'mixed_precision': False,
            'depth_loss_weight': 1.0,
            'edge_loss_weight': 0.5,
            'gradient_loss_weight': 0.1,
            'ssim_loss_weight': 0.2,
            'patience': 5,
            'min_delta': 0.0001,
            'save_frequency': 1,
        },
        'optimizer': {
            'type': 'adam',
            'betas': [0.9, 0.999],
            'eps': 1e-8,
        },
        'scheduler': {
            'type': 'cosine',
            'T_max': 2,
            'eta_min': 1e-6,
        },
        'paths': {
            'data_root': './data',
            'checkpoint_dir': './test_checkpoints',
            'results_dir': './test_results',
            'log_dir': './test_logs',
        },
        'mlflow': {
            'enabled': False,
        },
    }


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create temporary checkpoint directory."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
