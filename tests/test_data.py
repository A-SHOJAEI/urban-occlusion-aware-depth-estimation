"""Tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np

from urban_occlusion_aware_depth_estimation.data.loader import (
    SyntheticUrbanDataset,
    KITTIDepthDataset,
    create_dataloaders,
)
from urban_occlusion_aware_depth_estimation.data.preprocessing import (
    DepthNormalizer,
    compute_edge_map,
    get_train_augmentation,
    get_val_augmentation,
    SyntheticDataGenerator,
)


class TestDepthNormalizer:
    """Tests for DepthNormalizer."""
    
    def test_normalize(self):
        """Test depth normalization."""
        normalizer = DepthNormalizer(min_depth=0.1, max_depth=80.0)
        depth = np.array([[10.0, 20.0], [40.0, 80.0]])
        
        normalized = normalizer.normalize(depth)
        
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert normalized.dtype == np.float32
    
    def test_denormalize(self):
        """Test depth denormalization."""
        normalizer = DepthNormalizer(min_depth=0.1, max_depth=80.0)
        normalized = np.array([[0.0, 0.5], [0.75, 1.0]])
        
        denormalized = normalizer.denormalize(normalized)
        
        assert denormalized.min() >= normalizer.min_depth
        assert denormalized.max() <= normalizer.max_depth
    
    def test_normalize_denormalize_inverse(self):
        """Test that normalize and denormalize are inverses."""
        normalizer = DepthNormalizer(min_depth=0.1, max_depth=80.0)
        original = np.array([[10.0, 20.0], [40.0, 80.0]])
        
        normalized = normalizer.normalize(original)
        recovered = normalizer.denormalize(normalized)
        
        np.testing.assert_allclose(original, recovered, rtol=1e-5)


class TestEdgeMap:
    """Tests for edge map computation."""
    
    def test_compute_edge_map(self):
        """Test edge map computation."""
        depth = np.random.rand(256, 512).astype(np.float32)
        
        edges = compute_edge_map(depth, threshold=0.1)
        
        assert edges.shape == depth.shape
        assert edges.dtype == np.float32
        assert edges.min() >= 0.0
        assert edges.max() <= 1.0
    
    def test_edge_map_detects_discontinuities(self):
        """Test that edge map detects depth discontinuities."""
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[:, 50:] = 1.0  # Create a vertical edge
        
        edges = compute_edge_map(depth, threshold=0.1)
        
        # Edge should be detected around column 50
        assert edges[:, 45:55].sum() > edges[:, :20].sum()


class TestSyntheticDataGenerator:
    """Tests for synthetic data generator."""
    
    def test_generate_planar_scene(self):
        """Test planar scene generation."""
        generator = SyntheticDataGenerator(height=256, width=512)
        
        image, depth, edges = generator.generate_planar_scene()
        
        assert image.shape == (256, 512, 3)
        assert depth.shape == (256, 512)
        assert edges.shape == (256, 512)
        assert image.dtype == np.uint8
        assert depth.dtype == np.float32
        assert edges.dtype == np.float32
    
    def test_generate_batch(self):
        """Test batch generation."""
        generator = SyntheticDataGenerator(height=256, width=512)
        
        batch = generator.generate_batch(batch_size=4)
        
        assert batch['image'].shape == (4, 3, 256, 512)
        assert batch['depth'].shape == (4, 1, 256, 512)
        assert batch['edge'].shape == (4, 1, 256, 512)


class TestDatasets:
    """Tests for dataset classes."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset."""
        dataset = SyntheticUrbanDataset(num_samples=10, height=256, width=512)
        
        assert len(dataset) == 10
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'depth' in sample
        assert 'edge' in sample
    
    def test_kitti_dataset_fallback_to_synthetic(self):
        """Test KITTI dataset falls back to synthetic."""
        dataset = KITTIDepthDataset(
            data_root='/nonexistent/path',
            split='train',
            height=256,
            width=512,
        )
        
        # Should use synthetic data
        assert dataset.use_synthetic is True
        assert len(dataset) > 0
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'depth' in sample
        assert 'edge' in sample


class TestDataLoaders:
    """Tests for dataloader creation."""
    
    def test_create_dataloaders(self, test_config):
        """Test dataloader creation."""
        train_loader, val_loader = create_dataloaders(test_config, use_synthetic=True)
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Test batch loading
        batch = next(iter(train_loader))
        assert 'image' in batch
        assert 'depth' in batch
        assert 'edge' in batch
        assert batch['image'].shape[0] == test_config['data']['batch_size']


class TestAugmentations:
    """Tests for data augmentations."""
    
    def test_train_augmentation(self):
        """Test training augmentation pipeline."""
        transform = get_train_augmentation(height=256, width=512)
        
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        depth = np.random.rand(128, 128).astype(np.float32)
        edge = np.random.rand(128, 128).astype(np.float32)
        
        transformed = transform(image=image, depth=depth, edge=edge)
        
        assert transformed['image'].shape == (3, 256, 512)
        assert isinstance(transformed['image'], torch.Tensor)
    
    def test_val_augmentation(self):
        """Test validation augmentation pipeline."""
        transform = get_val_augmentation(height=256, width=512)
        
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        depth = np.random.rand(128, 128).astype(np.float32)
        edge = np.random.rand(128, 128).astype(np.float32)
        
        transformed = transform(image=image, depth=depth, edge=edge)
        
        assert transformed['image'].shape == (3, 256, 512)
        assert isinstance(transformed['image'], torch.Tensor)
