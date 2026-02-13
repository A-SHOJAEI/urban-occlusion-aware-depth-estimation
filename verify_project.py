#!/usr/bin/env python
"""Comprehensive project verification script."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test all critical imports."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
        from urban_occlusion_aware_depth_estimation.models.model import create_model
        from urban_occlusion_aware_depth_estimation.training.trainer import Trainer
        from urban_occlusion_aware_depth_estimation.evaluation.metrics import MultiTaskMetrics
        from urban_occlusion_aware_depth_estimation.utils.config import load_config
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from urban_occlusion_aware_depth_estimation.utils.config import load_config
        config = load_config('configs/default.yaml')
        
        # Check required keys
        required_keys = ['data', 'model', 'training', 'paths']
        for key in required_keys:
            assert key in config, f"Missing config key: {key}"
        
        print("✓ Configuration loaded successfully")
        print(f"  - Epochs: {config['training']['num_epochs']}")
        print(f"  - Learning rate: {config['training']['learning_rate']}")
        print(f"  - Batch size: {config['data']['batch_size']}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_data_pipeline():
    """Test data loading pipeline."""
    print("\n" + "=" * 60)
    print("TESTING DATA PIPELINE")
    print("=" * 60)
    
    try:
        import torch
        from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
        from urban_occlusion_aware_depth_estimation.utils.config import load_config
        
        config = load_config('configs/quick_test.yaml')
        train_loader, val_loader = create_dataloaders(config, use_synthetic=True)
        
        # Get a batch
        batch = next(iter(train_loader))
        
        assert 'image' in batch, "Missing 'image' in batch"
        assert 'depth' in batch, "Missing 'depth' in batch"
        assert 'edge' in batch, "Missing 'edge' in batch"
        
        print("✓ Data pipeline working")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Batch shape: {batch['image'].shape}")
        return True
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """Test model creation and forward pass."""
    print("\n" + "=" * 60)
    print("TESTING MODEL")
    print("=" * 60)
    
    try:
        import torch
        from urban_occlusion_aware_depth_estimation.models.model import create_model
        from urban_occlusion_aware_depth_estimation.utils.config import load_config
        
        config = load_config('configs/quick_test.yaml')
        model = create_model(config)
        
        # Test forward pass
        device = torch.device('cpu')
        model = model.to(device)
        x = torch.randn(1, 3, 128, 256).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        assert 'depth' in output, "Missing 'depth' in output"
        assert 'edge' in output, "Missing 'edge' in output"
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✓ Model working")
        print(f"  - Parameters: {num_params:,}")
        print(f"  - Depth output shape: {output['depth'].shape}")
        print(f"  - Edge output shape: {output['edge'].shape}")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Test metrics computation."""
    print("\n" + "=" * 60)
    print("TESTING METRICS")
    print("=" * 60)
    
    try:
        import torch
        from urban_occlusion_aware_depth_estimation.evaluation.metrics import (
            compute_depth_metrics,
            compute_edge_metrics,
            MultiTaskMetrics,
        )
        
        # Create dummy predictions
        pred_depth = torch.rand(2, 1, 64, 64)
        target_depth = torch.rand(2, 1, 64, 64)
        pred_edge = torch.rand(2, 1, 64, 64)
        target_edge = torch.rand(2, 1, 64, 64)
        
        # Test depth metrics
        depth_metrics = compute_depth_metrics(pred_depth, target_depth)
        assert 'abs_rel' in depth_metrics
        assert 'delta1' in depth_metrics
        
        # Test edge metrics
        edge_metrics = compute_edge_metrics(pred_edge, target_edge)
        assert 'f1' in edge_metrics
        assert 'precision' in edge_metrics
        
        # Test multi-task metrics
        mt_metrics = MultiTaskMetrics()
        mt_metrics.update(
            {'depth': pred_depth, 'edge': pred_edge},
            {'depth': target_depth, 'edge': target_edge}
        )
        final_metrics = mt_metrics.compute()
        
        # Check target metrics are present
        assert 'abs_rel_error' in final_metrics
        assert 'delta_1_accuracy' in final_metrics
        assert 'boundary_f1_score' in final_metrics
        assert 'occlusion_edge_recall' in final_metrics
        
        print("✓ Metrics working")
        print(f"  - Depth metrics: {len(depth_metrics)} computed")
        print(f"  - Edge metrics: {len(edge_metrics)} computed")
        print(f"  - Target metrics: abs_rel_error={final_metrics['abs_rel_error']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("URBAN OCCLUSION-AWARE DEPTH ESTIMATION")
    print("PROJECT VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_data_pipeline,
        test_model,
        test_metrics,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe project is ready for training!")
        print("Run: python scripts/train.py --config configs/default.yaml")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
