#!/usr/bin/env python
"""Quick import test to verify all modules load correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing imports...")

# Test data modules
from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
from urban_occlusion_aware_depth_estimation.data.preprocessing import SyntheticDataGenerator
print("✓ Data modules imported successfully")

# Test model modules
from urban_occlusion_aware_depth_estimation.models.model import EdgeGuidedDepthNet, create_model
print("✓ Model modules imported successfully")

# Test training modules
from urban_occlusion_aware_depth_estimation.training.trainer import Trainer
print("✓ Training modules imported successfully")

# Test evaluation modules
from urban_occlusion_aware_depth_estimation.evaluation.metrics import (
    DepthMetrics,
    BoundaryMetrics,
    MultiTaskMetrics,
    CombinedLoss,
)
print("✓ Evaluation modules imported successfully")

# Test utils
from urban_occlusion_aware_depth_estimation.utils.config import (
    load_config,
    set_seed,
    get_device,
)
print("✓ Utils modules imported successfully")

print("\n✓ All imports successful!")
print("Project is ready to use.")
