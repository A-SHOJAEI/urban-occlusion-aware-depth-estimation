"""Urban Occlusion-Aware Depth Estimation.

A multi-task learning system for monocular depth and occlusion boundary prediction
in urban driving scenarios with edge-guided attention mechanisms.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from urban_occlusion_aware_depth_estimation.models.model import (
    EdgeGuidedDepthNet,
    EdgeGuidedAttention,
    DecoderBlock,
    create_model,
)
from urban_occlusion_aware_depth_estimation.training.trainer import MultiTaskTrainer
from urban_occlusion_aware_depth_estimation.evaluation.metrics import (
    DepthMetrics,
    BoundaryMetrics,
    MultiTaskMetrics,
)

# Aliases for backwards compatibility
EdgeGuidedDepthEstimator = EdgeGuidedDepthNet
MultiTaskEncoder = EdgeGuidedDepthNet
DepthDecoder = DecoderBlock
OcclusionDecoder = DecoderBlock

__all__ = [
    "EdgeGuidedDepthNet",
    "EdgeGuidedDepthEstimator",
    "MultiTaskEncoder",
    "DepthDecoder",
    "OcclusionDecoder",
    "EdgeGuidedAttention",
    "DecoderBlock",
    "MultiTaskTrainer",
    "DepthMetrics",
    "BoundaryMetrics",
    "MultiTaskMetrics",
    "create_model",
]
