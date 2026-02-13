# Urban Occlusion-Aware Depth Estimation

Multi-task learning system that jointly predicts monocular depth and occlusion boundaries in urban driving scenarios. Uses a novel edge-guided attention mechanism to handle challenging cases like transparent surfaces, reflections, and thin structures, addressing depth discontinuity prediction at object boundaries.

## Installation

```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

Optional arguments:
- `--config`: Path to configuration file (default: configs/default.yaml)
- `--device`: Device to use (cuda/cpu, default: cuda)

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

Optional arguments:
- `--config`: Path to configuration file
- `--checkpoint`: Path to model checkpoint
- `--save-predictions`: Save prediction visualizations

### Example Usage

```python
import torch
from urban_occlusion_aware_depth_estimation.models.model import EdgeGuidedDepthNet
from urban_occlusion_aware_depth_estimation.utils.config import load_config

# Load configuration
config = load_config('configs/default.yaml')

# Create model
model = EdgeGuidedDepthNet(
    encoder_name='resnet50',
    pretrained=True,
    use_edge_attention=True
)

# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

with torch.no_grad():
    image = torch.randn(1, 3, 256, 512).to(device)
    predictions = model(image)
    depth = predictions['depth']
    edges = predictions['edge']
```

## Architecture

The system uses a multi-task encoder-decoder architecture:

- **Encoder**: ResNet-based feature extractor with pretrained ImageNet weights
- **Edge Branch**: Lightweight network for occlusion boundary prediction
- **Edge-Guided Attention**: Novel attention mechanism that uses predicted edges to enhance depth estimation at object boundaries
- **Depth Decoder**: Progressive upsampling with skip connections and edge guidance

## Configuration

Key configuration parameters in `configs/default.yaml`:

- **Model**: Encoder type, decoder channels, attention settings
- **Data**: Image dimensions, batch size, augmentation
- **Training**: Learning rate, loss weights, early stopping
- **Paths**: Data root, checkpoint directory, results directory

## Results

Run `python scripts/train.py` to reproduce the following metrics:

| Metric | Target | Result |
|--------|--------|--------|
| Absolute Relative Error | 0.085 | TBD |
| Delta 1 Accuracy | 0.900 | TBD |
| Boundary F1 Score | 0.750 | TBD |
| Occlusion Edge Recall | 0.820 | TBD |

Results will be saved to the `results/` directory after training.

## Dataset

The system supports:
- **KITTI**: Depth estimation dataset for autonomous driving
- **Cityscapes**: Urban scene understanding with semantic labels
- **Synthetic**: Generated data for quick testing and development

Place datasets in the `data/` directory or configure paths in `configs/default.yaml`.

## Testing

Run the test suite:

```bash
pytest tests/ -v --cov=urban_occlusion_aware_depth_estimation
```

Expected coverage: >70%

## Project Structure

```
urban-occlusion-aware-depth-estimation/
├── src/urban_occlusion_aware_depth_estimation/
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── training/           # Training loop and utilities
│   ├── evaluation/         # Metrics and evaluation
│   └── utils/              # Configuration and helpers
├── tests/                  # Test suite
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── notebooks/              # Jupyter notebooks for exploration
└── requirements.txt        # Dependencies
```

## Technical Details

### Loss Functions

Multi-task loss combining:
- L1 depth loss
- Gradient matching for depth discontinuities
- SSIM for structural similarity
- Weighted binary cross-entropy for edge detection

### Data Augmentation

Training augmentations include:
- Random horizontal flip
- Color jittering (brightness, contrast, hue, saturation)
- Gaussian noise and blur
- ImageNet normalization

### Training Features

- Mixed precision training with automatic scaling
- Gradient clipping for stability
- Cosine annealing learning rate schedule
- Early stopping with patience
- MLflow experiment tracking
- Automatic checkpoint saving

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
