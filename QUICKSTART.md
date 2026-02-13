# Quick Start Guide

## Installation

```bash
# Clone or navigate to project
cd urban-occlusion-aware-depth-estimation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_project.py
```

## Training

### Basic Training
```bash
python scripts/train.py
```

### Custom Configuration
```bash
python scripts/train.py --config configs/default.yaml --device cuda
```

### Quick Test (2 epochs)
```bash
python scripts/train.py --config configs/quick_test.yaml
```

## Evaluation

```bash
# Evaluate best model
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

# Save prediction visualizations
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --save-predictions
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=urban_occlusion_aware_depth_estimation --cov-report=term-missing

# Run specific test file
pytest tests/test_model.py -v
```

## Python API Usage

### Basic Inference

```python
import torch
from urban_occlusion_aware_depth_estimation.models.model import EdgeGuidedDepthNet

# Create model
model = EdgeGuidedDepthNet(
    encoder_name='resnet50',
    pretrained=True,
    use_edge_attention=True
)

# Load checkpoint (optional)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

with torch.no_grad():
    image = torch.randn(1, 3, 256, 512).to(device)  # Your image here
    predictions = model(image)
    
    depth = predictions['depth']  # (1, 1, 256, 512)
    edges = predictions['edge']   # (1, 1, 256, 512)
```

### Custom Training Loop

```python
from urban_occlusion_aware_depth_estimation.data.loader import create_dataloaders
from urban_occlusion_aware_depth_estimation.models.model import create_model
from urban_occlusion_aware_depth_estimation.training.trainer import Trainer
from urban_occlusion_aware_depth_estimation.utils.config import load_config, set_seed, get_device
import torch.optim as optim

# Load config
config = load_config('configs/default.yaml')
set_seed(config['seed'])

# Setup
device = get_device('cuda')
train_loader, val_loader = create_dataloaders(config, use_synthetic=True)
model = create_model(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=None,
    config=config,
    device=device,
    mlflow_logger=None
)

# Train
metrics = trainer.train()
```

### Compute Metrics

```python
from urban_occlusion_aware_depth_estimation.evaluation.metrics import (
    compute_depth_metrics,
    compute_edge_metrics,
    MultiTaskMetrics
)

# Single batch
depth_metrics = compute_depth_metrics(pred_depth, target_depth)
edge_metrics = compute_edge_metrics(pred_edge, target_edge)

# Aggregate over dataset
aggregator = MultiTaskMetrics()
for batch in dataloader:
    predictions = model(batch['image'])
    aggregator.update(predictions, batch)

final_metrics = aggregator.compute()
print(f"Abs Rel Error: {final_metrics['abs_rel_error']:.4f}")
print(f"Delta 1 Accuracy: {final_metrics['delta_1_accuracy']:.4f}")
print(f"Boundary F1: {final_metrics['boundary_f1_score']:.4f}")
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model architecture
model:
  encoder: resnet50              # or resnet18, resnet101, efficientnet_b0
  use_edge_attention: true       # Enable edge-guided attention
  decoder_channels: [256, 128, 64, 32]

# Training hyperparameters
training:
  num_epochs: 50
  learning_rate: 0.0001
  batch_size: 8
  
# Data settings
data:
  image_height: 256
  image_width: 512
  use_synthetic: true            # Set false for KITTI/Cityscapes
```

## Expected Output

After training, you'll find:
- `checkpoints/` - Saved model weights
- `results/` - Metrics and visualizations
- `logs/` - Training logs
- `mlruns/` - MLflow experiment tracking

## Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size in config
data:
  batch_size: 4  # or 2
```

### Slow Training
```yaml
# Enable mixed precision
training:
  mixed_precision: true

# Reduce image size
data:
  image_height: 192
  image_width: 384
```

### MLflow Not Available
MLflow is optional. Training will continue with console logging if MLflow fails.

## Next Steps

1. Train on synthetic data: `python scripts/train.py --config configs/quick_test.yaml`
2. Review results in `results/final_metrics.json`
3. Customize architecture in `configs/default.yaml`
4. Add your own dataset in `src/data/loader.py`
5. Extend with new features

## Support

For issues, check:
- `verify_project.py` - Verify installation
- `tests/` - Run test suite to check components
- `PROJECT_SUMMARY.md` - Full technical details
