"""Tests for model architecture."""

import pytest
import torch

from urban_occlusion_aware_depth_estimation.models.model import (
    EdgeGuidedAttention,
    DecoderBlock,
    EdgeGuidedDepthNet,
    create_model,
)


class TestEdgeGuidedAttention:
    """Tests for EdgeGuidedAttention module."""
    
    def test_forward_pass(self, device):
        """Test forward pass through attention module."""
        attn = EdgeGuidedAttention(channels=64).to(device)
        
        features = torch.randn(2, 64, 32, 32).to(device)
        edge_map = torch.rand(2, 1, 256, 256).to(device)
        
        output = attn(features, edge_map)
        
        assert output.shape == features.shape
        assert not torch.isnan(output).any()
    
    def test_attention_weights_in_range(self, device):
        """Test that attention weights are properly bounded."""
        attn = EdgeGuidedAttention(channels=64).to(device)
        
        features = torch.randn(2, 64, 32, 32).to(device)
        edge_map = torch.rand(2, 1, 32, 32).to(device)
        
        output = attn(features, edge_map)
        
        # Output should be bounded by input (due to sigmoid attention)
        assert output.abs().max() <= features.abs().max() * 2


class TestDecoderBlock:
    """Tests for DecoderBlock module."""
    
    def test_forward_without_attention(self, device):
        """Test decoder block without attention."""
        block = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            use_attention=False,
        ).to(device)
        
        x = torch.randn(2, 256, 16, 16).to(device)
        skip = torch.randn(2, 128, 32, 32).to(device)
        
        output = block(x, skip)
        
        assert output.shape == (2, 128, 32, 32)
        assert not torch.isnan(output).any()
    
    def test_forward_with_attention(self, device):
        """Test decoder block with attention."""
        block = DecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
            use_attention=True,
        ).to(device)
        
        x = torch.randn(2, 256, 16, 16).to(device)
        skip = torch.randn(2, 128, 32, 32).to(device)
        edge_map = torch.rand(2, 1, 64, 64).to(device)
        
        output = block(x, skip, edge_map)
        
        assert output.shape == (2, 128, 32, 32)
        assert not torch.isnan(output).any()


class TestEdgeGuidedDepthNet:
    """Tests for EdgeGuidedDepthNet model."""
    
    def test_model_creation(self, device):
        """Test model can be created."""
        model = EdgeGuidedDepthNet(
            encoder_name='resnet18',
            pretrained=False,
            decoder_channels=[128, 64, 32, 16],
        ).to(device)
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, device):
        """Test forward pass through model."""
        model = EdgeGuidedDepthNet(
            encoder_name='resnet18',
            pretrained=False,
            decoder_channels=[128, 64, 32, 16],
        ).to(device)
        
        x = torch.randn(2, 3, 256, 512).to(device)
        
        output = model(x)
        
        assert 'depth' in output
        assert 'edge' in output
        assert output['depth'].shape == (2, 1, 256, 512)
        assert output['edge'].shape == (2, 1, 256, 512)
        assert not torch.isnan(output['depth']).any()
        assert not torch.isnan(output['edge']).any()
    
    def test_output_ranges(self, device):
        """Test that outputs are in valid ranges."""
        model = EdgeGuidedDepthNet(
            encoder_name='resnet18',
            pretrained=False,
            decoder_channels=[128, 64, 32, 16],
        ).to(device)
        model.eval()
        
        x = torch.randn(2, 3, 256, 512).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        # Depth should be in [0, 1] due to sigmoid
        assert output['depth'].min() >= 0.0
        assert output['depth'].max() <= 1.0
        
        # Edge should be in [0, 1] due to sigmoid
        assert output['edge'].min() >= 0.0
        assert output['edge'].max() <= 1.0
    
    def test_gradient_flow(self, device):
        """Test that gradients flow properly."""
        model = EdgeGuidedDepthNet(
            encoder_name='resnet18',
            pretrained=False,
            decoder_channels=[128, 64, 32, 16],
        ).to(device)
        
        x = torch.randn(2, 3, 256, 512).to(device)
        
        output = model(x)
        loss = output['depth'].mean() + output['edge'].mean()
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients


class TestCreateModel:
    """Tests for model creation from config."""
    
    def test_create_model_from_config(self, test_config, device):
        """Test creating model from configuration."""
        model = create_model(test_config)
        model = model.to(device)
        
        x = torch.randn(2, 3, 256, 512).to(device)
        output = model(x)
        
        assert 'depth' in output
        assert 'edge' in output
    
    def test_create_model_without_attention(self, test_config, device):
        """Test creating model without edge attention."""
        test_config['model']['use_edge_attention'] = False
        
        model = create_model(test_config)
        model = model.to(device)
        
        x = torch.randn(2, 3, 256, 512).to(device)
        output = model(x)
        
        assert 'depth' in output
        assert 'edge' in output
