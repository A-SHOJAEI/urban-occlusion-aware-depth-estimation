"""Edge-guided depth estimation model architecture."""

import logging
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EdgeGuidedAttention(nn.Module):
    """Edge-guided spatial attention mechanism."""

    def __init__(self, channels: int, reduction: int = 8):
        """Initialize edge-guided attention.

        Args:
            channels: Number of input channels.
            reduction: Channel reduction factor.
        """
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, channels // reduction, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 3, padding=1),
        )
        self.feature_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, edge_map: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Feature tensor (B, C, H, W).
            edge_map: Edge prediction (B, 1, H, W).

        Returns:
            Attention-weighted features (B, C, H, W).
        """
        # Resize edge map to match feature size
        edge_resized = F.interpolate(edge_map, size=features.shape[2:], mode='bilinear', align_corners=False)

        # Compute edge attention
        edge_attn = self.edge_conv(edge_resized)

        # Compute feature attention
        feat_attn = self.feature_conv(features)

        # Combine attentions
        attention = self.sigmoid(edge_attn + feat_attn)

        return features * attention


class DecoderBlock(nn.Module):
    """Decoder block with skip connections."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_attention: bool = False,
    ):
        """Initialize decoder block.

        Args:
            in_channels: Number of input channels.
            skip_channels: Number of skip connection channels.
            out_channels: Number of output channels.
            use_attention: Whether to use edge-guided attention.
        """
        super().__init__()
        self.use_attention = use_attention

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        if use_attention:
            self.attention = EdgeGuidedAttention(out_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        edge_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C_in, H, W).
            skip: Skip connection tensor (B, C_skip, H*2, W*2).
            edge_map: Optional edge map for attention (B, 1, H_out, W_out).

        Returns:
            Output tensor (B, C_out, H*2, W*2).
        """
        x = self.upconv(x)

        # Apply attention if enabled
        if self.use_attention and edge_map is not None:
            x = self.attention(x, edge_map)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class EdgeGuidedDepthNet(nn.Module):
    """Multi-task network for depth and edge prediction with edge-guided attention."""

    def __init__(
        self,
        encoder_name: str = 'resnet50',
        pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_edge_attention: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize edge-guided depth network.

        Args:
            encoder_name: Name of timm encoder.
            pretrained: Whether to use pretrained weights.
            decoder_channels: List of decoder channel sizes.
            use_edge_attention: Whether to use edge-guided attention.
            dropout: Dropout rate.
        """
        super().__init__()

        # Create encoder
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Get encoder channel dimensions
        encoder_channels = self.encoder.feature_info.channels()

        # Edge prediction branch (lightweight)
        self.edge_branch = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

        # Depth decoder with edge-guided attention
        self.use_edge_attention = use_edge_attention

        self.decoder_blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]

        for i, out_ch in enumerate(decoder_channels):
            skip_ch = encoder_channels[-(i+2)] if i < len(encoder_channels) - 1 else 0
            use_attn = use_edge_attention and i < 2  # Apply attention in earlier layers

            self.decoder_blocks.append(
                DecoderBlock(in_ch, skip_ch, out_ch, use_attention=use_attn)
            )
            in_ch = out_ch

        # Final depth prediction
        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

        logger.info(f"Initialized EdgeGuidedDepthNet with encoder={encoder_name}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, H, W).

        Returns:
            Dictionary with 'depth' and 'edge' predictions.
        """
        input_size = x.shape[2:]

        # Encoder
        features = self.encoder(x)

        # Edge prediction from deepest features
        edge_pred = self.edge_branch(features[-1])
        edge_pred = F.interpolate(edge_pred, size=input_size, mode='bilinear', align_corners=False)
        edge_pred = torch.sigmoid(edge_pred)

        # Depth decoder with edge guidance
        depth_feat = features[-1]

        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = -(i + 2)
            skip = features[skip_idx] if -skip_idx <= len(features) else None

            if skip is not None:
                depth_feat = decoder_block(depth_feat, skip, edge_pred)
            else:
                # Last block without skip connection
                depth_feat = decoder_block.upconv(depth_feat)
                depth_feat = decoder_block.conv(depth_feat)

        # Final depth prediction
        depth_pred = self.depth_head(depth_feat)
        depth_pred = F.interpolate(depth_pred, size=input_size, mode='bilinear', align_corners=False)

        return {
            'depth': depth_pred,
            'edge': edge_pred,
        }


def create_model(config: Dict) -> nn.Module:
    """Create model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Initialized model.
    """
    model_config = config['model']

    model = EdgeGuidedDepthNet(
        encoder_name=model_config.get('encoder', 'resnet50'),
        pretrained=model_config.get('pretrained', True),
        decoder_channels=model_config.get('decoder_channels', [256, 128, 64, 32]),
        use_edge_attention=model_config.get('use_edge_attention', True),
        dropout=model_config.get('dropout', 0.1),
    )

    return model
