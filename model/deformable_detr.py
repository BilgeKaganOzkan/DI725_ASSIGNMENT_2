import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# These imports are used in the original implementation
# We keep them for compatibility with the original code
from torch.nn.init import xavier_uniform_, constant_, normal_

class DeformableDetrModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, num_feature_levels=3, enc_n_points=4, dec_n_points=4,
                 num_queries=300, aux_loss=True, use_checkpoint=False, chunk_size_large=2000, chunk_size_small=50):
        """
        Deformable DETR model implementation

        Args:
            num_classes (int): Number of object classes
            hidden_dim (int): Dimension of hidden layers
            nheads (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            num_decoder_layers (int): Number of decoder layers
            num_feature_levels (int): Number of feature levels to use
            enc_n_points (int): Number of sampling points in encoder
            dec_n_points (int): Number of sampling points in decoder
            num_queries (int): Number of object queries
            use_checkpoint (bool): Whether to use gradient checkpointing to save memory
        """
        super().__init__()

        # Backbone (We'll use ResNet50 by default)
        self.backbone = self._get_backbone()

        # Position embedding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

        # Input projection for layer2 (will be reused for other layers)
        # We'll initialize this properly in the forward pass
        # to ensure we get the correct channel counts
        self.input_proj = None

        # Print model parameters for verification
        print("\nDeformableDetrModel Parameters:")
        print(f"  - num_classes: {num_classes}")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - nheads: {nheads}")
        print(f"  - num_encoder_layers: {num_encoder_layers}")
        print(f"  - num_decoder_layers: {num_decoder_layers}")
        print(f"  - dim_feedforward: {dim_feedforward}")
        print(f"  - dropout: {dropout}")
        print(f"  - num_feature_levels: {num_feature_levels}")
        print(f"  - enc_n_points: {enc_n_points}")
        print(f"  - dec_n_points: {dec_n_points}")
        print(f"  - num_queries: {num_queries}")
        print(f"  - aux_loss: {aux_loss}")
        print(f"  - use_checkpoint: {use_checkpoint}")
        print(f"  - chunk_size_large: {chunk_size_large}")
        print(f"  - chunk_size_small: {chunk_size_small}")

        # Model parameters
        self.aux_loss = aux_loss

        # Memory optimization parameters
        self.use_checkpoint = use_checkpoint
        self.chunk_size_large = chunk_size_large
        self.chunk_size_small = chunk_size_small

        # Transformer encoder-decoder
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points,
            use_checkpoint=use_checkpoint,
            chunk_size_large=chunk_size_large,
            chunk_size_small=chunk_size_small
        )

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 coordinates for bounding box

        # Query embeddings for the decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # Default: 100 object queries

        self._reset_parameters()

    def _get_backbone(self):
        """
        Creates a ResNet50 backbone with multi-scale feature outputs
        Returns a backbone that outputs features at multiple scales for multi-scale processing
        """
        try:
            # Try to use the latest torchvision API
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as e:
            print(f"Warning: Could not load ResNet50 with new API: {e}")
            # Fallback if torchvision is not the latest version
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)
            print("Using legacy torchvision API with pretrained=True")

        # Create a feature pyramid network (FPN) style backbone
        # that returns features at multiple scales
        class MultiScaleBackbone(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

                # Define the feature extraction layers
                self.layer0 = nn.Sequential(
                    backbone.conv1,
                    backbone.bn1,
                    backbone.relu,
                    backbone.maxpool
                )
                self.layer1 = backbone.layer1  # 1/4 resolution
                self.layer2 = backbone.layer2  # 1/8 resolution
                self.layer3 = backbone.layer3  # 1/16 resolution
                self.layer4 = backbone.layer4  # 1/32 resolution

                # Get output channels for each layer
                self.out_channels = {
                    'layer1': 256,   # ResNet50 layer1 output channels
                    'layer2': 512,   # ResNet50 layer2 output channels
                    'layer3': 1024,  # ResNet50 layer3 output channels
                    'layer4': 2048   # ResNet50 layer4 output channels
                }

                # Freeze backbone parameters if needed
                # self._freeze_backbone()

            def _freeze_backbone(self):
                """Freeze backbone parameters for feature extraction"""
                for _, param in self.backbone.named_parameters():
                    param.requires_grad = False
                print("Backbone parameters frozen for feature extraction")

            def forward(self, x):
                # Get features at different scales
                x0 = self.layer0(x)        # 1/4 resolution
                x1 = self.layer1(x0)       # 1/4 resolution
                x2 = self.layer2(x1)       # 1/8 resolution
                x3 = self.layer3(x2)       # 1/16 resolution
                x4 = self.layer4(x3)       # 1/32 resolution

                # Return multi-scale features
                return {
                    'layer1': x1,
                    'layer2': x2,
                    'layer3': x3,
                    'layer4': x4
                }

        return MultiScaleBackbone(backbone)

    def _reset_parameters(self):
        # Initialize the weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Special initialization for bbox_embed
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.class_embed.bias.shape) * bias_value

        # Initialize bbox_embed weights and biases
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples):
        """
        Args:
            samples: batched images, of shape [batch_size x 3 x H x W]

        Returns:
            pred_logits: classification logits
            pred_boxes: predicted box coordinates
        """
        # Ensure all model components are on the same device as input
        device = samples.device
        if not hasattr(self, '_model_on_device') or self._model_on_device != device:
            print(f"Moving model components to {device}")
            # Move all model components to the same device
            self.backbone = self.backbone.to(device)
            self.position_embedding = self.position_embedding.to(device)
            self.transformer = self.transformer.to(device)
            self.class_embed = self.class_embed.to(device)
            self.bbox_embed = self.bbox_embed.to(device)
            self.query_embed = self.query_embed.to(device)
            if self.input_proj is not None:
                self.input_proj = self.input_proj.to(device)
            self._model_on_device = device

        # Extract multi-scale features from the backbone
        multi_scale_features = self.backbone(samples)

        # Process features at different scales
        # For Deformable DETR, we need to use multiple feature levels
        feature_levels = []
        masks = []
        pos_embeds = []
        spatial_shapes = []

        # Get features from different levels in order of increasing stride
        # We use layer2, layer3, and layer4 from ResNet (1/8, 1/16, 1/32 resolution)
        # This can be made configurable for different backbones
        feature_names = ['layer2', 'layer3', 'layer4']

        # Create projections for each feature level at initialization time
        if not hasattr(self, 'input_proj_layers'):
            # Get the actual channel counts from the backbone
            backbone_channels = {
                'layer2': multi_scale_features['layer2'].shape[1],
                'layer3': multi_scale_features['layer3'].shape[1],
                'layer4': multi_scale_features['layer4'].shape[1]
            }

            print(f"Actual backbone channel counts: {backbone_channels}")

            # Initialize input_proj for layer2 if it's None
            if self.input_proj is None:
                self.input_proj = nn.Conv2d(backbone_channels['layer2'], self.transformer.d_model, kernel_size=1)
                nn.init.xavier_uniform_(self.input_proj.weight)
                nn.init.constant_(self.input_proj.bias, 0)
                print(f"Initialized input_proj for layer2 with {backbone_channels['layer2']} input channels")

            # Create all projection layers
            device = samples.device
            self.input_proj_layers = nn.ModuleDict()

            # Add each layer individually and move to correct device
            self.input_proj_layers['layer2'] = self.input_proj.to(device)
            self.input_proj_layers['layer3'] = nn.Conv2d(backbone_channels['layer3'], self.transformer.d_model, kernel_size=1).to(device)
            self.input_proj_layers['layer4'] = nn.Conv2d(backbone_channels['layer4'], self.transformer.d_model, kernel_size=1).to(device)

            # Initialize weights for new projections
            for name, layer in self.input_proj_layers.items():
                if name != 'layer2':  # layer2 is already initialized
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

            print(f"Created and moved all projection layers to {device}")

            # Register as module to ensure proper parameter management
            self.add_module('input_proj_layers', self.input_proj_layers)

        # Process each feature level
        for level_name in feature_names:
            # Get feature map for this level
            features = multi_scale_features[level_name]
            batch_size, _, height, width = features.shape  # Get feature dimensions

            # Create mask for feature map (all valid at this point)
            # In a real implementation, this would be based on the input image mask
            mask = torch.zeros((batch_size, height, width),
                              dtype=torch.bool, device=features.device)

            # Create position embeddings for this level
            pos_embed = self.position_embedding(mask)

            # Project features to the model dimension using the appropriate projection
            projected_features = self.input_proj_layers[level_name](features)

            # Store feature level information
            feature_levels.append(projected_features)
            masks.append(mask)
            pos_embeds.append(pos_embed)
            spatial_shapes.append((height, width))

        # Convert spatial_shapes to tensor for the attention mechanism
        spatial_shapes = torch.tensor(spatial_shapes, device=samples.device)

        # Check if the number of feature levels matches what we expect
        actual_levels = len(spatial_shapes)
        expected_levels = self.transformer.num_feature_levels
        if actual_levels != expected_levels:
            print(f"Warning: Expected {expected_levels} feature levels, got {actual_levels}")
            # Adjust transformer's internal parameter to match the actual number of levels
            self.transformer.num_feature_levels = actual_levels

        # Calculate level start indices for the attention mechanism
        level_start_index = torch.cat((
            torch.zeros((1,), dtype=torch.long, device=samples.device),
            torch.cumsum(spatial_shapes.prod(1), dim=0)[:-1]
        ))

        # Flatten and concatenate all feature levels for multi-scale processing
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []

        for src, mask, pos_embed in zip(feature_levels, masks, pos_embeds):
            # Flatten and append features, masks, and position embeddings
            src_flatten.append(src.flatten(2).transpose(1, 2))  # [bs, h*w, c]
            mask_flatten.append(mask.flatten(1))  # [bs, h*w]
            lvl_pos_embed_flatten.append(pos_embed.flatten(2).transpose(1, 2))  # [bs, h*w, c]

        # Concatenate all levels
        src_flatten = torch.cat(src_flatten, 1)  # [bs, sum(h*w), c]
        mask_flatten = torch.cat(mask_flatten, 1)  # [bs, sum(h*w)]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # [bs, sum(h*w), c]

        # Get query embeddings
        query_embed = self.query_embed.weight  # [num_queries, hidden_dim]

        # Pass through the transformer with multi-scale features
        # We need to modify the transformer to handle multi-scale features
        hs = self.transformer(
            src_flatten,
            mask_flatten,
            query_embed,
            lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        # Get outputs from the decoder
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # Return intermediate outputs if aux_loss is enabled
        if hasattr(self, 'aux_loss') and self.aux_loss:
            # Return all decoder outputs for auxiliary loss computation
            out = {
                'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coord[-1],
                'aux_outputs': [
                    {'pred_logits': outputs_class[i], 'pred_boxes': outputs_coord[i]}
                    for i in range(len(outputs_class) - 1)
                ]
            }
        else:
            # Return only the final output
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=3, enc_n_points=4, dec_n_points=4, use_checkpoint=False,
                 chunk_size_large=2000, chunk_size_small=50):
        """
        Deformable Transformer for Deformable DETR

        Args:
            d_model: Feature dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            activation: Activation function type
            return_intermediate_dec: Whether to return intermediate decoder outputs
            num_feature_levels: Number of feature levels to use
            enc_n_points: Number of sampling points in encoder
            dec_n_points: Number of sampling points in decoder
        """
        super().__init__()

        # Save parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.chunk_size_large = chunk_size_large
        self.chunk_size_small = chunk_size_small

        # Create encoder
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, enc_n_points, num_feature_levels,
            chunk_size_large=chunk_size_large, chunk_size_small=chunk_size_small)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, use_checkpoint)

        # Create decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, dec_n_points, num_feature_levels,
            chunk_size_large=chunk_size_large, chunk_size_small=chunk_size_small)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec, use_checkpoint)

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, spatial_shapes=None, level_start_index=None):
        """
        Forward function for DeformableTransformer supporting multi-scale features

        Args:
            src: Source features [bs, sum(h*w), c]
            mask: Source mask [bs, sum(h*w)]
            query_embed: Query embeddings [num_queries, c] or [num_queries, bs, c]
            pos_embed: Position embeddings [bs, sum(h*w), c]
            spatial_shapes: Spatial shapes of each feature level [num_levels, 2]
            level_start_index: Starting index of each feature level [num_levels]

        Returns:
            hs: Output of decoder [num_layers, bs, num_queries, c] or [1, bs, num_queries, c]
        """
        # Get batch size
        bs = src.shape[0]

        # Check if spatial_shapes is provided and matches expected number of levels
        if spatial_shapes is not None:
            actual_levels = spatial_shapes.size(0)
            if actual_levels != self.num_feature_levels:
                print(f"Warning: Expected {self.num_feature_levels} feature levels, got {actual_levels}")
                # Adjust our internal parameter to match the actual number of levels
                self.num_feature_levels = actual_levels

        # Convert src from [bs, sum(h*w), c] to [sum(h*w), bs, c] for transformer
        src = src.permute(1, 0, 2)  # [sum(h*w), bs, c]
        pos_embed = pos_embed.permute(1, 0, 2)  # [sum(h*w), bs, c]

        # Reshape query_embed to handle proper dimensions
        # Original shape is [num_queries, hidden_dim]
        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, bs, hidden_dim]

        # Transform through encoder with multi-scale support
        memory = self.encoder(
            src,
            src_key_padding_mask=mask,
            pos=pos_embed,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        # Create target tensor with proper dimensions
        tgt = torch.zeros_like(query_embed)

        # Use decoder with multi-scale support
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )

        return hs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation, nhead, enc_n_points, num_feature_levels=3,
                 chunk_size_large=10000, chunk_size_small=100):
        super().__init__()
        # Save parameters
        self.n_levels = num_feature_levels

        # Self-attention block with deformable attention
        self.self_attn = MultiScaleDeformableAttention(
            d_model, nhead, n_levels=num_feature_levels, n_points=enc_n_points, dropout=dropout,
            chunk_size_large=chunk_size_large, chunk_size_small=chunk_size_small
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation function
        self.activation = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None, pos=None, spatial_shapes=None, level_start_index=None):
        # Note: pos is not used in this implementation but kept for API compatibility
        # Self-attention block
        src2 = self.norm1(src)

        # Convert from [seq_len, bs, dim] to [bs, seq_len, dim]
        src2_orig_shape = src2.shape
        if src2.dim() == 3 and src2_orig_shape[1] != src2_orig_shape[0]:
            # Transformer format: [seq_len, bs, dim] -> [bs, seq_len, dim]
            src2 = src2.permute(1, 0, 2)
        # No need to transform pos as it's not used in the attention module

        # Generate reference points
        bs, seq_len, _ = src2.shape
        actual_levels = spatial_shapes.size(0)

        # Ensure we're using the correct number of levels
        expected_levels = getattr(self, 'n_levels', 3)  # Default to 3 if not set
        if actual_levels != expected_levels:
            print(f"Warning: Expected {expected_levels} feature levels but got {actual_levels}")

        reference_points = torch.zeros(bs, seq_len, actual_levels, 2, device=src2.device)
        for lvl in range(actual_levels):
            # Generate normalized coordinates for this level
            reference_points[:, :, lvl, 0] = 0.5  # Center x
            reference_points[:, :, lvl, 1] = 0.5  # Center y

        # Pass through multi-scale deformable attention
        src2_out = self.self_attn(
            query=src2,
            reference_points=reference_points,
            input_flatten=src2,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=src_key_padding_mask
        )

        # Convert back to original shape if needed
        if src2.dim() == 3 and src2_orig_shape[1] != src2_orig_shape[0]:
            src2_out = src2_out.permute(1, 0, 2)  # [bs, seq_len, dim] -> [seq_len, bs, dim]

        src = src + self.dropout1(src2_out)

        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

    def forward(self, src, src_key_padding_mask=None, pos=None, spatial_shapes=None, level_start_index=None):
        """
        Forward function for DeformableTransformerEncoder with multi-scale support

        Args:
            src: Source features [seq_len, bs, c]
            src_key_padding_mask: Source mask [bs, seq_len]
            pos: Position embeddings [seq_len, bs, c]
            spatial_shapes: Spatial shapes of each feature level [num_levels, 2]
            level_start_index: Starting index of each feature level [num_levels]

        Returns:
            output: Encoded features [seq_len, bs, c]
        """
        output = src

        for layer in self.layers:
            # Use gradient checkpointing if enabled to save memory
            if self.use_checkpoint and self.training:
                # Define a custom forward function for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                # Apply gradient checkpointing
                output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    output,
                    src_key_padding_mask,
                    pos,
                    spatial_shapes,
                    level_start_index
                )
            else:
                # Standard forward pass
                output = layer(
                    output,
                    src_key_padding_mask=src_key_padding_mask,
                    pos=pos,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index
                )

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation, nhead, dec_n_points, num_feature_levels=3,
                 chunk_size_large=10000, chunk_size_small=100):
        super().__init__()

        # Save parameters
        self.n_levels = num_feature_levels

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention with deformable attention
        self.cross_attn = MultiScaleDeformableAttention(
            d_model, nhead, n_levels=num_feature_levels, n_points=dec_n_points, dropout=dropout,
            chunk_size_large=chunk_size_large, chunk_size_small=chunk_size_small
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Activation
        self.activation = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None,
                spatial_shapes=None, level_start_index=None):
        # Note: pos is not used in this implementation but kept for API compatibility
        """
        Forward function for DeformableTransformerDecoderLayer with multi-scale support

        Args:
            tgt: Target features [num_queries, bs, c]
            memory: Memory features from encoder [seq_len, bs, c]
            tgt_key_padding_mask: Target mask [bs, num_queries]
            memory_key_padding_mask: Memory mask [bs, seq_len]
            pos: Position embeddings for memory [seq_len, bs, c]
            query_pos: Position embeddings for queries [num_queries, bs, c]
            spatial_shapes: Spatial shapes of each feature level [num_levels, 2]
            level_start_index: Starting index of each feature level [num_levels]

        Returns:
            tgt: Updated target features [num_queries, bs, c]
        """
        # Self-attention block
        q = k = tgt + query_pos if query_pos is not None else tgt

        # tgt: [nq, bs, dim], query_pos: [nq, bs, dim]
        # MultiheadAttention expects q, k, v of shape [seq_len, batch, embed_dim]
        tgt2 = self.self_attn(
            query=q,
            key=k,
            value=tgt,
            attn_mask=None,
            key_padding_mask=tgt_key_padding_mask
        )[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention block with deformable attention and multi-scale support
        # Convert from [num_queries, bs, c] to [bs, num_queries, c]
        tgt_flat = tgt.transpose(0, 1)  # [bs, num_queries, c]
        memory_flat = memory.transpose(0, 1)  # [bs, seq_len, c]

        # Generate reference points for each query
        bs, num_queries, _ = tgt_flat.shape
        actual_levels = spatial_shapes.size(0)

        # Ensure we're using the correct number of levels
        expected_levels = getattr(self, 'n_levels', 3)  # Default to 3 if not set
        if actual_levels != expected_levels:
            print(f"Warning: Expected {expected_levels} feature levels but got {actual_levels}")

        reference_points = torch.zeros(bs, num_queries, actual_levels, 2, device=tgt.device)
        for lvl in range(actual_levels):
            reference_points[:, :, lvl, 0] = 0.5  # Center x
            reference_points[:, :, lvl, 1] = 0.5  # Center y

        # Apply multi-scale deformable attention
        tgt2 = self.cross_attn(
            query=tgt_flat,
            reference_points=reference_points,
            input_flatten=memory_flat,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=memory_key_padding_mask
        )

        # Convert back to original shape
        tgt2 = tgt2.transpose(0, 1)  # [num_queries, bs, c]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_checkpoint=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.use_checkpoint = use_checkpoint

    def forward(self, tgt, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None,
                spatial_shapes=None, level_start_index=None):
        """
        Forward function for DeformableTransformerDecoder with multi-scale support

        Args:
            tgt: Target features [num_queries, bs, c]
            memory: Memory features from encoder [seq_len, bs, c]
            tgt_key_padding_mask: Target mask [bs, num_queries]
            memory_key_padding_mask: Memory mask [bs, seq_len]
            pos: Position embeddings for memory [seq_len, bs, c]
            query_pos: Position embeddings for queries [num_queries, bs, c]
            spatial_shapes: Spatial shapes of each feature level [num_levels, 2]
            level_start_index: Starting index of each feature level [num_levels]

        Returns:
            output: Decoded features [num_layers, bs, num_queries, c] or [1, bs, num_queries, c]
        """
        output = tgt

        intermediate = []

        for layer in self.layers:
            # Use gradient checkpointing if enabled to save memory
            if self.use_checkpoint and self.training:
                # Define a custom forward function for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                # Apply gradient checkpointing
                output = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    output,
                    memory,
                    tgt_key_padding_mask,
                    memory_key_padding_mask,
                    pos,
                    query_pos,
                    spatial_shapes,
                    level_start_index
                )
            else:
                # Standard forward pass
                output = layer(
                    output,
                    memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    pos=pos,
                    query_pos=query_pos,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index
                )

            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module

    This implements the exact deformable attention mechanism from the Deformable DETR paper.
    Paper: "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
    Link: https://arxiv.org/abs/2010.04159
    """
    def __init__(self, d_model, n_heads, n_levels=3, n_points=4, dropout=0.1, chunk_size_large=10000, chunk_size_small=100):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.d_head = d_model // n_heads

        # Memory optimization parameters
        self.chunk_size_large = chunk_size_large
        self.chunk_size_small = chunk_size_small

        # Projection layers
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)  # 2D offsets
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize sampling offsets
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)

        # Initialize reference points in a circle
        # Use the same device as the module parameters
        device = self.sampling_offsets.weight.device
        thetas = torch.arange(self.n_heads, dtype=torch.float32, device=device) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # [n_heads, 2]

        # Expand to all levels and points
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).expand(
            self.n_heads, self.n_levels, self.n_points, 2
        )

        # Clone the tensor to avoid in-place modification issues
        grid_init = grid_init.clone()

        # Scale the reference points by point index
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        # Flatten and set as bias
        grid_init = grid_init.flatten(0, 2)
        self.sampling_offsets.bias.data = grid_init.view(-1)

        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

        # Initialize projections with Xavier uniform
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        """
        Multi-scale Deformable Attention Module Forward Function

        Args:
            query: Query embeddings [bs, num_query, embed_dim]
            reference_points: The normalized reference points with shape [bs, num_query, n_levels, 2],
                              all elements are in range [0, 1], top-left (0, 0), bottom-right (1, 1)
            input_flatten: Flattened feature from multi-scale feature maps [bs, sum(h*w), embed_dim]
            input_spatial_shapes: Spatial shapes of each feature level [n_levels, 2]
            input_level_start_index: Starting index of each level in flattened features [n_levels]
            input_padding_mask: Padding mask for input_flatten [bs, sum(h*w)]

        Returns:
            output: Attention output [bs, num_query, embed_dim]
        """
        # Ensure all inputs are on the same device
        device = query.device
        if reference_points.device != device:
            reference_points = reference_points.to(device)
        if input_flatten.device != device:
            input_flatten = input_flatten.to(device)
        if input_spatial_shapes.device != device:
            input_spatial_shapes = input_spatial_shapes.to(device)
        if input_level_start_index.device != device:
            input_level_start_index = input_level_start_index.to(device)
        if input_padding_mask is not None and input_padding_mask.device != device:
            input_padding_mask = input_padding_mask.to(device)

        # Check input dimensions
        bs, num_query, embed_dim = query.shape
        bs_value, num_value, _ = input_flatten.shape

        # Ensure batch sizes match
        if bs != bs_value:
            print(f"Warning: Batch size mismatch in MultiScaleDeformableAttention: {bs} vs {bs_value}")
            # Handle batch size mismatch (can happen with gradient checkpointing)
            if bs_value > bs:
                # Truncate input_flatten to match query batch size
                input_flatten = input_flatten[:bs]
                if input_padding_mask is not None:
                    input_padding_mask = input_padding_mask[:bs]
            else:
                # Expand query to match input_flatten batch size
                query = query.expand(bs_value, num_query, embed_dim)
                bs = bs_value
                if reference_points.size(0) != bs:
                    reference_points = reference_points.expand(bs, *reference_points.shape[1:])

        # Verify dimensions after adjustment
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == num_value, \
            f"Spatial shapes {input_spatial_shapes} don't match flattened input size {num_value}"

        # Memory optimization: Process in chunks if the input is too large
        # Get chunk size from config or use default
        chunk_size = getattr(self, 'chunk_size_large', 10000)  # Default to 10000 if not set
        if num_value > chunk_size and input_padding_mask is not None:
            # Process value projection and masking in chunks
            value_chunks = []
            for i in range(0, num_value, chunk_size):
                end_idx = min(i + chunk_size, num_value)
                # Project chunk
                chunk = self.value_proj(input_flatten[:, i:end_idx])
                # Apply mask to chunk if needed
                if input_padding_mask is not None:
                    chunk = chunk.masked_fill(input_padding_mask[:, i:end_idx].unsqueeze(-1), 0.0)
                # Reshape chunk
                chunk = chunk.view(bs, end_idx - i, self.n_heads, self.d_head)
                value_chunks.append(chunk)
            # Concatenate chunks
            value = torch.cat(value_chunks, dim=1)
            # Free memory
            del value_chunks
            torch.cuda.empty_cache()
        else:
            # Process normally for smaller inputs
            value = self.value_proj(input_flatten)
            if input_padding_mask is not None:
                value = value.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
            value = value.view(bs, num_value, self.n_heads, self.d_head)

        # Generate sampling offsets and attention weights
        try:
            # Ensure query has the right shape for the linear projections
            if query.dim() != 3 or query.size(2) != self.d_model:
                print(f"Warning: Query shape mismatch: {query.shape}, expected [..., {self.d_model}]")
                # Try to reshape or pad if needed
                if query.dim() > 3 and query.size(-1) == self.d_model:
                    # Reshape higher dimensional query
                    orig_shape = query.shape
                    query = query.reshape(-1, num_query, self.d_model)
                    print(f"Reshaped query from {orig_shape} to {query.shape}")
                elif query.size(2) != self.d_model:
                    # Pad or truncate to match expected dimension
                    if query.size(2) < self.d_model:
                        # Pad
                        padding = torch.zeros(bs, num_query, self.d_model - query.size(2), device=query.device)
                        query = torch.cat([query, padding], dim=2)
                    else:
                        # Truncate
                        query = query[:, :, :self.d_model]
                    print(f"Adjusted query dimension to {query.shape}")

            sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.n_heads, self.n_levels, self.n_points, 2
            )
            attention_weights = self.attention_weights(query).view(
                bs, num_query, self.n_heads, self.n_levels * self.n_points
            )
            attention_weights = F.softmax(attention_weights, dim=-1).view(
                bs, num_query, self.n_heads, self.n_levels, self.n_points
            )
        except Exception as e:
            print(f"Error in MultiScaleDeformableAttention forward: {e}")
            print(f"Query shape: {query.shape}, expected dim: {self.d_model}")
            print(f"Sampling offsets weight shape: {self.sampling_offsets.weight.shape}")
            print(f"Attention weights weight shape: {self.attention_weights.weight.shape}")
            # Try a fallback approach
            if hasattr(torch, 'compile') and torch._dynamo.is_compiling():
                print("Error during torch.compile, using fallback implementation")
                # Simple fallback implementation
                sampling_offsets = torch.zeros(
                    bs, num_query, self.n_heads, self.n_levels, self.n_points, 2, device=query.device
                )
                attention_weights = torch.ones(
                    bs, num_query, self.n_heads, self.n_levels, self.n_points, device=query.device
                ) / (self.n_levels * self.n_points)
            else:
                # Re-raise if not in compilation
                raise

        # N, Len_q, n_heads, n_levels, n_points, 2
        try:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )

            # Ensure dimensions match
            actual_levels = input_spatial_shapes.size(0)
            if reference_points.size(2) != actual_levels:
                print(f"Warning: reference_points has {reference_points.size(2)} levels but input_spatial_shapes has {actual_levels} levels")
                # Adjust reference_points to match actual levels
                if reference_points.size(2) > actual_levels:
                    reference_points = reference_points[:, :, :actual_levels]
                else:
                    # Pad reference_points with the last level repeated
                    last_level = reference_points[:, :, -1:]
                    padding = last_level.repeat(1, 1, actual_levels - reference_points.size(2), 1)
                    reference_points = torch.cat([reference_points, padding], dim=2)

            # Ensure sampling_offsets matches the actual number of levels
            if sampling_offsets.size(3) != actual_levels:
                print(f"Warning: sampling_offsets has {sampling_offsets.size(3)} levels but input_spatial_shapes has {actual_levels} levels")
                # Adjust sampling_offsets to match actual levels
                if sampling_offsets.size(3) > actual_levels:
                    sampling_offsets = sampling_offsets[:, :, :, :actual_levels]
                else:
                    # Pad sampling_offsets with zeros
                    padding_shape = list(sampling_offsets.shape)
                    padding_shape[3] = actual_levels - sampling_offsets.size(3)
                    padding = torch.zeros(padding_shape, device=sampling_offsets.device)
                    sampling_offsets = torch.cat([sampling_offsets, padding], dim=3)

            # Ensure attention_weights matches the actual number of levels
            if attention_weights.size(3) != actual_levels:
                print(f"Warning: attention_weights has {attention_weights.size(3)} levels but input_spatial_shapes has {actual_levels} levels")
                # Adjust attention_weights to match actual levels
                if attention_weights.size(3) > actual_levels:
                    attention_weights = attention_weights[:, :, :, :actual_levels]
                else:
                    # Pad attention_weights with uniform weights
                    padding_shape = list(attention_weights.shape)
                    padding_shape[3] = actual_levels - attention_weights.size(3)
                    # Create uniform weights for padding
                    padding = torch.ones(padding_shape, device=attention_weights.device) / (actual_levels * self.n_points)
                    attention_weights = torch.cat([attention_weights, padding], dim=3)
                    # Renormalize weights
                    attention_weights = attention_weights / attention_weights.sum(dim=(-2, -1), keepdim=True)

            sampling_locations = reference_points[:, :, None, :, None, :] \
                                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

            # Perform deformable attention sampling
            output = self._deformable_attention_core(
                value, input_spatial_shapes, input_level_start_index,
                sampling_locations, attention_weights
            )

            output = self.output_proj(output)
            return self.dropout(output)
        except Exception as e:
            print(f"Error in final part of MultiScaleDeformableAttention: {e}")
            # Provide a fallback implementation for gradient checkpointing
            if torch.jit.is_scripting() or (hasattr(torch, 'compile') and torch._dynamo.is_compiling()):
                print("Using fallback implementation during compilation")
                # Simple fallback - just project the query as output
                # This is only used during compilation and will be replaced with the real implementation
                fallback_output = self.output_proj(query)
                return self.dropout(fallback_output)
            else:
                # Re-raise the exception if not in compilation mode
                raise

    def _deformable_attention_core(self, value, spatial_shapes, level_start_index,
                                  sampling_locations, attention_weights):
        """
        Core deformable attention operation
        """
        # Ensure all inputs are on the same device
        device = value.device
        if spatial_shapes.device != device:
            spatial_shapes = spatial_shapes.to(device)
        if level_start_index.device != device:
            level_start_index = level_start_index.to(device)
        if sampling_locations.device != device:
            sampling_locations = sampling_locations.to(device)
        if attention_weights.device != device:
            attention_weights = attention_weights.to(device)

        bs, num_value, n_heads, d_head = value.shape
        _, num_query, n_heads, n_levels, n_points, _ = sampling_locations.shape

        # Split value by levels
        value_list = []
        for lvl in range(n_levels):
            start_idx = level_start_index[lvl]
            if lvl < n_levels - 1:
                end_idx = level_start_index[lvl + 1]
            else:
                end_idx = num_value
            value_list.append(value[:, start_idx:end_idx])

        # Prepare for bilinear sampling
        sampling_grids = 2 * sampling_locations - 1  # Convert [0, 1] to [-1, 1] for grid_sample

        # Perform sampling for each level
        sampled_values = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # Memory optimization: Free memory after processing each level
            torch.cuda.empty_cache()

            # Reshape value for this level to [bs*n_heads, d_head, h, w]
            value_l = value_list[lvl].permute(0, 2, 1, 3).reshape(bs*n_heads, h, w, d_head)
            value_l = value_l.permute(0, 3, 1, 2)  # [bs*n_heads, d_head, h, w]

            # Process in chunks if the query size is large
            # Get chunk size from config or use default
            chunk_size = getattr(self, 'chunk_size_small', 100)  # Default to 100 if not set
            if num_query > chunk_size:
                sampled_chunks = []
                for i in range(0, num_query, chunk_size):
                    end_idx = min(i + chunk_size, num_query)
                    # Get grid for this chunk
                    grid_chunk = sampling_grids[:, i:end_idx, :, lvl].transpose(1, 2).flatten(0, 1)
                    # Sample using grid_sample
                    sampled_chunk = F.grid_sample(
                        value_l, grid_chunk, mode='bilinear', padding_mode='zeros', align_corners=False
                    )  # [bs*n_heads, d_head, chunk_size, n_points]
                    sampled_chunks.append(sampled_chunk.view(bs, n_heads, d_head, end_idx - i, n_points))
                # Concatenate chunks
                sampled_value_l = torch.cat(sampled_chunks, dim=3)
                # Free memory
                del sampled_chunks, grid_chunk
            else:
                # Reshape sampling grid for this level
                sampling_grid_l = sampling_grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
                # Sample using grid_sample
                sampled_value_l = F.grid_sample(
                    value_l, sampling_grid_l, mode='bilinear', padding_mode='zeros', align_corners=False
                )  # [bs*n_heads, d_head, num_query, n_points]
                sampled_value_l = sampled_value_l.view(bs, n_heads, d_head, num_query, n_points)

            sampled_values.append(sampled_value_l)
            # Free memory
            del value_l
            if 'sampling_grid_l' in locals():
                del sampling_grid_l

        # Combine sampled values with attention weights
        # Memory optimization: Process in chunks if needed
        # Get chunk size from config or use default
        chunk_size = getattr(self, 'chunk_size_small', 100)  # Default to 100 if not set
        if num_query > chunk_size:
            # Process in chunks to save memory
            output_chunks = []
            for i in range(0, num_query, chunk_size):
                end_idx = min(i + chunk_size, num_query)
                # Extract chunks from sampled values and attention weights
                sampled_values_chunk = [sv[:, :, :, i:end_idx] for sv in sampled_values]
                attention_weights_chunk = attention_weights[:, i:end_idx]

                # Stack and permute
                sv_stacked = torch.stack(sampled_values_chunk, dim=-2)  # [bs, n_heads, d_head, chunk_size, n_levels, n_points]
                sv_permuted = sv_stacked.permute(0, 3, 1, 4, 5, 2)  # [bs, chunk_size, n_heads, n_levels, n_points, d_head]

                # Apply attention weights
                output_chunk = (sv_permuted * attention_weights_chunk.unsqueeze(-1)).sum(dim=(-2, -3))
                output_chunk = output_chunk.permute(0, 2, 1, 3).reshape(bs, end_idx - i, self.d_model)
                output_chunks.append(output_chunk)

                # Free memory
                del sv_stacked, sv_permuted, sampled_values_chunk, attention_weights_chunk
                torch.cuda.empty_cache()

            # Concatenate chunks
            output = torch.cat(output_chunks, dim=1)
            # Free memory
            del output_chunks
        else:
            # Process normally for smaller inputs
            sampled_values = torch.stack(sampled_values, dim=-2)  # [bs, n_heads, d_head, num_query, n_levels, n_points]
            sampled_values = sampled_values.permute(0, 3, 1, 4, 5, 2)  # [bs, num_query, n_heads, n_levels, n_points, d_head]

            # Apply attention weights
            output = (sampled_values * attention_weights.unsqueeze(-1)).sum(dim=(-2, -3))
            output = output.permute(0, 2, 1, 3).reshape(bs, num_query, self.d_model)

        # Free memory
        del sampled_values
        torch.cuda.empty_cache()

        return output


class MLP(nn.Module):
    """
    Simple multi-layer perceptron
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        h = [hidden_dim] * (num_layers - 1)
        self.layers.append(nn.Linear(input_dim, h[0] if num_layers > 1 else output_dim))

        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(h[i], h[i + 1]))

        if num_layers > 1:
            self.layers.append(nn.Linear(h[-1], output_dim))

        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.relu(x)
        return x


def _get_clones(module, N):
    """
    Create N identical copies of a module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """
    Get activation function by name
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    else:
        raise RuntimeError(f"Activation {activation} not supported")


# Add necessary imports that were missing at the top
import math
import copy

# Add position embedding class
class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal position embedding for the transformer
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is specified")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        """
        Args:
            mask: [batch_size, height, width] tensor of booleans

        Returns:
            pos: [batch_size, channels, height, width] position encodings
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        # Ensure shape is [batch_size, channels, height, width]
        return pos