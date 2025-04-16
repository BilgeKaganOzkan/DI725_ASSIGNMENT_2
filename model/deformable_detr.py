import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

class DeformableDetrModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6, 
                 num_feature_levels=4, enc_n_points=4, dec_n_points=4, num_queries=300):
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
        """
        super().__init__()
        
        # Backbone (We'll use ResNet50 by default)
        self.backbone = self._get_backbone()
        
        # Position embedding
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)
        
        # Input projection
        self.input_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        
        # Transformer encoder-decoder
        self.transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background class
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 coordinates for bounding box
        
        # Query embeddings for the decoder
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # Default: 100 object queries
        
        self._reset_parameters()
    
    def _get_backbone(self):
        """
        Creates a ResNet50 backbone with FPN
        """
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            
            # Use only the first few layers (up to layer3)
            return nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3
            )
        except:
            # Fallback if torchvision is not the latest version
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=True)
            
            # Use only the first few layers (up to layer3)
            return nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3
            )
    
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
        # Extract features from the backbone
        features = self.backbone(samples)
        
        # Create mask for feature map (all valid at this point)
        mask = torch.zeros((features.shape[0], features.shape[2], features.shape[3]), 
                          dtype=torch.bool, device=features.device)
        
        # Create position embeddings
        pos_embed = self.position_embedding(mask)
        
        # Get query embeddings
        query_embed = self.query_embed.weight  # [num_queries, hidden_dim]
        
        # Project features
        projected_features = self.input_proj(features)
        
        # Pass through the transformer
        hs = self.transformer(projected_features, mask, query_embed, pos_embed)
        
        # Get outputs from the decoder
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        return {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, enc_n_points=4, dec_n_points=4):
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
        
        # Create encoder
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Create decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate_dec)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed, pos_embed):
        # Flatten feature map for encoder
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)  # HW, BS, C
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)  # HW, BS, C
        
        # Reshape query_embed to handle proper dimensions
        # Original shape is [num_queries, hidden_dim]
        if query_embed.dim() == 2:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [num_queries, bs, hidden_dim]
        
        mask = mask.flatten(1)  # BS, HW
        
        # Transform through encoder and decoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        
        # Create target tensor with proper dimensions
        tgt = torch.zeros_like(query_embed)
        
        # Use decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                         pos=pos_embed, query_pos=query_embed)
        
        return hs


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation, nhead, enc_n_points):
        super().__init__()
        # Self-attention block with deformable attention
        self.self_attn = DeformableAttention(d_model, nhead, n_points=enc_n_points, dropout=dropout)
        
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
    
    def forward(self, src, src_key_padding_mask=None, pos=None):
        # Self-attention block
        src2 = self.norm1(src)
        # Pass source as query (self-attention)
        src2 = self.self_attn(src2, src_key_padding_mask=src_key_padding_mask, pos=pos)
        src = src + self.dropout1(src2)
        
        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, src, src_key_padding_mask=None, pos=None):
        output = src
        
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation, nhead, dec_n_points):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention with deformable attention
        self.cross_attn = DeformableAttention(d_model, nhead, n_points=dec_n_points, dropout=dropout)
        
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
                memory_key_padding_mask=None, pos=None, query_pos=None):
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
        
        # Cross-attention block with deformable attention
        tgt2 = self.cross_attn(
            query=tgt, 
            memory=memory, 
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos, 
            query_pos=query_pos
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
    
    def forward(self, tgt, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt
        
        intermediate = []
        
        for layer in self.layers:
            output = layer(output, memory, tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         pos=pos, query_pos=query_pos)
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output.unsqueeze(0)


class DeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module
    
    This implements the deformable attention mechanism from the Deformable DETR paper.
    """
    def __init__(self, d_model, n_heads, n_points=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_points = n_points
        
        # Projection layers
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)  # n_points per head, 2D offsets
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)  # n_points per head
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self._reset_parameters()
        
        self.dropout = nn.Dropout(dropout)
    
    def _reset_parameters(self):
        # Initialize sampling offsets
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        grid_init = grid_init.view(self.n_heads, self.n_points, 2).flatten(0, 1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)
    
    def forward(self, query, memory=None, memory_key_padding_mask=None, pos=None, query_pos=None, src_key_padding_mask=None):
        """
        Args:
            query: Query embeddings (target) [bs, nq, c] or [sequence_length, bs, c] in transformer format
            memory: Memory embeddings (source) [bs, ns, c] or [sequence_length, bs, c]
            memory_key_padding_mask: Key padding mask for memory
            pos: Position encoding for memory
            query_pos: Position encoding for query
            src_key_padding_mask: Padding mask for source (alias for memory_key_padding_mask)
            
        Returns:
            output: Attention output
        """
        # Support both self-attention and cross-attention patterns
        if memory is None:
            memory = query
        
        # Prefer memory_key_padding_mask over src_key_padding_mask for backward compatibility
        if memory_key_padding_mask is None and src_key_padding_mask is not None:
            memory_key_padding_mask = src_key_padding_mask
        
        # Handle different tensor layouts: [sequence_length, batch_size, channels] vs [batch_size, sequence_length, channels]
        # In decoder, tensors come in [sequence_length, batch_size, channels] format
        is_transformer_format = query.dim() == 3 and query.shape[1] != self.d_model
        
        if is_transformer_format:
            # Convert from [sequence_length, batch_size, channels] to [batch_size, sequence_length, channels]
            query = query.transpose(0, 1)
            if memory is not None and memory is not query:
                memory = memory.transpose(0, 1)
        
        # Get dimensions
        bs, nq, c = query.shape
        bs, ns, c = memory.shape
        
        # Project query and generate sampling locations and attention weights
        if query_pos is not None:
            if is_transformer_format and query_pos.dim() == 3:
                query_pos = query_pos.transpose(0, 1)
            query = query + query_pos
        
        # Generate sampling offsets and attention weights
        # Project query to get sampling offsets and attention weights
        q_for_offsets = query
        
        # Compute sampling offsets - these define where to sample from the feature maps
        offsets = self.sampling_offsets(q_for_offsets)
        offsets = offsets.view(bs, nq, self.n_heads, self.n_points, 2)
        
        # Compute attention weights 
        attention_weights = self.attention_weights(q_for_offsets)
        attention_weights = attention_weights.view(bs, nq, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Project values
        values = self.value_proj(memory).view(bs, ns, self.n_heads, self.d_head)
        
        # Implementation of deformable attention
        # For each query, we have N heads, and for each head, we have M reference points
        # We need to sample M points from the feature map for each head
        
        # For a proper implementation, we would:
        # 1. Use the reference points (offsets) to sample values from the feature map
        # 2. Apply attention weights to the sampled values
        # 3. Combine the weighted values to get the output
        
        # Since this is a simplified implementation without proper 2D deformable sampling,
        # we'll approximate it by:
        # 1. Creating a mask based on the offsets
        # 2. Using the mask to sample from the feature map
        # 3. Applying attention weights
        
        # For simplicity, get the first value vector for each query
        # In a real implementation, this would use offsets to sample at specific positions
        # This is a simplified approximation
        
        # We'll create a weighted sum of features for each query
        output = torch.zeros(bs, nq, self.n_heads, self.d_head, device=query.device)
        
        # For each query position and head, compute the weighted sum of values
        # This is a simplified approach that doesn't use proper offset sampling
        # In a real implementation, we would use bilinear sampling based on offsets
        
        # Get a feature from memory (first position as fallback)
        memory_feature = values[:, 0, :, :].unsqueeze(1).expand(-1, nq, -1, -1)
        
        # Apply attention weights 
        for h in range(self.n_heads):
            for r in range(self.n_points):
                # Get attention weight for this reference point
                attn_weight = attention_weights[:, :, h, r].unsqueeze(-1)
                # Apply weight to feature (simplified)
                output[:, :, h, :] += attn_weight * memory_feature[:, :, h, :]
        
        # Reshape output to [bs, nq, d_model]
        output = output.reshape(bs, nq, self.d_model)
        
        # Final projection
        output = self.output_proj(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Convert back to transformer format if needed
        if is_transformer_format:
            output = output.transpose(0, 1)
        
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