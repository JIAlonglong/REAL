"""
Cross-Attention Encoder for scan_dot OR depth camera observations
Based on the original Parkour work architecture.
This module can encode either scan_dot or depth_latent using cross-attention mechanism.
"""
import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size = x.size(0)
        
        # Linear projections and split into heads
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    def __init__(self, d_model, max_len=50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x: [batch_size, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class CrossAttentionEncoder(nn.Module):
    """
    Cross-Attention Encoder that can encode either scan_dot OR depth_latent.
    
    This encoder uses self-attention mechanism to process the input features,
    enabling rich feature interactions. It can handle both scan_dot and depth_latent
    as inputs (but not simultaneously - they are mutually exclusive).
    
    Support for frame attention: process multiple history frames with attention.
    """
    def __init__(
        self,
        scan_input_dim=132,  # scan_dot dimension
        depth_latent_dim=32,  # depth encoder output dimension
        d_model=256,  # model dimension for attention
        num_heads=8,
        num_layers=2,  # number of attention layers
        scan_proj_dims=[256, 256],  # MLP for scan_dot projection
        depth_proj_dim=256,  # projection dimension for depth features
        output_dim=256,  # final output dimension (should match scan_encoder_output_dim)
        activation=nn.ELU(),
        dropout=0.1,
        frame_attention=False,  # whether to use frame attention
        num_history_frames=2,  # number of history frames (including current)
        use_positional_encoding=True,  # whether to use positional encoding
        use_layer_norm=True,  # whether to use layer normalization
    ):
        super().__init__()
        
        self.d_model = d_model
        self.scan_input_dim = scan_input_dim
        self.depth_latent_dim = depth_latent_dim
        self.frame_attention = frame_attention
        self.num_history_frames = num_history_frames
        self.use_positional_encoding = use_positional_encoding
        self.use_layer_norm = use_layer_norm
        
        # Project scan_dot to d_model
        scan_proj_layers = []
        scan_proj_layers.append(nn.Linear(scan_input_dim, scan_proj_dims[0]))
        scan_proj_layers.append(activation)
        for i in range(len(scan_proj_dims) - 1):
            scan_proj_layers.append(nn.Linear(scan_proj_dims[i], scan_proj_dims[i + 1]))
            scan_proj_layers.append(activation)
        # Final projection to d_model
        scan_proj_layers.append(nn.Linear(scan_proj_dims[-1], d_model))
        self.scan_proj = nn.Sequential(*scan_proj_layers)
        
        # Project depth latent to d_model
        self.depth_proj = nn.Sequential(
            nn.Linear(depth_latent_dim, depth_proj_dim),
            activation,
            nn.Linear(depth_proj_dim, d_model)
        )
        
        # Positional encoding for frame attention
        if self.use_positional_encoding and self.frame_attention:
            self.positional_encoding = PositionalEncoding(d_model, max_len=self.num_history_frames)
        
        # Self-attention layers (can be used for both scan and depth, and for frame attention)
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        ff_dim = d_model * 2
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_dim),
                activation,
                nn.Dropout(dropout),
                nn.Linear(ff_dim, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Output projection + residual/skips for stability
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            activation,
            nn.Linear(output_dim, output_dim),
            nn.Tanh()  # Match original scan_encoder output activation
        )
        self.output_residual = nn.Linear(d_model, output_dim)
        self.output_ln = nn.LayerNorm(d_model)
        
    def forward(self, scan_dot=None, depth_latent=None):
        """
        Args:
            scan_dot: [batch_size, seq_len, scan_input_dim] or [batch_size, scan_input_dim]
                     - scan_dot observations (optional)
            depth_latent: [batch_size, seq_len, depth_latent_dim] or [batch_size, depth_latent_dim]
                          - depth encoder output (optional)
        Returns:
            output: [batch_size, output_dim] - encoded features
        Note: Either scan_dot or depth_latent should be provided, but not both.
        """
        if scan_dot is not None and depth_latent is not None:
            raise ValueError("Cannot provide both scan_dot and depth_latent. Use one or the other.")
        if scan_dot is None and depth_latent is None:
            raise ValueError("Must provide either scan_dot or depth_latent.")
        
        # Process scan_dot input
        if scan_dot is not None:
            batch_size = scan_dot.size(0)
            
            # Check if we have frame history
            if scan_dot.dim() == 2:  # [batch_size, scan_input_dim] - no history
                # Project to d_model
                features = self.scan_proj(scan_dot).unsqueeze(1)  # [batch_size, 1, d_model]
            else:  # [batch_size, seq_len, scan_input_dim] - with history
                seq_len = scan_dot.size(1)
                # Reshape to process all frames
                # scan_dot may be non-contiguous (e.g. created via view/permute upstream); use reshape for safety
                scan_dot_flat = scan_dot.reshape(-1, self.scan_input_dim)  # [batch_size*seq_len, scan_input_dim]
                features_flat = self.scan_proj(scan_dot_flat)  # [batch_size*seq_len, d_model]
                features = features_flat.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        # Process depth_latent input
        else:  # depth_latent is not None
            batch_size = depth_latent.size(0)
            
            # Check if we have frame history
            if depth_latent.dim() == 2:  # [batch_size, depth_latent_dim] - no history
                # Project to d_model
                features = self.depth_proj(depth_latent).unsqueeze(1)  # [batch_size, 1, d_model]
            else:  # [batch_size, seq_len, depth_latent_dim] - with history
                seq_len = depth_latent.size(1)
                # Reshape to process all frames
                # depth_latent may be non-contiguous; use reshape for safety
                depth_latent_flat = depth_latent.reshape(-1, self.depth_latent_dim)  # [batch_size*seq_len, depth_latent_dim]
                features_flat = self.depth_proj(depth_latent_flat)  # [batch_size*seq_len, d_model]
                features = features_flat.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding if enabled and we have frame history
        if self.use_positional_encoding and self.frame_attention and features.size(1) > 1:
            features = self.positional_encoding(features)
        
        # Apply self-attention layers
        x = features
        
        for i, attn_layer in enumerate(self.attention_layers):
            # Self-attention
            attn_output = attn_layer(x)
            x = self.layer_norms[i](x + attn_output)
            # Feed-forward
            ff_output = self.ff_layers[i](x)
            x = self.layer_norms[i](x + ff_output)
        
        # Global average pooling over sequence dimension to get a single vector per batch
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        x = self.output_ln(x)
        x = torch.nan_to_num(x)
        
        # Final output projection
        output = self.output_proj(x)
        residual = self.output_residual(x)
        # 让 Teacher latent 多一条 residual skip，减缓 attention 抖动
        return (output + residual) * 0.5
