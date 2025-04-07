"""Conformer Model Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and split into multiple heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Combine heads
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        return self.out(output)

class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, expansion_factor=2, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * expansion_factor, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        
        # Depthwise conv with padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, groups=d_model
        )
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        residual = x
        x = self.layer_norm(x)
        
        # Change to [batch_size, d_model, seq_len] for Conv1d
        x = x.transpose(1, 2)
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Change back to [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        return x + residual

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        
        # Half-step feed-forward modules
        self.ff1 = FeedForward(d_model, d_ff, dropout)
        self.ff1_norm = nn.LayerNorm(d_model)
        
        # Self-attention module
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv = ConvModule(d_model, kernel_size, dropout=dropout)
        
        # Second half-step feed-forward
        self.ff2 = FeedForward(d_model, d_ff, dropout)
        self.ff2_norm = nn.LayerNorm(d_model)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # First half-step feed-forward
        residual = x
        x = self.ff1_norm(x)
        x = 0.5 * self.ff1(x)
        x = residual + self.dropout(x)
        
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, x, x, mask)
        x = residual + self.dropout(x)
        
        # Convolution module
        x = x + self.conv(x)
        
        # Second half-step feed-forward
        residual = x
        x = self.ff2_norm(x)
        x = 0.5 * self.ff2(x)
        x = residual + self.dropout(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x

class Conformer(nn.Module):
    def __init__(
        self, 
        input_dim=80,
        d_model=256, 
        num_heads=4, 
        d_ff=2048, 
        num_layers=12,
        kernel_size=31,
        dropout=0.1,
        vocab_size=31
    ):
        super().__init__()
        
        # Feature projection
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        
        # Project features to model dimension
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        # Final projection to vocabulary size
        output = self.output_projection(x)
        
        return output
    