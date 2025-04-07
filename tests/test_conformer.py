import pytest
import torch
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss

import sys
sys.path.append('../')

from src.conformer import *

class TestConformer:
    
    @pytest.fixture
    def model_params(self):
        return {
            'input_dim': 80,
            'd_model': 256,
            'num_heads': 4,
            'd_ff': 1024,
            'num_layers': 4,
            'kernel_size': 31,
            'dropout': 0.1,
            'vocab_size': 31
        }
    
    def test_positional_encoding(self):
        batch_size, seq_len, d_model = 2, 10, 64
        pe = PositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        # Check if positional encoding was added
        assert torch.any(output != x)
    
    def test_feed_forward(self):
        batch_size, seq_len, d_model, d_ff = 2, 10, 64, 256
        ff = FeedForward(d_model, d_ff, dropout=0.1)
        x = torch.randn(batch_size, seq_len, d_model)
        output = ff(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_multi_head_attention(self):
        batch_size, seq_len, d_model = 2, 10, 64
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads, dropout=0.1)
        q = k = v = torch.randn(batch_size, seq_len, d_model)
        output = mha(q, k, v)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Test with mask
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, 5:, :] = 0  # Mask out some positions
        output_masked = mha(q, k, v, mask)
        
        # Check output shape with mask
        assert output_masked.shape == (batch_size, seq_len, d_model)
        # Output should be different with mask
        assert not torch.allclose(output, output_masked, atol=1e-5)
    
    def test_conv_module(self):
        batch_size, seq_len, d_model = 2, 10, 64
        kernel_size = 7  # Smaller kernel for testing
        
        conv = ConvModule(d_model, kernel_size, dropout=0.1)
        x = torch.randn(batch_size, seq_len, d_model)
        output = conv(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_conformer_block(self):
        batch_size, seq_len, d_model = 2, 10, 64
        num_heads, d_ff, kernel_size = 4, 256, 7
        
        block = ConformerBlock(d_model, num_heads, d_ff, kernel_size, dropout=0.1)
        x = torch.randn(batch_size, seq_len, d_model)
        output = block(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Test with mask
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        output_masked = block(x, mask)
        assert output_masked.shape == (batch_size, seq_len, d_model)
    
    def test_conformer(self, model_params):
        batch_size, seq_len = 2, 10
        input_dim = model_params['input_dim']
        vocab_size = model_params['vocab_size']
        
        model = Conformer(**model_params)
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, vocab_size)
    
    def test_conformer_forward_backward(self, model_params):
        batch_size, seq_len = 2, 10
        input_dim = model_params['input_dim']
        vocab_size = model_params['vocab_size']
        
        model = Conformer(**model_params)
        x = torch.randn(batch_size, seq_len, input_dim)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        output = model(x)
        
        # Calculate loss
        criterion = CrossEntropyLoss()
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        
        # Check if loss is a tensor and has gradient
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Backward pass
        loss.backward()
        
        # Check if all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
    
    def test_conformer_with_mask(self, model_params):
        batch_size, seq_len = 2, 10
        input_dim = model_params['input_dim']
        vocab_size = model_params['vocab_size']
        
        model = Conformer(**model_params)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Create attention mask
        mask = torch.ones(batch_size, 1, seq_len, seq_len)
        mask[:, :, seq_len//2:, :] = 0  # Mask out bottom half
        
        # Forward pass with mask
        output = model(x, mask)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, vocab_size)
    
    def test_model_reproducibility(self, model_params):
        # Test that model produces the same output given the same input and seed
        torch.manual_seed(42)
        np.random.seed(42)
        
        batch_size, seq_len = 2, 10
        input_dim = model_params['input_dim']
        
        model = Conformer(**model_params)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output1 = model(x)
        
        # Reset seed and recreate model and input
        torch.manual_seed(42)
        np.random.seed(42)
        
        model2 = Conformer(**model_params)
        x2 = torch.randn(batch_size, seq_len, input_dim)
        
        output2 = model2(x2)
        
        # Check outputs are the same
        assert torch.allclose(output1, output2, atol=1e-6)
