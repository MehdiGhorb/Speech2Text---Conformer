import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append('../')

from src.train import *

class MockModel(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256, output_dim=30):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

class MockDataset(Dataset):
    def __init__(self, num_samples=10, max_feature_len=100, feature_dim=80, max_token_len=20):
        self.num_samples = num_samples
        self.max_feature_len = max_feature_len
        self.feature_dim = feature_dim
        self.max_token_len = max_token_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create variable length features and tokens
        feature_len = np.random.randint(50, self.max_feature_len)
        token_len = np.random.randint(10, self.max_token_len)
        
        features = torch.randn(feature_len, self.feature_dim)
        tokens = torch.randint(0, 29, (token_len,), dtype=torch.long)
        
        return features, tokens, feature_len, token_len
    
    def decode_ctc(self, indices):
        # Mock implementation of CTC decoding
        # Remove repeats and special tokens
        prev = -1
        result = []
        for idx in indices:
            idx = idx.item()
            if idx != prev and idx != 0:  # 0 is blank token
                result.append(chr(idx + 96))  # Convert to ascii characters
            prev = idx
        return ''.join(result)

# Now write tests for each function in the training utilities

def test_collate_fn():
    # Create a batch of samples with different lengths
    batch = [
        (torch.randn(5, 80), torch.randint(0, 29, (3,)), 5, 3),
        (torch.randn(8, 80), torch.randint(0, 29, (6,)), 8, 6),
        (torch.randn(3, 80), torch.randint(0, 29, (2,)), 3, 2)
    ]
    
    # Call collate_fn
    features_batch, tokens_batch, feature_lengths, token_lengths = collate_fn(batch)
    
    # Check shapes
    assert features_batch.shape == (3, 8, 80), f"Expected shape (3, 8, 80), got {features_batch.shape}"
    assert tokens_batch.shape == (3, 6), f"Expected shape (3, 6), got {tokens_batch.shape}"
    assert torch.all(feature_lengths == torch.tensor([5, 8, 3])), f"Expected [5, 8, 3], got {feature_lengths}"
    assert torch.all(token_lengths == torch.tensor([3, 6, 2])), f"Expected [3, 6, 2], got {token_lengths}"
    
    # Check padding is correct
    assert torch.all(features_batch[0, 5:] == 0), "Padding in features is incorrect"
    assert torch.all(tokens_batch[0, 3:] == 0), "Padding in tokens is incorrect"
    assert torch.all(tokens_batch[2, 2:] == 0), "Padding in tokens is incorrect"

def test_evaluate():
    # Set up test components
    device = torch.device("cpu")
    model = MockModel()
    dataset = MockDataset(num_samples=10)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    criterion = nn.CTCLoss()
    
    # Call evaluate function
    avg_loss = evaluate(model, dataloader, criterion, device)
    
    # Check that loss is a float
    assert isinstance(avg_loss, float), f"Expected float, got {type(avg_loss)}"

def test_decode_predictions():
    # Create mock outputs (batch_size=2, seq_len=5, num_classes=30)
    outputs = torch.zeros(2, 5, 30)
    
    # Set specific indices to have high probability
    # First sequence: "hello"
    outputs[0, 0, 8] = 10.0  # 'h'
    outputs[0, 1, 5] = 10.0  # 'e'
    outputs[0, 2, 12] = 10.0  # 'l'
    outputs[0, 3, 12] = 10.0  # 'l'
    outputs[0, 4, 15] = 10.0  # 'o'
    
    # Second sequence: "test"
    outputs[1, 0, 20] = 10.0  # 't'
    outputs[1, 1, 5] = 10.0   # 'e'
    outputs[1, 2, 19] = 10.0  # 's'
    outputs[1, 3, 20] = 10.0  # 't'
    outputs[1, 4, 0] = 10.0   # blank (should be ignored)
    
    dataset = MockDataset()
    
    # Call decode_predictions
    texts = decode_predictions(outputs, dataset)
    
    # Check results
    assert len(texts) == 2, f"Expected 2 decoded texts, got {len(texts)}"
    assert texts[1] == "test", f"Expected 'test', got '{texts[1]}'"

# Test edge cases

def test_collate_fn_empty_batch():
    # Empty batch should be handled gracefully
    with pytest.raises(ValueError):
        collate_fn([])

def test_collate_fn_single_item():
    # Single item batch
    batch = [(torch.randn(5, 80), torch.randint(0, 29, (3,)), 5, 3)]
    
    features_batch, tokens_batch, feature_lengths, token_lengths = collate_fn(batch)
    
    assert features_batch.shape == (1, 5, 80)
    assert tokens_batch.shape == (1, 3)
    assert feature_lengths.shape == (1,)
    assert token_lengths.shape == (1,)

def test_decode_predictions_empty():
    # Empty batch
    outputs = torch.zeros(0, 5, 30)
    dataset = MockDataset()
    
    texts = decode_predictions(outputs, dataset)
    assert len(texts) == 0
