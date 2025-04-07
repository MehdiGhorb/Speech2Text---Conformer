import pytest
import torch
import numpy as np
import torchaudio
import os
from unittest.mock import MagicMock, patch

import sys
sys.path.append('../')

from src.data_loader import * 

# Create a fixture for a mock LibriSpeech dataset
@pytest.fixture
def mock_librispeech_data():
    # Sample data that would be returned by the LIBRISPEECH dataset
    waveform = torch.rand(1, 16000)  # 1 second of audio at 16kHz
    sample_rate = 16000
    utterance = "hello world"
    speaker_id = 1
    chapter_id = 1
    utterance_id = 1
    return [(waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)]

# Patch the LIBRISPEECH class
@pytest.fixture
def mock_dataset(mock_librispeech_data):
    with patch('torchaudio.datasets.LIBRISPEECH', autospec=True) as mock_librispeech:
        mock_instance = mock_librispeech.return_value
        mock_instance.__len__.return_value = len(mock_librispeech_data)
        mock_instance.__getitem__.side_effect = lambda idx: mock_librispeech_data[idx]
        yield mock_librispeech

class TestLibriSpeechDataset:
    @pytest.fixture
    def dataset(self, mock_dataset):
        # Create a dataset instance with the mocked LIBRISPEECH
        return LibriSpeechDataset(root="./data", split="test-clean")
    
    def test_initialization(self, dataset):
        """Test that the dataset initializes correctly."""
        assert dataset is not None
        assert hasattr(dataset, 'dataset')
        assert hasattr(dataset, 'char_map')
        assert hasattr(dataset, 'idx_map')
        assert dataset.blank_idx == 30
        
    def test_get_item(self, dataset):
        """Test that __getitem__ returns the expected format."""
        features, tokens, feat_len, token_len = dataset[0]
        
        # Check types
        assert isinstance(features, torch.Tensor)
        assert isinstance(tokens, torch.Tensor)
        assert isinstance(feat_len, int)
        assert isinstance(token_len, int)
        
        # Check shapes and values
        assert features.dim() == 2  # (time, features)
        assert tokens.dim() == 1    # (sequence_length)
        assert feat_len == features.size(0)
        assert token_len == tokens.size(0)
        
    def test_get_features(self, dataset):
        """Test the feature extraction functionality."""
        # Create a sample waveform
        waveform = torch.rand(1, 16000)  # 1 second at 16kHz
        sample_rate = 16000
        
        features = dataset.get_features(waveform, sample_rate)
        
        # Check output type and shape
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2  # (time, features)
        assert features.size(1) == 80  # n_mels=80
        
    def test_resampling(self, dataset):
        """Test that audio gets resampled correctly."""
        # Create a sample waveform at a different sample rate
        waveform = torch.rand(1, 8000)  # 1 second at 8kHz
        sample_rate = 8000
        
        features = dataset.get_features(waveform, sample_rate)
        
        # Check that resampling doesn't crash and produces valid output
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert features.size(1) == 80  # n_mels=80
        
    def test_tokenize(self, dataset):
        """Test the tokenization of text."""
        text = "hello"
        tokens = dataset.tokenize(text)
        
        # Expected token indices for "hello"
        expected = torch.tensor([8, 5, 12, 12, 15], dtype=torch.long)
        
        assert torch.equal(tokens, expected)
        
    def test_tokenize_with_unsupported_chars(self, dataset):
        """Test tokenization with characters not in the char map."""
        text = "hello!"  # '!' is not in the char map
        tokens = dataset.tokenize(text)
        
        # Expected token indices for "hello" (! is ignored)
        expected = torch.tensor([8, 5, 12, 12, 15], dtype=torch.long)
        
        assert torch.equal(tokens, expected)
        
    def test_decode(self, dataset):
        """Test decoding token indices back to text."""
        # Indices for "hello"
        indices = torch.tensor([8, 5, 12, 12, 15])
        text = dataset.decode(indices)
        
        assert text == "hello"
        
    def test_decode_with_blank(self, dataset):
        """Test decoding with blank tokens."""
        # Indices for "hello" with blank tokens (30)
        indices = torch.tensor([8, 30, 5, 12, 30, 12, 15])
        text = dataset.decode(indices)
        
        assert text == "hello"  # Blanks should be ignored
        
    def test_decode_ctc(self, dataset):
        """Test CTC decoding (merging repeated tokens and removing blanks)."""
        # Indices for "hello" with repeats and blanks
        indices = torch.tensor([8, 8, 30, 5, 12, 12, 30, 12, 15, 15])
        text = dataset.decode_ctc(indices)
        
        assert text == "hello"  # Repeats and blanks should be merged/removed
            
    def test_all_characters_in_char_map(self, dataset):
        """Test that all characters in the map are correctly handled."""
        # Create a string with all characters in the map
        all_chars = " abcdefghijklmnopqrstuvwxyz'.-"
        tokens = dataset.tokenize(all_chars)
        decoded = dataset.decode(tokens)
        
        assert decoded == all_chars
        
    def test_idx_map_is_inverse_of_char_map(self, dataset):
        """Test that idx_map is the correct inverse of char_map."""
        for char, idx in dataset.char_map.items():
            if char != "<blank>":  # Skip the special blank token
                assert dataset.idx_map[idx] == char

# Integration test with actual data (optional, depends on availability)
@pytest.mark.skipif(not os.path.exists("./data"), 
                    reason="LibriSpeech data not available")
def test_with_real_data():
    """Test with real LibriSpeech data if available."""
    dataset = LibriSpeechDataset(root="./data", split="dev-clean")
    assert len(dataset) > 0
    
    # Test getting an item
    features, tokens, feat_len, token_len = dataset[0]
    
    assert features.dim() == 2
    assert tokens.dim() == 1
    assert feat_len > 0
    assert token_len > 0
