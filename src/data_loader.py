"""1. Data Preparation"""

from torchaudio.datasets import LIBRISPEECH
import torch
import torchaudio
import torch.utils.data as data
import torchaudio.transforms as T

class LibriSpeechDataset(data.Dataset):
    def __init__(self, root, split="train-clean-100", transform=None, target_transform=None):
        self.dataset = LIBRISPEECH(root=root, url=split, download=True)
        self.transform = transform
        self.target_transform = target_transform
        
        # Character mapping for CTC loss
        self.char_map = {
            " ": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9,
            "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18,
            "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26,
            "'": 27, ".": 28, "-": 29, "<blank>": 30
        }
        self.idx_map = {v: k for k, v in self.char_map.items()}
        
        # blank token for CTC loss
        self.blank_idx = 30
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        waveform, sample_rate, utterance, _, _, _ = self.dataset[idx]
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(waveform)
        else:
            # Default transform: mel spectrogram
            features = self.get_features(waveform, sample_rate)
        
        # Tokenize the utterance
        tokens = self.tokenize(utterance.lower())
        
        return features, tokens, len(features), len(tokens)
    
    def get_features(self, waveform, sample_rate):
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Extract mel spectrogram features
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        
        features = mel_transform(waveform)
        # Convert to log mel
        features = torch.log(features + 1e-9)
        
        # Normalize
        mean = features.mean()
        std = features.std()
        features = (features - mean) / (std + 1e-9)
        
        # Convert to (time, feature)
        features = features.squeeze(0).transpose(0, 1)
        
        return features
    
    def tokenize(self, text):
        # Convert text to token indices
        tokens = []
        for char in text:
            if char.lower() in self.char_map:
                tokens.append(self.char_map[char.lower()])
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, token_indices):
        # Convert token indices back to text
        text = ""
        for idx in token_indices:
            idx = idx.item()
            if idx != self.blank_idx:  # Skip blank token
                text += self.idx_map.get(idx, "")
        return text
    
    def decode_ctc(self, token_indices):
        # Merge repeated characters and remove blanks
        result = []
        prev_idx = -1
        for idx in token_indices:
            idx = idx.item()
            if idx != self.blank_idx and idx != prev_idx:
                result.append(idx)
            prev_idx = idx
        
        # Convert indices to characters
        text = ""
        for idx in result:
            if idx in self.idx_map:
                text += self.idx_map[idx]
        return text
    