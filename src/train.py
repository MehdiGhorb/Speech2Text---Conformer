import torch
import torchaudio
from tqdm import tqdm

def collate_fn(batch):
    """
    Custom collate function for variable length sequences.
    """
    features, tokens, feature_lengths, token_lengths = zip(*batch)
    
    # Pad features
    max_feature_len = max(feature_lengths)
    padded_features = []
    for feat, feat_len in zip(features, feature_lengths):
        padded_feat = torch.zeros(max_feature_len, feat.size(1))
        padded_feat[:feat_len] = feat
        padded_features.append(padded_feat)
    
    # Pad tokens
    max_token_len = max(token_lengths)
    padded_tokens = []
    for tok, tok_len in zip(tokens, token_lengths):
        padded_tok = torch.zeros(max_token_len, dtype=torch.long)
        padded_tok[:tok_len] = tok
        padded_tokens.append(padded_tok)
    
    # Stack into batches
    features_batch = torch.stack(padded_features)
    tokens_batch = torch.stack(padded_tokens)
    feature_lengths = torch.tensor(feature_lengths)
    token_lengths = torch.tensor(token_lengths)
    
    return features_batch, tokens_batch, feature_lengths, token_lengths

def train(model, train_loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training")
    
    for batch_idx, (features, targets, feature_lengths, target_lengths) in enumerate(pbar):
        features, targets = features.to(device), targets.to(device)
        feature_lengths, target_lengths = feature_lengths.to(device), target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        with torch.amp.autocast('cuda'):  # Enable mixed precision
            outputs = model(features)
            outputs = outputs.log_softmax(dim=-1)
            outputs = outputs.transpose(0, 1)  # CTC needs (time, batch, classes)
            
            loss = criterion(outputs, targets, feature_lengths, target_lengths)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for features, targets, feature_lengths, target_lengths in val_loader:
            features, targets = features.to(device), targets.to(device)
            feature_lengths, target_lengths = feature_lengths.to(device), target_lengths.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Compute loss (CTC)
            outputs = outputs.log_softmax(dim=-1)
            outputs = outputs.transpose(0, 1)  # CTC needs (time, batch, classes)
            
            loss = criterion(outputs, targets, feature_lengths, target_lengths)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def decode_predictions(outputs, dataset):
    """
    Decode model predictions from probabilities to text.
    """
    # Get the most likely class at each timestep
    pred_indices = torch.argmax(outputs, dim=-1)
    
    texts = []
    for indices in pred_indices:
        text = dataset.decode_ctc(indices)
        texts.append(text)
    
    return texts

def inference(model, dataset, audio_path, device):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Extract features
    features = dataset.get_features(waveform, sample_rate)
    
    # Add batch dimension
    features = features.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        outputs = outputs.log_softmax(dim=-1)
    
    # Decode
    pred_indices = torch.argmax(outputs[0], dim=-1)
    text = dataset.decode_ctc(pred_indices)
    
    return text
