import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

start_time = time.time()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LENGTH = 256

VOCAB = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 
    'E': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11,
    'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17,
    'Y': 18, 'V': 19, '_': 20
}

SS_MAP = {
    'H': 0,   # α-helix (4-turn helix)
    'G': 1,   # 3-10 helix (3-turn helix)
    'I': 2,   # π-helix (5-turn helix)
    'E': 3,   # β-strand (extended)
    'B': 4,   # β-bridge (isolated β-bridge)
    'T': 5,   # Turn
    'S': 6,   # Bend
    '-': 7,   # Coil/irregular (includes loops)
    '_': -100 # Padding
}
REVERSE_SS_MAP = {
    0: 'H', 1: 'G', 2: 'I', 3: 'E',
    4: 'B', 5: 'T', 6: 'S', 7: '-',
    -100: ''
}


class SSDataset(Dataset):
    def __init__(self, fasta_path, max_length=SEQ_LENGTH):
        self.sequences = []
        self.labels = []
        self._parse_fasta(fasta_path, max_length)
        
    def _parse_fasta(self, path, max_length):
        current_seq = ""
        current_ss = ""
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        self._add_entry(current_seq, current_ss, max_length)
                    current_seq = ""
                    current_ss = ""
                elif not current_seq:
                    current_seq = line
                else:
                    current_ss = line
            if current_seq:
                self._add_entry(current_seq, current_ss, max_length)
                
    def _add_entry(self, seq, ss, max_length):
        # Truncate sequences longer than max_length
        seq = seq[:max_length]
        ss = ss[:max_length]
        
        # Pad shorter sequences
        if len(seq) < max_length:
            pad_length = max_length - len(seq)
            seq += '_' * pad_length
            ss += '_' * pad_length
        
        # Convert to token IDs and labels
        token_ids = [VOCAB[aa] for aa in seq]
        labels = [SS_MAP[char] for char in ss]
        
        self.sequences.append(token_ids)
        self.labels.append(labels)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# Example usage:
train_dataset = SSDataset("dssp_train.dat")
val_dataset = SSDataset("dssp_val.dat")






train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

class SimpleSSPredictor(nn.Module):
    def __init__(self, num_aa=21, hidden_dim=2 * len(REVERSE_SS_MAP), kernel_size=9, dropout=0.01):
        super().__init__()
        # Embedding layer
        self.embed = nn.Embedding(num_aa, hidden_dim)

        # Add dropout layer
        # During training, inputs are randomly masked with probability p
        # the output of the layer is then rescaled by 1/(1-p)
        # The layer is not used in evaluation mode
        
        # Separable convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                      padding=kernel_size//2, groups=1),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                      padding=kernel_size//2, groups=1),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                      padding=kernel_size//2, groups=1),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Per-residue classifier
        self.classifier = nn.Linear(hidden_dim, 8)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embed' in name:
                    # small variance for embeddings prevent any feature to initially dominate
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'conv' in name or 'classifier' in name:
                    if len(param.shape) > 1:
                        # for relu variance set to 2 / # of input units to the layer
                        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
    def forward(self, x):
        # Input shape: (batch, seq_len)
        x = self.embed(x)  # (batch, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, seq_len)
        
        # Apply separable convolutions
        x = self.conv1(x)
        
        # Back to (batch, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)
        return self.classifier(x)
    

model = SimpleSSPredictor().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=SS_MAP['_'])  # Ignore padding
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training functions
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss (auto-ignores padding via ignore_index)
        loss = criterion(outputs.permute(0, 2, 1), labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = outputs.argmax(dim=-1)
        valid_mask = (labels != SS_MAP['_'])  # Exclude padding
        correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
        
        # Update stats
        total_loss += loss.item() * inputs.size(0)
        total_correct += correct
        total_tokens += valid_mask.sum().item()
    
    return total_loss / len(loader.dataset), total_correct / total_tokens

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.permute(0, 2, 1), labels)
            
            preds = outputs.argmax(dim=-1)
            valid_mask = (labels != SS_MAP['_'])
            correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
            
            total_loss += loss.item() * inputs.size(0)
            total_correct += correct
            total_tokens += valid_mask.sum().item()
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, optimizer, criterion, 
                num_epochs=20, model_save_path="ss_model.pth", check_every=1):
    """
    Train and validate the model with automatic best-model saving
    
    Args:
        model: Initialized model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Initialized optimizer
        criterion: Loss function
        num_epochs: Number of training epochs
        model_save_path: Path to save best model weights
        check_every: How often (in epochs) we check the model performance
        
    Returns:
        Tuple of (best_model, training_stats) where stats contains:
        {'train_loss': [...], 'train_acc': [...], 'val_loss': [...], 'val_acc': [...]}
    """
    model = model.to(device)
    best_val_acc = 0.0
    training_stats = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Save stats
        training_stats['train_loss'].append(train_loss)
        training_stats['train_acc'].append(train_acc)
        training_stats['val_loss'].append(val_loss)
        training_stats['val_acc'].append(val_acc)

        if (epoch+1)%check_every==0:
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                print(f"Saved new best model with val_acc: {val_acc:.2%}")
    
    # Load best model weights before returning
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    return model, training_stats


# Run training
trained_model, stats = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=20,
    model_save_path="best_ss_model.pth",
    check_every=1
)

def predict_ss(sequence, model, max_length=SEQ_LENGTH):
    """Predict secondary structure for any length sequence"""
    # Preprocess sequence
    seq = sequence[:max_length]  # Truncate if needed
    if len(seq) < max_length:
        seq += '_' * (max_length - len(seq))  # Pad if needed
    
    # Convert to tokens
    token_ids = [VOCAB.get(aa, VOCAB['_']) for aa in seq]
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        preds = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
    
    # Convert to SS symbols
    ss_pred = ''.join(REVERSE_SS_MAP[p] for p in preds)
    
    # Remove padding predictions
    return ss_pred[:len(sequence)]  # Return only the length of original sequence

sequence = "MSEEKAVSTEERGSRKVRTGYVVSDKMEKTIVVELEDRVKHPLYGKIIRRTSKVKAHDENGVAGIGDRVQLMETRPLSATKHWRLVEVLEKAK"
ss_pred = predict_ss(sequence, model)
print(f"Seq: {sequence}")
print(f"Pre: {ss_pred}")
print("Tru: ----------------EEEEEEEEEEETTEEEEEEEEEEE-TTT--EEEEEEEEEEE-SS----TT-EEEEEE-S-SSSSEEEEEEEEEE---")





end_time = time.time()

print(f"実行時間:{end_time - start_time}sec")