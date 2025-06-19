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

print(len(train_dataset))

all_ss=np.concatenate(val_dataset.labels)
n_pad=sum(all_ss==SS_MAP['_'])
for ss in SS_MAP:
    if ss=='_': continue
    fraction=sum(all_ss==SS_MAP[ss])/(len(all_ss)-n_pad)
    print(f"{ss} {100*fraction:5.1f}%")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

class SimpleSSPredictor(nn.Module):
    def __init__(self, num_aa=21, hidden_dim=4, kernel_size=5, dropout=0.0):
        super().__init__()
        # Embedding layer
        self.embed = nn.Embedding(num_aa, hidden_dim)

        # Add dropout layer
        # During training, inputs are randomly masked with probability p
        # the output of the layer is then rescaled by 1/(1-p)
        # The layer is not used in evaluation mode
        
        # Separable convolution layers
        self.conv1 = nn.Sequential(
            # Depthwise convolution (per-channel)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, 
                      padding=kernel_size//2, groups=hidden_dim),
            # Pointwise convolution (channel mixing)
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

#print("manko")



























end_time = time.time()

print(f"実行時間:{end_time - start_time}sec")



