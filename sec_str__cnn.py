import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


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


