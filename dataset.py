import re, os
import numpy as np

token_map = {'P': [1.0,0.0,0.0], 'B': [0.0,1.0,0.0], 'T': [0.0,0.0,1.0]}
idx_map = {'P':0,'B':1,'T':2}

def parse_history(text):
    toks = re.findall(r"\b([PpBbTt])[A-Za-z]*\b", text)
    if not toks:
        toks = re.findall(r"[PpBbTt]", text)
    return [t.upper() for t in toks]

def make_dataset(history, seq_len=5):
    X = []
    Y = []
    for i in range(0, len(history)-seq_len):
        seq = history[i:i+seq_len]
        nxt = history[i+seq_len]
        X.append([token_map.get(ch, [0,0,0]) for ch in seq])
        y = [0,0,0]; 
        if nxt in idx_map: y[idx_map[nxt]] = 1
        Y.append(y)
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='float32')
    return X, Y
