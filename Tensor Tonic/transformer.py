import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    return nn.Embedding(vocab_size,d_model)

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    return embedding(tokens) * math.sqrt(d_model)

d = 32
emb = create_embedding_layer(int(d/2),d)
print(emb)
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])

print(embed_tokens(emb,input,d)[0,0])


input2 = torch.LongTensor([1])
print(embed_tokens(emb,input2,d)[0])



import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    res = np.zeros(seq_length,d_model)
    
    
    

x = np.arange(10)

print(x)
