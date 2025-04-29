import torch
import math 
from tqdm import tqdm 

def flatten_tensor(x):
    """Flattens a tensor across all dimensions except the batch dimension."""
    return x.view(x.size(0), -1)

def unflatten_tensor(x, orig_shape):
    """Reshapes a flattened tensor back to (B, *orig_shape)."""
    return x.view(x.size(0), *orig_shape)

def ensure_time_tensor(t, batch_size):
    """
    Ensures t is a 1D tensor of shape (batch_size,).
    If t is a scalar, expands it.
    """
    if t.dim() == 0:
        return t.expand(batch_size)
    elif t.dim() == 1:
        return t if t.size(0) == batch_size else t.expand(batch_size)
    else:
        if t.size(-1) == 1:
            return t.squeeze(-1)
        return t

def encode_xt_in_chunks(ae, batch, cond, T_render, chunk):
    """
    Encode input batch into stochastic latent representations in chunks.
    """
    B = batch.size(0)
    enc_chunk = min(chunk, B)
    n_encode_chunks = math.ceil(B / enc_chunk)
    xT_pieces = []

    for i in tqdm(range(0, B, enc_chunk), desc="Encoding chunks", total=n_encode_chunks):
        b_chunk = batch[i : i + enc_chunk]
        cond_chunk = cond[i : i + enc_chunk]
        xT_chunk = ae.encode_stochastic(b_chunk, cond_chunk, T=T_render)
        xT_pieces.append(xT_chunk)

    return torch.cat(xT_pieces, dim=0)
