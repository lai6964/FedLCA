import random

import numpy as np
import torch

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_id=0):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid device_id={device_id}; available CUDA devices: "
            f"0..{torch.cuda.device_count() - 1}"
        )
    return torch.device(f"cuda:{device_id}")


def flatten_tensors(tensors):
    if len(tensors) == 0:
        return torch.tensor([])
    return torch.cat([t.detach().reshape(-1).cpu() for t in tensors])


def cosine_similarity(x, y, eps=1e-12):
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    return float(torch.dot(x, y) / (torch.norm(x) * torch.norm(y) + eps))


def normalize_dict(d):
    values = np.array(list(d.values()), dtype=np.float64)
    if len(values) == 0:
        return d
    min_v, max_v = values.min(), values.max()
    if abs(max_v - min_v) < 1e-12:
        return {k: 1.0 for k in d}
    return {k: float((v - min_v) / (max_v - min_v)) for k, v in d.items()}


def ema_update(old, new, beta=0.9):
    return beta * old + (1.0 - beta) * new
