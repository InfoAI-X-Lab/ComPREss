"""Shared utility helpers."""

import gc
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
