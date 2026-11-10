import os
import random

import numpy as np
import torch

from pathlib import Path

def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch and request deterministic torch operations.

    Also configure available CUDA backends, set PYTHONHASHSEED for subsequent
    Python processes and print the selected seed. Unsupported deterministic
    operations produce warnings rather than errors.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'Setted seed to {seed}')

def resolve_path(path):
    path = Path(path)
    base_path = Path(__file__).resolve().parents[1]
    return path if path.is_absolute() else base_path / path
