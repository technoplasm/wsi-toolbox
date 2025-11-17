import random

import numpy as np
import torch

__GLOBAL_SEED = 42

def get_global_seed():
    return __GLOBAL_SEED

def fix_global_seed(seed=None):
    if seed is None:
        seed = get_global_seed()
    global __GLOBAL_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    __GLOBAL_SEED = seed
