import random

import numpy as np

__GLOBAL_SEED = 42


def get_global_seed():
    return __GLOBAL_SEED


def fix_global_seed(seed=None):
    # Lazy import: torch is slow to load (~800ms), defer until needed
    import torch  # noqa: PLC0415

    if seed is None:
        seed = get_global_seed()
    global __GLOBAL_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # (Earlier versions assigned ``torch.use_deterministic_algorithms = True`` here, which replaced the
    # torch *function* with a bool: a no-op for eager but it made torch.compile / dynamo fail.)
    __GLOBAL_SEED = seed
