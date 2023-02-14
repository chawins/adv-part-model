import contextlib

import numpy as np

from .dataloader_visualizer import *
from .distributed import *
from .image import *
from .loss import *
from .metric import *


@contextlib.contextmanager
def np_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
