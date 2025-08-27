import numpy as np
import pandas as pd

# def normalize_i(
#     i: np.ndarray | pd.Series,
#     ugw_n: tuple,
#     # n_finger: float = 2.0,
#     # width: float = 25.0,
# ):
#     return i * 1e6 / (ugw_n[0] * ugw_n[1])

### New
def normalize_i(
    i: np.ndarray | pd.Series,
    # ugw_n: tuple,
    n_finger: float = 2.0,
    width: float = 75.0,
):
    return i * 1e6 / (n_finger * width)