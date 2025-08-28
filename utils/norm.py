import numpy as np
import pandas as pd

### New
def normalize_i(
    i: np.ndarray | pd.Series,
    ugw_n: tuple | None = None,
):
    if ugw_n is None:
        return i * 1e3
    return i * 1e6 / (ugw_n[0] * ugw_n[1])