from typing import Tuple, Union

import pandas as pd


def mad(x: pd.Series, constant: float = 1.4826, center=None) -> pd.Series:
    center = center or x.median()
    return constant * abs(x - center).median()


def get_gran_and_period(ts: pd.Series) -> Tuple[str, Union[int, None]]:
    sorted_ts = sorted(list(ts))
    sec_diff = (sorted_ts[-1] - sorted_ts[-2]).seconds

    if sec_diff >= 604800:
        return 'week', 52
    elif sec_diff >= 86400:
        return 'day', 7
    elif sec_diff >= 3600:
        return 'hour', 24
    elif sec_diff >= 60:
        return 'min', 1440
    elif sec_diff >= 1:
        return 'sec', 1440
    else:
        return 'ms', None
