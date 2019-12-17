import pandas as pd
import scipy.stats
from math import floor, sqrt
from statistics import median, stdev

from main.pySTL import pySTL


def get_gran(df, col_name='timestamp'):
    sorted_ts = sorted(list(df[col_name]))
    sec_diff = (sorted_ts[-1] - sorted_ts[-2]).seconds

    if sec_diff >= 604800:
        return 'week'
    elif sec_diff >= 86400:
        return 'day'
    elif sec_diff >= 3600:
        return 'hour'
    elif sec_diff >= 60:
        return 'min'
    elif sec_diff >= 1:
        return 'sec'
    else:
        return 'ms'


def calculate_e_value(df: pd.DataFrame, calc='orig', weights=None) -> pd.Series:
    median_val = median(list(df['value']))

    e_value = df['seasonal']
    exp_value_median = e_value + median_val
    exp_value_trend = e_value + df['trend']

    # Calc e_value based on pythonic 'switch'
    if calc == 'orig':
        exp_value = exp_value_median
    elif calc == 'trend':
        exp_value = exp_value_trend
    elif calc == 'hybrid':
        exp_value = (exp_value_median * 0.5) + (exp_value_trend * 0.5)
    elif calc == 'other':
        exp_value = (exp_value_median * weights['median']) + (exp_value_trend * weights['trend'])
    else:
        raise ValueError("incorrect 'calc'")

    exp_value[exp_value < 0] = 0
    return exp_value


def mad(x: pd.Series, constant=1.4826, center=None) -> pd.Series:
    cntr = center or median(x)
    return constant * median(abs(x - cntr))


def extract_anomalies(data, max_outliers, alpha=0.05, one_tail=False, upper_tail=True, verbose=False):
    wip = data.copy()
    n = len(wip)
    anom_df = pd.DataFrame.from_dict({
        name: pd.Series(data=None, dtype=series.dtype)
        for name, series in wip.iteritems()
    })
    num_anoms = 0
    for i in range(1, max_outliers+1, 1):
        data_mad = mad(wip['univ_rem'])

        # Check for and protect against constant time series (Uniform values)
        if data_mad == 0:
            break

        # Calculate abs_dev for each datapoint
        if one_tail:
            if upper_tail:
                abs_dev = wip['univ_rem'] - median(wip['univ_rem'])
            else:
                abs_dev = median(wip['univ_rem']) - wip['univ_rem']
        else:
            abs_dev = abs(wip['univ_rem'] - median(wip['univ_rem']))

        # Calc Med's away from median for each and find max datapoint
        # Calculates absolute deviation for each point / total mad.
        # Then we find the max deviation across all points and filter for these -
        # We drop them in the R-idx list and drop from working set.
        ares = abs_dev/data_mad
        R = max(ares)

        # Calculate Probability
        if one_tail:
            p_val = 1 - alpha/(n-i+1)
        else:
            p_val = 1 - alpha/(2*(n-i+1))

        # Calculate Threshold based on prob.
        t_val = scipy.stats.t.ppf(p_val, (n-i-1))
        lam = t_val * (n-i) / sqrt((n-i-1+t_val**2)*(n-i+1))

        # Find the index in WIP for the the 'potential' anomaly.
        # Add it to anom_df then remove from wip
        tmp_idx = ares[ares == R].index[0]
        anom_df = anom_df.append(wip.loc[tmp_idx, :])
        wip.drop(tmp_idx, inplace=True)

        # If R is larger than the threshold then increment num_anoms counter.
        if R > lam:
            num_anoms = i

        if verbose:
            print(f"{i}/{max_outliers} completed.")

    anoms = anom_df.iloc[0:num_anoms, :].copy().sort_index(inplace=True)

    # Add Significance by # of StDev's
    anoms['significance'] = anoms['univ_rem']/stdev(data['univ_rem'])
    return anoms


def detect_anoms(data, k=0.49, alpha=0.05, period=None, one_tail=False, upper_tail=True, robust=True,
                 e_value_calc='orig', data_seasonal=True, verbose=False, **kwargs):

    # Init Variables
    num_obs = len(data)
    max_outliers = floor(num_obs*k)

    # Set Robust Params:
    inner, outer = 2, 1
    if robust:
        inner, outer = 1, 15

    # Check kwargs for certain keys:
    weights = kwargs.get('weights', None)
    s_window = kwargs.get('s_window', 'periodic')

    # Data QA Checks
    # Check data input for correct formatting
    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a dataframe.")
    if len(data.columns) != 2:
        raise ValueError("'data' must be a 2 column data frame with the first column being a set"
                         " of timestamps and the second column being numeric values.")

    data.columns = ['timestamp', 'value']

    # Make sure second column is float
    data.iloc[:, 1] = data.iloc[:, 1].astype(float)

    # Check s_window param:
    if s_window and not isinstance(s_window, str) and s_window < 4:
        s_window = 4

    if not period and data_seasonal:
        raise ValueError("Must Supply period length for time series decomposition")

    # Check to make sure we have at least two periods worth of data for anomaly context
    if num_obs < (period * 2) and data_seasonal:
        raise ValueError("Anom detection needs at least 2 periods worth of data")

    if len(list(filter(lambda x: pd.isnull(x), [None] + list(data.iloc[:, 1]) + [None]))) > 3:
        raise ValueError("Data contains non-leading null values. Please ensure that all values in the set are non-null.")
    else:
        data.dropna(subset=['value'])

    if max_outliers == 0:
        raise ValueError(f"Not enough observations in this period for AnomalyDetection. "
                         f"There are {num_obs}, and k = {k}; If the maximum number of anomalies that "
                         f"can be detected are k * num_obs, then the maximum number of anomalies "
                         f"that can be detected is less than 0.")

    # Check e_value_calc and weights:
    if e_value_calc == 'other':
        if not weights or not isinstance(weights, dict):
            raise ValueError("If 'other' is specified for 'e_value_calc' then weights must be supplied as dict")
        elif sum(weights.values()) != 1:
            raise ValueError("If 'other' is specified for 'e_value_calc' then weights must sum to 1")
        elif not all(x in weights for x in ['median', 'trend']):
            raise ValueError("If 'other' is specified then 'median' and 'trend' must be specified in weights.")
    else:
        weights = None

    # Main Piece
    # Step 1 - decompose data into raw, seasonal, trend and remainder;
    # Then attach seasonal, trend, and remainder to orig data and store as data_decomp
    stl = pySTL(list(data.iloc[:, 1]), period=period, s_window=s_window, inner=inner, outer=outer)
    data_decomp = pd.concat([data, stl], axis=1)

    if not data_seasonal:
        data_decomp['seasonal'] = 0

    data_decomp['e_value'] = calculate_e_value(data_decomp, calc=e_value_calc, weights=weights)
    data_decomp['univ_rem'] = data_decomp['value'] - data_decomp['e_value']

    data_wip = data_decomp[['timestamp', 'value', 'e_value', 'univ_rem']].copy()

    anoms = extract_anomalies(data_wip, max_outliers, alpha, one_tail, upper_tail, verbose)

    return data_decomp, anoms


if __name__ == '__main__':

    data_path = '/Users/brandonarino/Desktop/test_hourly.csv'
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    period = 24
    data_seasonal = True
    one_tail = True
    upper_tail = True
    test = detect_anoms(data, period=period, data_seasonal=data_seasonal, one_tail=one_tail, upper_tail=upper_tail)
    print(test)
