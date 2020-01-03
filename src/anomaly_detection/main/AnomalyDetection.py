from typing import Tuple

import scipy.stats
import pandas as pd
import datetime as dt
from math import floor, sqrt

from anomaly_detection.main import models
from anomaly_detection.main.utils import get_gran_and_period, mad
from anomaly_detection.main.stl_decomposition.STLDecomposition import STLDecomposition


class AnomalyDetection(object):

    def __init__(self,
                 max_anoms: float = 0.10,
                 alpha: float = 0.05,
                 direction: str = 'pos',
                 threshold: str = 'None',
                 verbose: bool = False):

        if max_anoms > 0.49 or max_anoms <= 0:
            raise ValueError("'max_anoms' must be a percentage greater than 0 and less than 50%.")

        if not (0.01 <= alpha <= 0.1):
            raise ValueError("Warning: alpha is the statistical significance, and is usually between 0.01 and 0.1")

        if threshold.lower() not in ['none', 'med_max', 'p95', 'p99']:
            raise ValueError("'threshold' options are one of: {'None', 'med_max', 'p95', 'p99'}")

        if direction.lower() not in ['pos', 'neg', 'both']:
            raise ValueError("'direction' options are one of: {'pos', 'neg', 'both'}")

        self.max_anoms = max_anoms
        self.alpha = alpha
        self.direction = direction
        self.threshold = threshold
        self.verbose = verbose

    @staticmethod
    def _prepare_dateframe(df):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            raise ValueError("Cannot convert first column of dataframe to datetime.")

        # Deal with NA's in timestamps - drop rows where timestamp is null
        if any(pd.isnull(df['timestamp'])):
            df.dropna(subset=['timestamp'], inplace=True)

        out_df = df.copy()
        out_df.columns = ['timestamp', 'value']
        out_df['value'] = out_df['value'].astype(float)

        return out_df, df

    @staticmethod
    def _convert_to_longterm_array(df, gran, period, periods_in_longterm):
        num_obs = df.shape[0]
        if gran == 'day':
            num_obs_in_period = period * periods_in_longterm + 1
            num_days_in_period = (7 * periods_in_longterm) + 1
        elif gran == 'week':
            num_obs_in_period = period * periods_in_longterm + 1
            num_days_in_period = (52 * periods_in_longterm) + 1
        else:
            num_obs_in_period = period * 7 * periods_in_longterm
            num_days_in_period = (7 * periods_in_longterm)

        # Store last date in time series
        all_data = []
        last_date = df.iloc[max(df.index), 0]
        # Subset x into piecewise_median_period_weeks chunks
        for i in range(0, num_obs, num_obs_in_period):
            start_date = df.iloc[i, 0]
            end_date = min(start_date + dt.timedelta(days=num_days_in_period), last_date)
            # if there is at least 'num_days_in_period' days left, subset it,
            # otherwise subset last_date - num_days_in_period
            if (end_date - start_date).days == num_days_in_period:
                all_data.append(df.loc[(df.timestamp >= start_date) & (df.timestamp < end_date), :].copy())
            else:
                alt_start = last_date - dt.timedelta(days=num_days_in_period)
                all_data.append(df.loc[(df.timestamp > alt_start) & (df.timestamp <= last_date), :].copy())
        return all_data

    @staticmethod
    def _determine_one_and_upper_tail(direction: str):
        one_tail = True if direction in ['pos', 'neg'] else False
        upper_tail = True if direction in ['pos', 'both'] else False
        return one_tail, upper_tail

    def _extract_anomalies(self, data: pd.DataFrame, max_outliers: int) -> pd.DataFrame:
        wip = data.copy()
        n = len(wip)
        one_tail, upper_tail = self._determine_one_and_upper_tail(self.direction)

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
                    abs_dev = wip['univ_rem'] - wip['univ_rem'].median()
                else:
                    abs_dev = wip['univ_rem'].median() - wip['univ_rem']
            else:
                abs_dev = abs(wip['univ_rem'] - wip['univ_rem'].median())

            # Calc Med's away from median for each and find max datapoint
            # Calculates absolute deviation for each point / total mad.
            # Then we find the max deviation across all points and filter for these -
            # We drop them in the R-idx list and drop from working set.
            ares = abs_dev/data_mad
            R = max(ares)

            # Calculate Probability
            if one_tail:
                p_val = 1 - self.alpha/(n-i+1)
            else:
                p_val = 1 - self.alpha/(2*(n-i+1))

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

            if self.verbose:
                print(f"{i}/{max_outliers} completed.")

        if anom_df.empty:
            anom_df['significance'] = None
            return anom_df

        anom_df = anom_df.iloc[0:num_anoms, :].copy().sort_index(inplace=False)
        anom_df['significance'] = anom_df['univ_rem'] / data['univ_rem'].std()
        return anom_df

    def _detect_anoms(self,
                      data: pd.DataFrame,
                      period: int = None,
                      robust: bool = True,
                      e_value_calc: str = 'orig',
                      data_seasonal: bool = True,
                      **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # Init Variables
        num_obs = data.shape[0]
        max_outliers = floor(num_obs * self.max_anoms)

        # Check kwargs for certain keys:
        weights = kwargs.get('weights', None)
        s_window = kwargs.get('s_window', 'periodic')
        periodic = True if s_window.lower() == 'periodic' else False

        # Check s_window param:
        if s_window and not isinstance(s_window, str) and s_window < 4:
            s_window = 4

        if not period and data_seasonal:
            raise ValueError("Must Supply period length for time series decomposition")

        # Check to make sure we have at least two periods worth of stl_data for anomaly context
        if num_obs < (period * 2) and data_seasonal:
            raise ValueError("Anom detection needs at least 2 periods worth of stl_data")

        if len(list(filter(lambda x: pd.isnull(x), [None] + list(data.iloc[:, 1]) + [None]))) > 3:
            raise ValueError("Data contains non-leading null values. "
                             "Please ensure that all values in the set are non-null.")
        else:
            data.dropna(subset=['value'])

        if max_outliers == 0:
            raise ValueError(f"Not enough observations in this period for AnomalyDetection. "
                             f"There are {num_obs}, and k = {self.max_anoms}; If the maximum number of anomalies that "
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

        # Decompose stl_data into raw, seasonal, trend and remainder;
        # Then attach seasonal, trend, and remainder to orig stl_data and store as data_decomp
        stl = STLDecomposition(data['value'], period=period, periodic=periodic, robust=robust, s_window=s_window).stl()
        data_decomp = pd.concat([data, stl.data[['seasonal', 'trend', 'remainder']]], axis=1)

        # Adjustment if we don't want to include seasonality into the univ_rem
        if not data_seasonal:
            data_decomp['seasonal'] = 0

        data_decomp['e_value'] = calculate_e_value(data_decomp, calc=e_value_calc, weights=weights)
        data_decomp['univ_rem'] = data_decomp['value'] - data_decomp['e_value']

        anoms = self._extract_anomalies(data_decomp[['timestamp', 'value', 'e_value', 'univ_rem']], max_outliers)

        return data_decomp, anoms

    def _filter_anoms_by_threshold(self, anoms: pd.DataFrame, periodic_maxs: pd.DataFrame) -> pd.DataFrame:
        if anoms.empty or not self.threshold or self.threshold == 'None':
            return anoms

        if self.threshold == 'med_max':
            thresh = periodic_maxs['value'].median()
        elif self.threshold.lower() == 'p95':
            thresh = periodic_maxs['value'].quantile(q=0.95)
        elif self.threshold.lower() == 'p99':
            thresh = periodic_maxs['value'].quantile(q=0.99)
        else:
            raise ValueError(f"'{self.threshold}' is unknown, use one of: 'med_max', 'p95', 'p99'")

        return anoms.loc[anoms['value'] >= thresh, :].copy().reset_index(inplace=False)

    def from_dataframe(self,
                       df: pd.DataFrame,
                       longterm: bool = False,
                       periods_in_longterm: int = 2,
                       e_value_calc: str = 'orig',
                       data_seasonal: bool = True):

        if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] < 2 or df.shape[1] > 2:
            raise ValueError("'df' must be a non-empty pandas dataframe with 2 columns: timestamp & value")

        if not isinstance(longterm, bool):
            raise ValueError("'longterm' must be either True or False")

        if periods_in_longterm < 2:
            raise ValueError("'periods_in_longterm' must be greater than 2")

        # Find max pct of anomalies to find
        num_obs = df.shape[0]
        if self.max_anoms < 1/num_obs:
            self.max_anoms = 1/num_obs

        # Normalize format of DF to 2 col df, with first column as 'timestamp' and second column as 'value'
        df, orig_df = self._prepare_dateframe(df)

        # Find Granularity of stl_data.
        gran, period = get_gran_and_period(df['timestamp'])

        # aggregate stl_data to min gran if sec gran
        if gran == 'sec':
            df['timestamp'] = df['timestamp'].dt.floor('min')
            df = df.groupby('timestamp', as_index=False).sum()

        # If dataset is weekly and not enough records
        if gran == 'week' and num_obs < 105:
            period = 4
            if data_seasonal:
                data_seasonal = False
                print("Warning: Will set 'data_seasonal' = False; "
                      "as weekly stl_data contains less than 2 periods (2 years)")
            if longterm:
                longterm = False
                print("Warning: Will set 'longterm' = False; "
                      "as weekly stl_data contains less than 2 periods (2 years)")

        # Setup for longterm time series - breaking the stl_data into subset dataframes and store in all_data
        if longterm:
            all_data = self._convert_to_longterm_array(df, gran, period, periods_in_longterm)
        else:
            all_data = [df]

        all_data_decomp = pd.DataFrame()
        all_anoms = pd.DataFrame()
        for segment in all_data:
            tmp_decomp, tmp_anoms = self._detect_anoms(
                segment, period=period, e_value_calc=e_value_calc, data_seasonal=data_seasonal
            )
            all_data_decomp = all_data_decomp.append(tmp_decomp, ignore_index=True)
            all_anoms = all_anoms.append(tmp_anoms, ignore_index=True)

        # Clean up Potential Duplicates by timestamp
        all_data_decomp.drop_duplicates(subset=['timestamp'], inplace=True)
        all_anoms.drop_duplicates(subset=['timestamp'], inplace=True)

        # Filter Anomalies by threshold
        periodic_maxs = all_data_decomp.groupby(by=all_data_decomp['timestamp'].dt.date, as_index=False)['value'].max()
        all_anoms = self._filter_anoms_by_threshold(all_anoms, periodic_maxs)

        # Label Anomalies - Low or High
        out_anoms = all_anoms[['timestamp', 'value', 'e_value', 'significance']].copy()
        out_anoms['label'] = out_anoms[['value', 'e_value']].apply(lambda x: 'High' if x[0] > x[1] else 'Low', axis=1)

        # Build out_labeled and attach anomaly labels. Where not labeled, set to 'Normal'
        out_labeled = all_data_decomp[['timestamp', 'value', 'e_value']].copy()
        out_labeled = out_labeled.merge(out_anoms[['timestamp', 'label']], how='left', on='timestamp')
        out_labeled.loc[pd.isnull(out_labeled['label']), 'label'] = 'Normal'

        return models.AnomDetect(
            orig=orig_df,
            labeled=out_labeled,
            anoms=out_anoms,
            stl=models.STLDecomp(
                stl_data=all_data_decomp[['value', 'seasonal', 'trend', 'remainder']],
                x_axis=all_data_decomp['timestamp']
            ),
            args=dict(
                max_anoms=self.max_anoms,
                alpha=self.alpha,
                direction=self.direction,
                threshold=self.threshold,
                e_value_calc=e_value_calc,
                data_seasonal=data_seasonal,
                longterm=longterm,
                periods_in_longterm=periods_in_longterm
            )
        )

    def from_vector(self, vec, direction, longterm, periods_in_longterm, e_value_calc, data_seasonal):
        pass


def calculate_e_value(df: pd.DataFrame, calc='orig', weights=None) -> pd.Series:
    median_val = df['value'].median()

    e_value = df['seasonal']
    exp_value_median = e_value + median_val
    exp_value_trend = e_value + df['trend']

    # Calc e_value based on calc string
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
