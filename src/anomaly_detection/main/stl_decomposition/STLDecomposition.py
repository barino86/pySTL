from typing import List, Tuple, Union, Iterable

import pandas as pd
from math import sqrt, ceil
from statistics import mean
from itertools import cycle, groupby

from anomaly_detection.main.models import STLDecomp


def next_odd(x) -> int:
    x = round(x)
    x += 1 if x % 2 == 0 else 0
    return int(x)


def degree_check(degree: bool) -> bool:
    if isinstance(degree, bool):
        return degree
    else:
        raise ValueError("Degree arguments must be of boolean type.")


def p_sort(a: List) -> Tuple:
    """
    Complete, all that needs to be returned are the median(s)
        If there are 2, then return (higher, lower) in tuple.
    """

    n = len(a)
    list_sorted = sorted(a)
    if n < 2:
        raise ValueError("Length of List must be larger than 1.")
    elif n % 2 == 0:
        return tuple([list_sorted[int(n/2)], list_sorted[int((n/2)-1)]])
    else:
        return tuple([list_sorted[n//2], list_sorted[n//2]])


class STLDecomposition(object):
    """Decompose a time series into the seasonal, trend and remainder components using loess.
    """

    def __init__(self,
                 series: Iterable[Union[int, float]],
                 period: int,
                 periodic: bool = False,
                 robust: bool = False,
                 **kwargs):
        """
        Additional arguments that can be specified by type are:
            seasonal args:
                s_window (int):   The span (in lags) of the loess window for seasonal extraction, which
                    should be odd and at least 7, according to Cleveland et al.
                s_degree:   (bool)
                s_jump:

            trend args:
                t_window:
                t_degree:
                t_jump:

            loess args:
                l_window:
                l_degree:
                l_jump:

            robustness args:
                inner:
                outer:
        """
        self.series: List[float] = [float(x) for x in series]
        self.n: int = int(len(self.series))
        self.period: int = period
        self.periodic: bool = periodic
        self.robust: bool = robust

        # Ensure periodic and that there 2 or more periods of stl_data.
        if self.period < 2 or self.n <= (2 * self.period):
            raise ValueError("'series' is not periodic or has less than 2 periods.")

        # Check Windows
        self._s_window: int = (10 * self.n + 1) if self.periodic else kwargs.get('s_window')
        self._t_window: int = kwargs.get('t_window', next_odd(ceil(1.5 * self.period/(1 - 1.5/self._s_window))))
        self._l_window: int = kwargs.get('l_window', next_odd(self.period))

        if not self._s_window:
            raise ValueError("If periodic is not specified then the 's_window' arg must be specified.")

        self._check_windows()

        # Check Degrees
        self._s_degree: bool = degree_check(False if self.periodic else kwargs.get('s_degree', False))
        self._t_degree: bool = degree_check(kwargs.get('t_degree', True))
        self._l_degree: bool = degree_check(kwargs.get('l_degree', self._t_degree))

        # Check Jumps
        self._s_jump: int = kwargs.get('s_jump', ceil(self._s_window/10))
        self._t_jump: int = kwargs.get('t_jump', ceil(self._t_window/10))
        self._l_jump: int = kwargs.get('l_jump', ceil(self._l_window/10))

        # Set Inner and Outer
        self.inner: int = kwargs.get('inner', (1 if robust else 2))
        self.outer: int = kwargs.get('outer', (15 if robust else 0))

    def _check_windows(self):
        window_attrs = ['_s_window', '_t_window', '_l_window']
        for attr in window_attrs:
            setattr(self, attr, next_odd(max(3, getattr(self, attr))))

    def stl(self):
        # create working lists
        robust_weights = [0.0 for _ in range(self.n + 1)]
        season = [0.0 for _ in range(self.n + 1)]
        trend = [0.0 for _ in range(self.n + 1)]
        work = [[0.0 for _ in range((self.n + (2 * self.period)) + 1)] for _ in range(5)]

        # Outer Robustness Iterations
        k = 0
        use_robust_weights = False
        while True:
            self.stl_stp(use_robust_weights, robust_weights, season, trend, work)
            k += 1
            if k > self.outer:
                break

            for i in range(1, self.n+1):
                work[0][i] = trend[i]+season[i]

            self.stl_rwt(work[0], robust_weights)
            use_robust_weights = True

        if self.outer <= 0:
            robust_weights = [1.0 for _ in range(self.n + 1)]

        out = {
            'value': self.series,
            'seasonal': [x for x in season[1:]],
            'trend': [x for x in trend[1:]]
        }

        if self.periodic:
            which_cycle = list(zip(cycle(range(self.period)), out['seasonal']))
            agg = groupby(sorted(which_cycle, key=lambda z: z[0]), key=lambda y: y[0])
            agg_dict = {k: mean(x for _, x in group) for k, group in agg}
            out['seasonal'] = [agg_dict[x[0]] for x in which_cycle]

        final = pd.DataFrame(out, columns=['value', 'seasonal', 'trend'])
        final['remainder'] = final['value'] - final['seasonal'] - final['trend']

        return STLDecomp(stl_data=final)

    def stl_stp(self, use_robust_weights: bool, robust_weights: List, season: List, trend: List, work: List):

        for j in range(1, self.inner + 1):
            for i in range(1, self.n+1):
                work[0][i] = self.series[i-1] - trend[i]

            self.stl_ss(work[0], use_robust_weights, robust_weights, work[1], work[2], work[3], work[4], season)
            self.stl_fts(work[1], (self.n + (2 * self.period)), work[2], work[0])
            self.stl_ess(work[2], self.n, self._l_window, self._l_degree, self._l_jump, False, work[3], work[0], work[4])

            for i in range(1, self.n+1):
                season[i] = work[1][self.period + i] - work[0][i]

            for i in range(1, self.n+1):
                work[0][i] = self.series[i-1] - season[i]

            self.stl_ess(work[0], self.n, self._t_window, self._t_degree, self._t_jump,
                         use_robust_weights, robust_weights, trend, work[2])

    def stl_rwt(self, fit: List, robust_weights: List):
        """ Complete - Don't need to return anything, just altering robust_weights list """

        for i in range(1, self.n+1):
            robust_weights[i] = abs(self.series[i-1] - fit[i])

        mid1, mid2 = p_sort(robust_weights[1:])
        cmad = 3.0 * (mid1 + mid2)
        c9 = 0.999*cmad
        c1 = 0.001*cmad
        for i in range(1, self.n+1):
            r = abs(self.series[i-1] - fit[i])
            if r <= c1:
                robust_weights[i] = 1.0
            elif r <= c9:
                robust_weights[i] = (1.0 - (r / cmad) ** 2) ** 2
            else:
                robust_weights[i] = 0.0

    def stl_fts(self, x: List, n: int, trend: List, work: List):
        """ Complete - Don't need to return anything, just altering 'trend' & 'work' lists """

        self.stl_ma(x, n, self.period, trend)
        self.stl_ma(trend, n - self.period + 1, self.period, work)
        self.stl_ma(work, (n - (2 * self.period) + 2), 3, trend)

    @staticmethod
    def stl_ma(x: List, n: int, length: int, ave: List) -> None:
        new_n = n - length + 1
        f_len = float(length)
        v = 0.0

        for i in range(1, (length + 1)):
            v = v + x[i]

        ave[1] = v / f_len
        if new_n > 1:
            k = length
            m = 0
            for j in range(2, new_n+1):
                k += 1
                m += 1
                v = v - x[m] + x[k]
                ave[j] = v / f_len

    def stl_ss(self, y: List, use_robust_weights: bool, robust_weights: List,
               season: List, work1: List, work2: List, work3: List, work4: List):
        if self.period < 1:
            return

        for j in range(1, self.period + 1):
            k = (self.n - j) // self.period + 1
            for i in range(1, (k + 1)):
                work1[i] = y[(i - 1) * self.period + j]

            if use_robust_weights:
                for i in range(1, (k + 1)):
                    work3[i] = robust_weights[(i - 1) * self.period + j]

            ys = self.stl_ess(work1, k, self._s_window, self._s_degree, self._s_jump, use_robust_weights, work3, work2[2:], work4)
            for i in range(2, len(work2)):
                work2[i] = ([0] + ys + [0])[i]

            xs = 0
            nright = min(self._s_window, k)
            ys, ok = self.stl_est(work1, k, self._s_window, self._s_degree, xs, work2[1], 1, nright, work4, use_robust_weights, work3)
            work2[1] = ys

            if not ok:
                work2[1] = work2[2]

            xs = k+1
            nleft = max(1, k - self._s_window + 1)
            ys, ok = self.stl_est(work1, k, self._s_window, self._s_degree, xs, work2[k + 2], nleft, k, work4, use_robust_weights, work3)
            work2[k+2] = ys

            if not ok:
                work2[k+2] = work2[k+1]

            for m in range(1, (k+2)+1):
                season[(m-1) * self.period + j] = work2[m]

    def stl_ess(self, y: List, n: int, window: int, degree: bool, jump: int,
                use_robust_weights: bool, robust_weights: List, ys: List, res: List):
        if n < 2:
            ys[1] = y[1]
            return

        newnj = min(jump, n - 1)
        if window >= n:
            nleft = 1
            nright = n

            for i in range(1, (n+1), newnj):
                ys_out, ok = self.stl_est(y, n, window, degree, i, ys[i], nleft, nright, res, use_robust_weights, robust_weights)
                ys[i] = ys_out

                if not ok:
                    ys[i] = y[i]

        else:
            if newnj == 1:
                nsh = (window + 1) // 2
                nleft = 1
                nright = window
                for i in range(1, (n+1)):
                    if i > nsh and nright != n:
                        nleft += 1
                        nright += 1
                    ys_out, ok = self.stl_est(y, n, window, degree, i, ys[i], nleft, nright, res, use_robust_weights, robust_weights)
                    ys[i] = ys_out
                    if not ok:
                        ys[i] = y[i]
            else:
                nsh = (window + 1) // 2
                for i in range(1, (n+1), newnj):
                    if i < nsh:
                        nleft = 1
                        nright = window
                    elif i >= n-nsh+1:
                        nleft = n - window + 1
                        nright = n
                    else:
                        nleft = i-nsh+1
                        nright = window + i - nsh
                    ys_out, ok = self.stl_est(y, n, window, degree, i, ys[i], nleft, nright, res, use_robust_weights, robust_weights)
                    ys[i] = ys_out
                    if not ok:
                        ys[i] = y[i]

        if newnj != 1:
            for i in range(1, (n-newnj+1), newnj):
                delta = (ys[i+newnj] - ys[i]) / newnj
                for j in range(i+1, (i+newnj-1)+1):
                    ys[j] = ys[i]+delta*(j-i)

            k = ((n-1)//newnj)*newnj+1

            if k != n:
                ys_out, ok = self.stl_est(y, n, window, degree, n, ys[n], nleft, nright, res, use_robust_weights, robust_weights)
                ys[n] = ys_out
                if not ok:
                    ys[n] = y[n]

                if k != n-1:
                    delta = (ys[n] - ys[k])/n-k
                    for j in range(k+1, (n-1)+1):
                        ys[j] = ys[k]+delta*(j-k)

        return ys

    @staticmethod
    def stl_est(y: List, n: int, window: int, degree: bool, xs: float, ys: float,
                nleft: int, nright: int, w: List, use_robust_weights: bool, robust_weights: List):
        rng = float(n) - 1
        h = max(xs - float(nleft), float(nright) - xs)
        if window > n:
            h += float((window - n) // 2)

        h9 = 0.999*h
        h1 = 0.001*h
        a = 0.0

        for j in range(nleft, nright+1):
            r = abs(j-xs)
            if r <= h9:
                if r <= h1:
                    w[j] = 1.0
                else:
                    w[j] = (1.0 - (r/h)**3)**3

                if use_robust_weights:
                    w[j] = robust_weights[j] * w[j]

                a += w[j]
            else:
                w[j] = 0.0

        if a <= 0.0:
            ok = False
        else:
            ok = True

            for j in range(nleft, nright+1):
                w[j] = w[j]/a

            if h > 0.0 and degree:
                a = 0.0
                for j in range(nleft, nright+1):
                    a += w[j] * float(j)

                b = xs - a
                c = 0.0
                for j in range(nleft, nright+1):
                    c += w[j]*(float(j)-a)**2

                if sqrt(c) > 0.001*rng:
                    b = b/c
                    for j in range(nleft, nright+1):
                        w[j] = w[j]*(b*(float(j)-a)+1.0)

            ys = 0.0
            for j in range(nleft, nright+1):
                ys += w[j]*y[j]

        return ys, ok
