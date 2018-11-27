import pandas as pd
from math import sqrt, ceil
from typing import List, Tuple


def next_odd(x):
    """ Complete - Helper Function """

    x = round(x)
    if x % 2 == 0:
        x += 1
    return int(x)


def deg_check(x):
    """ Complete - Helper Function """

    if not x:
        return x
    try:
        x = int(x)
        if x not in [0, 1]:
            raise ValueError("Degree's must be either 0 or 1")
        else:
            return x
    except:
        raise ValueError("Degrees must be numeric.")


def stless(y: List, n: int, len: int, ideg: int, njump: int
           , userw: bool, rw: List, ys: List, res: List):

    if n < 2:
        ys[1] = y[1]
        return

    newnj = min(njump, n-1)
    if len >= n:
        nleft = 1
        nright = n

        for i in range(1, (n+1), newnj):
            ys_out, ok = stlest(y,n,len,ideg,i,ys[i],nleft,nright,res,userw,rw)
            ys[i] = ys_out

            if not ok:
                ys[i] = y[i]

    else:
        if newnj == 1:
            nsh = (len+1)//2
            nleft = 1
            nright = len
            for i in range(1, (n+1)):
                if i > nsh and nright != n:
                    nleft += 1
                    nright += 1
                ys_out, ok = stlest(y, n, len, ideg, i, ys[i], nleft, nright, res, userw, rw)
                ys[i] = ys_out
                if not ok:
                    ys[i] = y[i]
        else:
            nsh = (len+1)//2
            for i in range(1, (n+1), newnj):
                if i < nsh:
                    nleft = 1
                    nright = len
                elif i >= n-nsh+1:
                    nleft = n-len+1
                    nright = n
                else:
                    nleft = i-nsh+1
                    nright = len+i-nsh
                ys_out, ok = stlest(y, n, len, ideg, i, ys[i], nleft, nright, res, userw, rw)
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
            ys_out, ok = stlest(y, n, len, ideg, n, ys[n], nleft, nright, res, userw, rw)
            ys[n] = ys_out
            if not ok:
                ys[n] = y[n]

            if k != n-1:
                delta = (ys[n] - ys[k])/n-k
                for j in range(k+1, (n-1)+1):
                    ys[j] = ys[k]+delta*(j-k)

    return ys


def stlest(y: List, n: int, len: int, ideg: int, xs: float, ys: float,
           nleft: int, nright: int, w: List, userw: bool, rw: List):

    rng = float(n) - 1
    h = max(xs - float(nleft), float(nright) - xs)
    if len > n:
        h += float((len-n)//2)

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

            if userw:
                w[j] = rw[j]*w[j]

            a += w[j]
        else:
            w[j] = 0.0

    if a <= 0.0:
        ok = False
    else:
        ok = True

        for j in range(nleft, nright+1):
            w[j] = w[j]/a

        if h > 0.0 and ideg > 0:
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


def stlfts(x: List, n: int, np: int, trend: List, work: List):
    """ Complete - Don't need to return anything, just altering 'trend' & 'work' lists """

    stlma(x, n, np, trend)
    stlma(trend, n - np + 1, np, work)
    stlma(work, (n-(2*np)+2), 3, trend)


def stlma(x: List, n: int, len: int, ave: List):
    """ Complete - Don't need to return anything, just altering 'ave' list """

    newn = n - len + 1
    flen = float(len)
    v = 0.0

    for i in range(1, len+1):
        v = v + x[i]

    ave[1] = v / flen

    if newn > 1:
        k = len
        m = 0
        for j in range(2, newn+1):
            k += 1
            m += 1
            v = v - x[m] + x[k]
            ave[j] = v / flen


def stlstp(y: List, n: int, np: int, ns: int, nt: int, nl: int, isdeg: int, itdeg: int, ildeg: int, nsjump: int,
           ntjump: int, nljump: int, ni: int, userw: bool, rw: List, season: List, trend: List, work: List):

    for j in range(1, ni+1):
        for i in range(1, n+1):
            work[0][i] = y[i]-trend[i]

        stlss(work[0], n, np, ns, isdeg, nsjump, userw, rw, work[1], work[2], work[3], work[4], season)
        stlfts(work[1], (n + (2 * np)), np, work[2], work[0])
        stless(work[2], n, nl, ildeg, nljump, False, work[3], work[0], work[4])

        for i in range(1, n+1):
            season[i] = work[1][np+i] - work[0][i]

        for i in range(1, n+1):
            work[0][i] = y[i] - season[i]

        stless(work[0], n, nt, itdeg, ntjump, userw, rw, trend, work[2])


def stlrwt(y: List, n: int, fit: List, rw: List):
    """ Complete - Don't need to return anything, just altering rw list """

    for i in range(1, n+1):
        rw[i] = abs(y[i]-fit[i])

    mid1, mid2 = psort(rw[1:])
    cmad = 3.0 * (mid1+mid2)
    c9 = 0.999*cmad
    c1 = 0.001*cmad
    for i in range(1, n+1):
        r = abs(y[i] - fit[i])
        if r <= c1:
            rw[i] = 1.0
        elif r <= c9:
            rw[i] = (1.0 - (r/cmad)**2)**2
        else:
            rw[i] = 0.0


def stlss(y: List, n: int, np: int, ns: int, isdeg: int, nsjump: int, userw: bool, rw: List, season: List,
          work1: List, work2: List, work3: List, work4: List):

    if np < 1:
        return

    for j in range(1, np+1):
        k = (n - j)//np + 1
        for i in range(1, k+1):
            work1[i] = y[(i-1)*np+j]

        if userw:
            for i in range(1, k+1):
                work3[i] = rw[(i-1)*np+j]

        ys = stless(work1, k, ns, isdeg, nsjump, userw, work3, work2[2:], work4)
        for i in range(2, len(work2)):
            work2[i] = ([0] + ys + [0])[i]

        xs = 0
        nright = min(ns, k)
        ys, ok = stlest(work1, k, ns, isdeg, xs, work2[1], 1, nright, work4, userw, work3)
        work2[1] = ys

        if not ok:
            work2[1] = work2[2]

        xs = k+1
        nleft = max(1, k-ns+1)
        ys, ok = stlest(work1, k, ns, isdeg, xs, work2[k+2], nleft, k, work4, userw, work3)
        work2[k+2] = ys

        if not ok:
            work2[k+2] = work2[k+1]

        for m in range(1, (k+2)+1):
            season[(m-1)*np+j] = work2[m]


def psort(a: List) -> Tuple:
    """
    Complete, all that needs to be returned are the median(s)
        If there are 2, then return (higher, lower) in tuple.
    """

    n = len(a)
    list_sorted = sorted(a)
    if n < 2:
        raise ValueError("Length of List must be larger than 1.")
    elif n%2 == 0:
        return tuple([list_sorted[int(n/2)], list_sorted[int((n/2)-1)]])
    else:
        return tuple([list_sorted[n//2], list_sorted[n//2]])


def stl(y: List, n: int, np: int, ns: int, nt: int, nl: int, isdeg: int, itdeg: int, ildeg: int,
        nsjump: int, ntjump: int, nljump: int, ni: int, no: int):

    userw = False
    y_to_return = [x for x in y]
    y = [None]+[x for x in y]

    # create working lists
    rw = [0.0 for x in range(n+1)]
    season = [0.0 for x in range(n+1)]
    trend = [0.0 for x in range(n+1)]
    work = [[0.0 for x in range((n+(2*np))+1)] for y in range(5)]

    # S, T, & L spans must be greater than 3 and odd:
    newns = max(3, ns)
    newnt = max(3, nt)
    newnl = max(3, nl)

    for span in [newns, newnt, newnl]:
        if span%2 == 0:
            span += 1

    # Periodicity Must be at least 2
    newnp = max(2, np)

    # Outer Robustness Iterations
    k = 0
    while True:

        stlstp(y, n, newnp, newns, newnt, newnl, isdeg, itdeg, ildeg, nsjump, ntjump,
               nljump, ni, userw, rw, season, trend, work)
        k += 1
        if k > no:
            break

        for i in range(1, n+1):
            work[0][i] = trend[i]+season[i]

        stlrwt(y, n, work[0], rw)
        userw = True

    if no <= 0:
        rw = [None]+[1.0 for x in range(n)]

    final = {
        'original': [float(x) for x in y_to_return],
        'seasonal': [x for x in season[1:]],
        'trend': [x for x in trend[1:]],
        'robust_weights': [x for x in rw[1:]]
    }

    return final


def pySTL(x, period, s_window, s_degree=None, t_window=None, t_degree=None, l_window=None, l_degree=None,
              s_jump=None, t_jump=None, l_jump=None, robust=False, inner=2, outer=0):
    x = [z for z in x]
    n = int(len(x))

    if period < 2 or n <= 2*period:
        raise ValueError("series is not periodic or has less than 2 periods.")

    # Check for 'periodic' s_window value
    periodic = False
    if isinstance(s_window, str):
        if s_window != 'periodic':
            raise ValueError("Unknown string value for s_window")
        else:
            periodic = True
            s_window = 10 * n + 1
            s_degree = 0

    # Check Degrees - Must be either 0 or 1
    s_degree = deg_check(s_degree)
    t_degree = deg_check(t_degree)
    l_degree = deg_check(l_degree)

    # Set Defaults
    s_degree = 0 if not s_degree else s_degree
    t_window = next_odd(ceil(1.5 * period/(1 - 1.5/s_window))) if not t_window else t_window
    t_degree = 1 if not t_degree else t_degree
    l_window = next_odd(period) if not l_window else l_window
    l_degree = t_degree if not l_degree else l_degree
    s_jump = ceil(s_window/10) if not s_jump else s_jump
    t_jump = ceil(t_window/10) if not t_jump else t_jump
    l_jump = ceil(l_window/10) if not l_jump else l_jump
    if robust:
        inner = 1
        outer = 15

    out = stl(x, n, period, s_window, t_window, l_window,
              s_degree, t_degree, l_degree,
              s_jump, t_jump, l_jump,
              inner, outer)

    # TODO: Still need to figure out the final "periodic" piece from the R pacakge.
    """
      if (periodic) {
        which.cycle <- cycle(x)
        z$seasonal <- tapply(z$seasonal, which.cycle, mean)[which.cycle]
      }
      remainder <- as.vector(x) - z$seasonal - z$trend
    """

    final = pd.DataFrame(out, columns=['original', 'seasonal', 'trend', 'robust_weights'])

    return final

