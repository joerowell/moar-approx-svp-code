# -*- coding: utf-8 -*-
"""
Utility functions for simulations etc
"""

import csv
import logging
import os
import sys
from collections import OrderedDict

from functools import wraps, update_wrapper


def read_csv(filename, columns, read_range=None, ytransform=lambda y: y):
    """
    Read CSV data and return two tuples, one for the x and one for the y coordinate.

    :param filename: csv file name
    :param columns: columns to read
    :param read_range: read data when first column is in this range
    :param ytransform: function to call on second column

    """
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = []
        for i, row in enumerate(reader):
            if not row:
                continue
            if i == 0:
                columns = row.index(columns[0]), row.index(columns[1])
                continue
            data.append((int(row[columns[0]]), ytransform(float(row[columns[1]]))))

    if read_range is not None:
        data = [(x, y) for x, y in data if x in read_range]
    data = sorted(data)
    logging.debug(data)
    X = [x for x, y in data]
    Y = [y for x, y in data]
    return tuple(X), tuple(Y)


def memoize(function):
    """
    Add caching to ``function``.
    """
    memo = {}

    @wraps(function)
    def wrapper(*args):
        try:
            return memo[args]
        except KeyError:
            rv = function(*args)
            memo[args] = rv
            return rv

    wrapper = update_wrapper(wrapper, function)

    return wrapper


class SuppressStream(object):
    """
    Prevent the output stream from being shown.
    """

    def __init__(self, stream=sys.stderr):
        self.orig_stream_fileno = stream.fileno()

    def __enter__(self):
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, "w")
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()


def seq(start, stop, step, prec=2):
    """
    Iterate from ``start`` to ``stop`` in increments of ``step``.

    .. note :: this supports floating point values

    """
    curr = start
    while curr < stop:
        yield float(curr)
        curr = round(10.0 ** prec * (curr + step)) / 10.0 ** prec


def mkdict(seq, init=None):
    """
    Make an ``OrderedDict`` with the entries of ``seq`` as keys

    :param seq:
    :param init:

    """
    d = OrderedDict()
    for e in seq:
        if isinstance(init, str):
            d[e] = eval(init)()
        else:
            d[e] = init
    return d


def lghf(d):
    """
    `log(GH(d))`

    :param d:

    """
    from scipy.special import loggamma
    from math import sqrt, pi, log

    d = float(d)
    return float(loggamma(d / 2 + 1) / d - log(sqrt(pi)))


def ghf(d):
    """
    `GH(d)`

    :param d:

    """
    from math import exp

    return exp(lghf(d))


def rhff(k, alpha):
    """
    `α⋅GH(k)^{1/(k-1)}`

    :param k:
    :param alpha:

    """
    from math import log, exp

    small = (
        (0, 0),
        (1, 0),
        (2, -0.08841287960306055),
        (3, -0.07544412913470162),
        (4, -0.07544412913470162),
        (5, -0.07544412913470162),
        (6, -0.07048974749276288),
        (7, -0.06694511236665464),
        (8, -0.06693551836216481),
        (9, -0.06514479158624617),
        (10, -0.06504048330773132),
        (11, -0.06300742595211455),
        (12, -0.06292516413568243),
        (13, -0.06144932038223158),
        (14, -0.061335194884657974),
        (15, -0.05979787485927308),
        (16, -0.059698538727778944),
        (17, -0.05854990651097318),
        (18, -0.05842140906014142),
        (19, -0.05743190747788606),
        (20, -0.057204890112602075),
        (21, -0.05625926852349446),
        (22, -0.0561373418825689),
        (23, -0.05513018652148212),
        (24, -0.055035755076112546),
        (25, -0.05444317674587422),
        (26, -0.054169276891314226),
        (27, -0.05393347149410525),
        (28, -0.05377470641416448),
        (29, -0.05373663760840141),
        (30, -0.05367325837865427),
        (31, -0.05368859276991003),
        (32, -0.053211946661946876),
        (33, -0.05289686076369957),
        (34, -0.05256983259115176),
        (35, -0.05208132942258301),
        (36, -0.051994765423887495),
        (37, -0.05088847622411832),
        (38, -0.05074262422310026),
        (39, -0.050366640049666535),
        (40, -0.05005110125528332),
        (41, -0.05273129943387767),
        (42, -0.052517722118313626),
        (43, -0.05205856884875133),
        (44, -0.051824674774980664),
        (45, -0.05141429845211216),
        (46, -0.05118951330085285),
        (47, -0.050749690271035014),
        (48, -0.05048220060429904),
        (49, -0.050113727094377634),
        (50, -0.04986388279651427),
    )

    if k <= 2:
        return 1.0219
    elif k <= 40:
        return alpha * exp(-small[k][1] / 4)
    else:
        return exp((log(alpha) + lghf(k)) / (k - 1))


@memoize
def k_alphaf(k, alpha):
    """
    Return k_α.
    """
    rhf = rhff(k, 1.0)
    for k_alpha in range(k, 100 * k, 1):
        if rhff(k_alpha, alpha) <= rhf:
            return k_alpha
    raise ValueError("k_α failed for {k},{alpha:.2f}".format(k=k, alpha=alpha))


def plot_bkz(d, k, alpha=1.00, c=0.00, max_loops=4):
    """

    :param d:
    :param k:
    :param alpha:
    :param c:
    :param max_loops:

    :returns: plots of block_sizes and log of Gram-Schmidt norms

    """
    from sage.all import line
    from fpylll import BKZ
    from fpylll.tools.bkz_simulator import simulate as fplll_simulate
    from simu import SConf, bootstrap_strategies, Strategy, sample_r
    from simu import BKZQualitySimulation, ProcrastinatingBKZQualitySimulation
    from math import log

    r = sample_r(500)

    strategies = bootstrap_strategies(SConf.k_low, c=c)
    for k_ in range(SConf.k_low, k + 1):
        strategies.append(Strategy(k_, [k_ - 20], [], alpha=alpha, c=c))

    if c == 0.0:
        cls = BKZQualitySimulation
    elif c == 0.25:
        cls = ProcrastinatingBKZQualitySimulation
    else:
        raise ValueError

    params = BKZ.EasyParam(k, strategies=strategies, max_loops=max_loops)
    r_fplll = fplll_simulate(list(r), params)[0]
    bkz = cls(list(r))
    r_bkz = bkz(params)

    shapes = line([(i, log(r_, 2) / 2) for i, r_ in enumerate(r_fplll)], color="green", legend_label="FPLLL") + line(
        [(i, log(r_, 2) / 2) for i, r_ in enumerate(r_bkz)], color="blue", legend_label="This work"
    )

    block_sizes = bkz.trace.child(("tour", 0)).data["block_sizes"]
    block_sizes = line(([i, b] for i, b in enumerate(block_sizes)), color="green")

    return block_sizes, shapes
