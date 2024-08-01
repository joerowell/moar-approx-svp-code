#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line interfaces to Python code.
"""

import csv
import glob
import logging
import os
import pickle
import re
from collections import OrderedDict

import begin
from fpylll.tools.quality import get_current_slope

from bkz import BKZ, BKZQualitySimulation, BKZSimulationTreeTracer, ProcrastinatingBKZQualitySimulation
from conv import cost_simulation as conv_cost_simulation
from math import ceil, e, log, pi, exp
from simu import SConf, Strategy, bootstrap_strategies, sample_r, simulate as simulate_cost
from utils import lghf, read_csv, seq

@begin.subcommand
@begin.convert(low_index=int, high_index=int)
def lll_cost_fit( #noqa
        filename: "filename to fit",
        low_index: "start fitting at this index" = 50,
        high_index: "stop fitting at this index(exclusive)" = 250,
        columns: "csv columns to select" = ("rank", "average"),
        dominant_coefficient: "leading coefficient of expression" = None,
):
    """ 
    Fit the cost for the LLL runs 
    Please note that, similarly to the enumeration cost, we are aiming to fit coefficients here,
    rather than a polynomial.
    """

    ## We only accept files produced by process.py: this means that we must have a .csv
    ## extension
    if filename.endswith(".csv") == False:
        print("Error: lll_cost_fit only accepts a file extension of .csv", file=sys.stderr)
        exit(1)


    import numpy as np
    from math import log
    from scipy.optimize import curve_fit

    if isinstance(columns, str):
        # I don't understand regex syntax, so this mirrors the code in "cost_fit"
        columns = re.match("(.+),(.+)", columns).groups()
    X, Y = read_csv(filename, columns=columns, read_range=range(low_index, high_index))

    if dominant_coefficient is None:
        # This is also just meant to test if our cubic fit is sound:
        # you might be able to find a nicer fit using more variables.
        def f(x, a, b):
            return a * np.log2(x) + b
        
    r = list(curve_fit(f, X, Y)[0])
    if dominant_coefficient is not None:
        r = [dominant_coefficient] + r
        
    print("{r0:.4f}*log2(x) + {r1:.4f}".format(fn=os.path.basename(filename), r0=r[0], r1=r[1]))
        
def cost_fit(  # noqa
    filename: "filename to fit",
    low_index: "start fitting at this index" = 100,
    high_index: "stop fitting at this index (exclusive)" = 250,
    columns: "csv columns to select" = ("k alpha", "total cost"),
    dominant_coefficient: "leading coefficient of expression" = None,
):
    """
    Fit cost
    """

    if filename.endswith(".sobj"):
        filenames_maps = conv_cost_simulation(filename)
        ret = OrderedDict()
        for alpha, filename in filenames_maps.items():
            logging.info("# α: {alpha:.2f}".format(alpha=alpha))
            ret[alpha] = cost_fit(
                filename,
                low_index=low_index,
                high_index=high_index,
                columns=columns,
                dominant_coefficient=dominant_coefficient,
            )

        return ret

    import numpy as np
    from math import log
    from scipy.optimize import curve_fit

    if isinstance(columns, str):
        columns = re.match("(.+),(.+)", columns).groups()

    X, Y = read_csv(
        filename, columns=columns, read_range=range(low_index, high_index), ytransform=lambda y: log(y, 2.0)
    )

    if dominant_coefficient is None:

        def f(x, a, b, c):
            return a * x * np.log2(x) + b * x + c

    else:

        def f(x, b, c):
            return dominant_coefficient * x * np.log2(x) + b * x + c

    r = list(curve_fit(f, X, Y)[0])
    if dominant_coefficient is not None:
        r = [dominant_coefficient] + r
    logging.info(
        "{r0:.4f}*x*log(x,2) - {r1:5.3f}*x + {r2:5.2f}".format(
            fn=os.path.basename(filename), r0=r[0], r1=-r[1], r2=r[2]
        )
    )
    return r


def tex_cost_fit(LC):
    from utils import rhff

    for c in (0.00, 0.15, 0.25):
        cls = "c%03d" % ceil(100 * c)
        denominator = "8" if c > 0.00 else "2\\me"
        cost0 = "0.1250" if c > 0.00 else "0.1839"
        baseline_cost = {}

        for filename in sorted(glob.glob("../data/approx-hsvp-simulations,qary,*,{c:.2f},*.csv".format(c=c))):
            alpha, _, preproc, tag = re.match(".*,qary,([^,]+),([^,]+),([^,]+)(.*)\\.csv", filename).groups()
            tag = tag.replace(",", "/")

            if preproc != "best":
                preproc = ceil(100 * float(preproc))

            if alpha != "best":
                alpha = float(alpha)
                low_index = ceil(alpha * 100)
                high_index = ceil(alpha * 250)
                columns = ("k alpha", "total cost")
            else:
                low_index = 200
                high_index = 350
                columns = ("k", "total cost")

            cost = cost_fit(
                filename,
                low_index=low_index,
                high_index=high_index,
                columns=columns,
                dominant_coefficient=0.1250 if c > 0.00 else 0.1839,
            )

            if alpha == 1.0:
                baseline_cost[(preproc, tag)] = cost

            if alpha == "best":
                LC[
                    "/formulas/{cls}/p{preproc}{tag}/all".format(cls=cls, preproc=preproc, tag=tag)
                ] = "\\frac{{k \\log k}}{{{denominator}}} - {cost1:.3f}\\,k + {cost2:.2f}".format(
                    cost1=-cost[1], cost2=cost[2], denominator=denominator
                )
                LC[
                    "/formulas/{cls}/p{preproc}{tag}/speedup/all".format(cls=cls, preproc=preproc, tag=tag)
                ] = "{cost1:.3f}\\,k{sign}{cost2:.2f}".format(
                    sign="" if (baseline_cost[(preproc, tag)][2] - cost[2] < 0) else "+",
                    cost1=baseline_cost[(preproc, tag)][1] - cost[1],
                    cost2=baseline_cost[(preproc, tag)][2] - cost[2],
                )
                LC[
                    "/functions/{cls}/p{preproc}{tag}/all".format(cls=cls, preproc=preproc, tag=tag)
                ] = "{cost0}*x*log2(x) - {cost1:.3f}*x + {cost2:.2f}".format(cost1=-cost[1], cost2=cost[2], cost0=cost0)
                sieving_crossover = 2
                for k in range(50, 1000):
                    if (0.5 * (float(cost0) * k * log(k, 2.0) + round(cost[1], 3) * k + round(cost[2], 2))) < 0.265 * k:
                        sieving_crossover = k + 1
                LC[
                    "/crossovers/{cls}/p{preproc}{tag}/sieving".format(cls=cls, preproc=preproc, tag=tag)
                ] = sieving_crossover

                LC["/crossovers/{cls}/p{preproc}{tag}/sieving/m1".format(cls=cls, preproc=preproc, tag=tag)] = (
                    sieving_crossover - 1
                )

                LC[
                    "/crossovers/{cls}/p{preproc}{tag}/sieving/rhf".format(cls=cls, preproc=preproc, tag=tag)
                ] = "%.4f" % rhff(sieving_crossover, 1.0)

            else:
                LC[
                    "/formulas/{cls}/p{preproc}{tag}/alpha{alpha:3d}".format(
                        alpha=round(100 * alpha), cls=cls, preproc=preproc, tag=tag
                    )
                ] = "\\frac{{k \\log k}}{{{denominator}}} - {cost1:.3f}\\,k + {cost2:.2f}".format(
                    cost1=-cost[1], cost2=cost[2], denominator=denominator
                )
                LC[
                    "/formulas/{cls}/p{preproc}{tag}/speedup/alpha{alpha:3d}".format(
                        alpha=round(100 * alpha), cls=cls, preproc=preproc, tag=tag
                    )
                ] = "{cost1:.3f}\\,k{sign}{cost2:.2f}".format(
                    sign="" if (baseline_cost[(preproc, tag)][2] - cost[2] < 0) else "+",
                    cost1=baseline_cost[(preproc, tag)][1] - cost[1],
                    cost2=baseline_cost[(preproc, tag)][2] - cost[2],
                )
                LC[
                    "/functions/{cls}/p{preproc}{tag}/alpha{alpha:3d}".format(
                        alpha=round(100 * alpha), cls=cls, preproc=preproc, tag=tag
                    )
                ] = "{cost0}*x*log2(x) - {cost1:.3f}*x + {cost2:.2f}".format(cost1=-cost[1], cost2=cost[2], cost0=cost0)

    return LC


@begin.subcommand
@begin.convert(host_name=str)
def tex(host_name=None):
    """
    Produce data in format that TeX can consume.
    :param host_name name of computer used to generate strategies. If not None, we remove the host name from the
    output filename.
    """

    """
    Add the dash at the front for nicer filenames.
    """
    if host_name is not None:
        if host_name[0] != "-":
            host_name="-"+host_name

    from conv import cost_simulation, call_sobj_csv, verify_csv

    for filename in glob.glob("../data/approx-hsvp-simulations,qary,*-*,*,*.sobj"):
        if "-strategies.sobj" in filename:
            continue
        cost_simulation(filename)

    for filename in glob.glob("../data/approx-hsvp-fplll-observations,*,*.*.sobj"):
        call_sobj_csv(filename)

    for filename in glob.glob("../data/verify-*-*-*-*.sobj"):
        verify_csv(filename)

    LC = dict()
    LC = tex_cost_fit(LC)

    with open("../paper/constants.tex", "w") as fh:
        fh.write("\\pgfkeys{\n")
        for key in sorted(LC):
            fh.write("  {key}/.initial={value},\n".format(key=key, value=LC[key]))
        fh.write("}\n")
        fh.write("\n")
        for filename in glob.glob("../data/approx-hsvp-simulations,qary*.csv"):
            fh.write(
                (
                    "\\embeddedfile{{{basename}}}%\n" "             [{basename}]%\n" "             {{{filename}}}\n"
                ).format(filename=filename, basename=os.path.split(filename)[-1])
            )
        for filename in glob.glob("../data/approx-hsvp-fplll-observations,1.*.csv"):
            # Tex seems to get annoyed at the inner square brackets: remove them.
            # To do this, we replace the square brackets: but for the sake of neatness, we also remove
            # anything beyond the json part. This also means we need to add the .csv extension back at the end.
            basename=os.path.split(filename)[-1]
            basename, endpart=basename.split(".json")

            # Because we have multiple files with the same prefix, we also care about which simulation files
            # they were generated from. So, we include this too, but that also means we need to tidy it up
            endpart=endpart.split("-simulations")[1]

            # Now we remove any square brackets. To avoid a double '.csv' string, we remove it from endpart
            endpart=endpart.replace(".csv]","")
            basename=basename.replace("[","")
            basename=basename.replace("]","")
            # Remove the hostname, if set
            # We don't really need this, but
            if host_name is not None:
                basename=basename.replace(host_name,"")

            # Form the final filename for the supplementary material
            basename=basename+endpart

            fh.write(
                (
                    "\\embeddedfile{{{basename}}}%\n" "             [{basename}]%\n" "             {{{filename}}}\n"
                ).format(filename=filename, basename=basename)
            )

        for filename in glob.glob("../data/verify-*.csv"):
            fh.write(
                (
                    "\\embeddedfile{{{basename}}}%\n" "             [{basename}]%\n" "             {{{filename}}}\n"
                ).format(filename=filename, basename=os.path.split(filename)[-1])
            )
def verify_kernel(args):
    from copy import copy
    from bkz import BKZReduction, ProcrastinatingBKZReduction
    from fpylll import FPLLL, IntegerMatrix, LLL

    d, seed, params = args

    FPLLL.set_random_seed(seed)
    A = LLL.reduction(IntegerMatrix.random(d, "qary", k=d // 2, q=16777259))
    if params["c"] == 0:
        bkz = BKZReduction(copy(A))
    else:
        bkz = ProcrastinatingBKZReduction(copy(A))
    bkz(params)
    return bkz.M.r(), bkz.trace


@begin.subcommand
@begin.convert(d=int, k=int, c=float, jobs=int, alpha=float, tours=int, seed=int, impl=bool)
def verify(d, k, filename, alpha=1.10, c=0.00, jobs=1, tours=8, seed=0, impl=True):
    """

    :param d: lattice dimension
    :param k: block size
    :param filename: dump data to this filename
    :param alpha: α ≥ 1.0
    :param c: c ≥ 0
    :param jobs: number of jobs to run in parallel
    :param tours: number of BKZ tours
    :param seed: randomness seed
    :param impl: run implementation

    """
    from fpylll import FPLLL, IntegerMatrix, GSO, LLL
    from bkz import BKZQualitySimulation, BKZSimulation
    from bkz import ProcrastinatingBKZQualitySimulation, ProcrastinatingBKZSimulation
    from simu import SConf, simulate
    import pickle
    from multiprocessing import Pool

    if c:
        QualitySimulation = ProcrastinatingBKZQualitySimulation
        FullSimulation = ProcrastinatingBKZSimulation
    else:
        QualitySimulation = BKZQualitySimulation
        FullSimulation = BKZSimulation

    SConf.k_low = 20
    SConf.step_size = 1
    costs, strategies = simulate(k_high=k, alphas=[alpha], c=c, preproc_strategy=alpha, dump_filename=False, jobs=jobs)

    FPLLL.set_random_seed(seed)
    A = LLL.reduction(IntegerMatrix.random(d, "qary", k=d // 2, q=16777259))
    M = GSO.Mat(A)
    M.update_gso()
    r = list(M.r())

    params = BKZ.EasyParam(k, strategies=strategies, max_loops=tours, flags=BKZ.VERBOSE)
    params["c"] = c
    print("# Rough Simulation: ")
    bkz_qual = QualitySimulation(list(r))
    r_qual = bkz_qual(params)
    # print(bkz_qual.trace.report())
    print()
    print("# Full Simulation: ")
    bkz_full = FullSimulation(list(r))
    r_full = bkz_full(params)
    print()
    print("# Implementation: ")

    if impl:
        tasks = []
        for i in range(jobs):
            tasks.append((d, seed + i, params))
        if jobs > 1:
            pool = Pool(jobs)
            results = pool.map(verify_kernel, tasks)
        else:
            results = map(verify_kernel, tasks)

        r_impl = [0.0] * d
        traces = []
        for result in results:
            for i in range(d):
                r_impl[i] += result[0][i] / jobs
            traces.append(result[1])
    else:
        r_impl = None
        traces = None

    print()
    print("# Rough v Full")
    for i in range(len(r)):
        if i % 10 == 0:
            print()
        print("(%3d, %5.2f)" % (i, log(r_qual[i], 2) / 2 - log(r_full[i], 2) / 2), end=" ")
    print()
    if impl:
        print()
        print("# Rough v Implementation ")
        for i in range(len(r)):
            if i % 10 == 0:
                print()
            print("(%3d, %5.2f)" % (i, log(r_qual[i], 2) / 2 - log(r_impl[i], 2) / 2), end=" ")
        print()
        print()

    ret = {
        "r_qual": r_qual,
        "r_full": r_full,
        "r_impl": r_impl,
        "trace_qual": bkz_qual.trace,
        "trace_full": bkz_full.trace,
        "trace_impl": traces,
        "costs": costs,
        "strategies": strategies,
    }

    pickle.dump(ret, open(filename, "wb"))
    return ret


def set_sconf_values(**kwds):
    for k, v in kwds.items():
        setattr(SConf, k, eval(v))


@begin.subcommand  # noqa
@begin.convert(k_high=int, c=float, greedy=bool, jobs=int, rho=float)
def cost_simulation(
    k_high: "compute up to ``RHF = GH(k_high)^(1/(k_high-1))`` (inclusive).",
    alpha_spec: "a tuple of the form α_low:α_high:α_incr (e.g. 1.00:2.00:0.05) to set those three variables",
    c: "overshooting parameter" = 0.00,
    preproc_strategy: 'either some `α ∈ α_{spec}` or "best"' = None,
    dump_filename: """results are regularly written to this filename, if ``None``
               then a filename will be chosen.""" = None,
    jobs: "number of cores to use in parallel" = 1,
    greedy: "use Greedy pruning strategy" = False,
    rho: "Fix ρ as in [LN20, Theorem 3]" = None,
    sconf: "Adjust ``SConf`` values." = "",
):
    """
    Estimate cost of enumeration using `α-GH(k_α)^(1/(k_α-1))` oracles.
    """

    if ":" not in alpha_spec:
        alpha_low = float(alpha_spec)
        alpha_high = float(alpha_spec)
        alpha_incr = 0.2
        if preproc_strategy is not None and preproc_strategy != alpha_low:
            raise ValueError(
                ("preproc_strategy={preproc_stategy} is incompatible with alpha_spec={alpha_spec}").format(
                    preproc_strategy=preproc_strategy, alpha_spec=alpha_spec
                )
            )
        preproc_strategy = alpha_low
    else:
        alpha_low, alpha_high, alpha_incr = tuple(map(float, alpha_spec.split(":")))

    alpha_high = (
        alpha_high + alpha_incr / 2
    )  # we want α_high included as an endpoint, so we increase the bound slightly

    alphas = seq(alpha_low, alpha_high, alpha_incr)

    if sconf:
        set_sconf_values(**dict([kv.split("=") for kv in sconf.split(",")]))

    if preproc_strategy is None:
        preproc_strategy = "best"

    if preproc_strategy != "best":
        preproc_strategy = float(preproc_strategy)

    simulate_cost(
        k_high=k_high,
        alphas=alphas,
        c=c,
        preproc_strategy=preproc_strategy,
        dump_filename=dump_filename,
        jobs=jobs,
        greedy=greedy,
        rho=rho,
    )


@begin.subcommand  # noqa
@begin.convert(d=int, k=int, alpha=float, c=float, tours=int)
def quality_simulation(
    d: "lattice dimension d≥2",
    k: "BKZ parameter 2≤k≤d",
    alpha: "relaxation parameter α≥1",
    c: "overshooting parameter c≥0",
    tours: "number of tours > 0" = 4,
):
    r = sample_r(d)
    strategies = bootstrap_strategies(SConf.k_low, c=c)
    for k_ in range(SConf.k_low, k + 1):
        if k_ >= 60:
            preproc = [k_ - 20] * tours
        else:
            preproc = []
        strategies.append(Strategy(k_, preproc, [], alpha=alpha, c=c))
    params = BKZ.EasyParam(k, strategies=strategies, max_loops=tours)
    if c == 0.0:
        bkz = BKZQualitySimulation(r)
    else:
        bkz = ProcrastinatingBKZQualitySimulation(r)
    tracer = BKZSimulationTreeTracer(bkz, verbosity=1)
    r = bkz(params, tracer=tracer)

    fn = "../data/bkz-{k}-shape,{d},{alpha:.2f},{c:.2f},{tours}.csv".format(k=k, d=d, alpha=alpha, c=c, tours=tours)
    with open(fn, "w") as fh:
        csvwriter = csv.writer(fh, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(("i", "log(r_i)", "r_i"))
        for i, r_ in enumerate(r):
            csvwriter.writerow((i, log(r_, 2) / 2.0, r_ ** (0.5)))


@begin.subcommand
@begin.convert(upper_limit=int, lower_limit=int, step_size=int, dfact=float, max_loops=int, c=float, sd=bool)
def rhf(
    upper_limit: "Compute up to this dimension (inclusive).",
    lower_limit: "Compute starting at this dimension, if ``None`` dimension 2 is chosen." = 2,
    step_size: "Increase dimension by this much in each step." = 1,
    dfact: "Dimension of the lattice is larger by this factor than block size" = 2,
    max_loops: "Maximum number of BKZ tours." = 8,
    filename: "Output filename" = "../data/bkz-rhf-{dfact:.2f}.csv",
):
    """
    Print and save δ for different BKZ variants.
    """

    strategies = {
        0.00: "../data/approx-hsvp-simulations,qary,1.00-1.50,0.00,1.00-strategies.sobj",
        0.15: "../data/approx-hsvp-simulations,qary,1.00-1.50,0.15,best-strategies.sobj",
        0.25: "../data/approx-hsvp-simulations,qary,1.00-1.50,0.25,best-strategies.sobj",
    }

    filename = filename.format(dfact=dfact)

    results = OrderedDict()

    from fpylll.tools.bkz_simulator import simulate as bkz_simulate

    def rhff(r):
        d = len(r)
        log_vol = sum(log(r_) for r_ in r) / 2
        return exp((log(r[0]) / 2 - log_vol / d) / (d - 1))

    def chenf(k):
        return float(k / (2 * pi * e) * (pi * k) ** (1.0 / k)) ** (1.0 / (2.0 * (k - 1)))

    results["ghk"] = OrderedDict()  # (gh(k)^(1/(k-1)))
    results["k2k"] = OrderedDict()  # k^(1/(2k))
    results["chen"] = OrderedDict()
    results["bkz"] = OrderedDict()
    for k in range(lower_limit, upper_limit + step_size, step_size):
        results["ghk"][k] = exp(lghf(k) * (1 / (k - 1)))
        results["k2k"][k] = k ** (1.0 / (2.0 * k))
        results["chen"][k] = chenf(k)

    for c in (0.00, 0.15, 0.25):
        strategy = pickle.load(open(strategies[c], "rb"))
        key_nd = "c{c:03d}".format(c=int(100 * c))
        key_sd = "sd-c{c:03d}".format(c=int(100 * c))
        results[key_nd] = OrderedDict()
        results[key_sd] = OrderedDict()

        if c == 0.00:
            from bkz import BKZQualitySimulation as Simulator
        else:
            from bkz import ProcrastinatingBKZQualitySimulation as Simulator

        for k in range(lower_limit, min(len(strategy), upper_limit + step_size), step_size):
            r = sample_r(round(dfact * k))
            params = BKZ.EasyParam(block_size=k, max_loops=max_loops, c=c, strategies=strategy)
            r_nd = Simulator(list(r), preprocessing_levels=0)(params)
            r_bkz = bkz_simulate(list(r), params)[0]

            params = BKZ.EasyParam(block_size=k, max_loops=max_loops // 2, c=c, strategies=strategy)
            r_sd = [1.0 / r_ for r_ in reversed(r)]
            r_sd = Simulator(list(r_sd), preprocessing_levels=0)(params)
            r_sd = [1.0 / r_ for r_ in reversed(r_sd)]
            r_sd = Simulator(list(r_sd), preprocessing_levels=0)(params)

            results["bkz"][k] = rhff(r_bkz)
            results[key_nd][k] = rhff(r_nd)
            results[key_sd][k] = rhff(r_sd)

    with open(filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header = ["k"]
        fmt = "k: {k:3d}, "
        for key in results.keys():
            header.append(key)
            fmt += "{key}: {{{key}:s}}, ".format(key=key)
        fmt = fmt[:-2]
        print(fmt)
        csvwriter.writerow(header)

        for k in range(lower_limit, upper_limit + step_size, step_size):
            row = OrderedDict()
            row["k"] = k
            for key in results.keys():
                try:
                    row[key] = "%7.5f" % results[key][k]
                except KeyError:
                    row[key] = "nan"

            print(fmt.format(**row))
            csvwriter.writerow(row.values())

    return results


@begin.start
@begin.logging
def run():
    pass
