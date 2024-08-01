#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run HSVP_α reduction and record statistics.

"""
from collections import OrderedDict
from fpylll import FPLLL, IntegerMatrix, GSO, LLL, BKZ, Pruning, Enumeration, EnumerationError, load_strategies_json
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.tools.bkz_stats import BKZTreeTracer
from fpylll.util import gaussian_heuristic
from math import log, ceil, sqrt
from multiprocessing import Queue, Process, active_children
import begin
import logging
import os
import pickle

# Verbose logging

logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S %Z")

# define a Handler which writes INFO messages or higher to the sys.stderr
logger = logging.getLogger(__name__)


def sample_matrix(d, lattice_type="qary", seed=None):
    """
    Sample a matrix in dimension `d`.

    :param d: lattice dimension
    :param lattice_type: see module level documentation
    :param seed: optional random seed
    :returns: LLL-reduced integer matrix

    .. note :: This function seeds the FPLLL RNG, i.e. it is deterministic.

    """

    if seed is None:
        FPLLL.set_random_seed(d)
    else:
        FPLLL.set_random_seed(seed)

    if lattice_type == "qary":
        A = IntegerMatrix.random(d, "qary", bits=30, k=d // 2, int_type="long")
    elif lattice_type == "qary-lv":
        A = IntegerMatrix.random(d, "qary", bits=10 * d, k=d // 2)
    else:
        raise ValueError("Lattice type '%s' not supported." % lattice_type)

    A = LLL.reduction(A)
    return A


def chunk_iterator(lst, step):
    """Return up to ``step`` entries from ``lst`` each time this function is called.

    :param lst: a list
    :param step: number of elements to return

    """
    for i in range(0, len(lst), step):
        yield tuple(lst[j] for j in range(i, min(i + step, len(lst))))


def svp_time(seed, params, return_queue=None):
    """Run SVP reduction on ``A`` using ``params``.

    :param seed: random seed for matrix creation
    :param params: BKZ parameters
    :param return_queue: if not ``None``, the result is put on this queue.

    """

    if params["c"] > 0:
        raise NotImplementedError("c≥0 not supported.")

    A = sample_matrix(params.block_size, seed=seed)
    M = GSO.Mat(A)
    bkz = BKZ2(M)
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    with tracer.context(("tour", 0)):
        bkz.svp_reduction(0, params.block_size, params, tracer)
        bkz.M.update_gso()

    tracer.exit()

    tracer.trace.data["|A_0|"] = A[0].norm()
    ppbs = params.strategies[params.block_size].preprocessing_block_sizes
    tracer.trace.data["preprocessing_block_sizes"] = ppbs

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


def approx_svp_time(seed, params, return_queue=None):
    """Run HSVP_α reduction on ``A`` using ``params``.

    :param seed: random seed for matrix creation
    :param params: BKZ preprocessing parameters, preprocessing block size is ignored
    :param return_queue: if not ``None``, the result is put on this queue.

    """
    from fpylll.algorithms.bkz import BKZReduction as BKZBase

    c = params["c"]

    FPLLL.set_random_seed(seed)
    d = ceil((1 + c) * params.block_size)
    A = sample_matrix(d, seed=seed)
    M = GSO.Mat(A)
    M.update_gso()

    nodes_per_second = 2.0 * 10 ** 9 / 64.0  # NOTE: This is a somewhat rough approximation

    self = BKZ2(M)
    tracer = BKZTreeTracer(self, start_clocks=True)

    preproc_block_sizes = params["ahsvp preprocessing block sizes"]

    rerandomize, preproc_cost = False, None
    repeat_count = 0
    with tracer.context(("tour", 0)):
        while True:
            repeat_count += 1
            with tracer.context("preprocessing"):
                if rerandomize:
                    self.randomize_block(1, d, density=params.rerandomization_density, tracer=tracer)
                with tracer.context("reduction"):
                    BKZBase.svp_preprocessing(self, 0, d, params, tracer)  # LLL
                    for preproc_block_size in preproc_block_sizes:
                        prepar = params.__class__(
                            block_size=preproc_block_size, strategies=params.strategies, flags=BKZ.GH_BND
                        )
                        self.tour(prepar, 0, d, tracer=tracer)

            if preproc_cost is None:
                preproc_cost = float(tracer.trace.find("preprocessing")["cputime"])
                preproc_cost *= nodes_per_second

            gh = gaussian_heuristic(M.r(0, params.block_size))
            target_norm = params["alpha"] ** 2 * gh

            with tracer.context("pruner"):
                pruner = Pruning.Pruner(
                    target_norm, preproc_cost, [M.r(0, params.block_size)], target=1, metric=Pruning.EXPECTED_SOLUTIONS
                )
                coefficients = pruner.optimize_coefficients([1.0] * params.block_size)
            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration", enum_obj=enum_obj, full=True):
                    max_dist, solution = enum_obj.enumerate(0, params.block_size, target_norm, 0, pruning=coefficients)[
                        0
                    ]
                with tracer.context("postprocessing"):
                    self.svp_postprocessing(0, params.block_size, solution, tracer=tracer)
                rerandomize = False
            except EnumerationError:
                rerandomize = True

            self.M.update_gso()
            logger.debug(
                "r_0: %7.2f, target: %7.2f, preproc: %3d"
                % (log(M.get_r(0, 0), 2), log(target_norm, 2), preproc_block_size)
            )
            if self.M.get_r(0, 0) <= target_norm:
                break

    tracer.exit()
    tracer.trace.data["repeat"] = repeat_count
    tracer.trace.data["alpha"] = A[0].norm() / sqrt(gaussian_heuristic(M.r(0, params.block_size)))
    tracer.trace.data["preprocessing_block_sizes"] = preproc_block_sizes

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


@begin.start(auto_convert=True)
@begin.logging
@begin.convert(max_block_size=float, lower_bound=int, step_size=int, alpha=float, c=float, samples=int, jobs=int)
def call(
    max_block_size: "compute up to this block size",
    bkz_strategies: "BKZ strategies (used for preprocessing)",
    ahsvp_strategies: "HSVP strategies (used for deciding preprocessing block size)" = None,
    dump_filename: """results are stored in this filename, if ``None``
         then ``../data/approx-hsvp-fplll-observations,{alpha:.2f},{c:.2f},[{bkz_strategies},{ahsvp_strategies}].sobj`` is used.""" = None,
    jobs: "number of experiments to run parallel" = 1,
    alpha: "approximation factor ≥ 1" = 1.00,
    c: "overshooting parameter ≥ 0" = 0.00,
    lower_bound: "Start experiment in this dimension" = 60,
    step_size: "Increment dimension by this much each iteration" = 2,
    samples: "number of samples to take" = 64,
):
    """
    Run (Approx-)SVP reduction and record statistics.

    """
    results = OrderedDict()
    # this is to support alpha*100 etc as block size
    max_block_size = int(round(max_block_size))

    if isinstance(bkz_strategies, str):
        if bkz_strategies.endswith(".json"):
            bkz_strategies_ = load_strategies_json(bytes(bkz_strategies, "ascii"))
        elif bkz_strategies.endswith(".sobj"):
            bkz_strategies_ = pickle.load(open(bkz_strategies, "rb"))

    if alpha > 1:
        if ahsvp_strategies is None:
            ahsvp_strategies = "../data/approx-hsvp-simulations,qary,{alpha:.2f},{c:.2f},1.00.csv".format(
                alpha=alpha, c=c
            )

        if isinstance(ahsvp_strategies, str):
            import csv

            reader = csv.DictReader(open(ahsvp_strategies, "r"))
            ahsvp_strategies_ = [None] * (max_block_size + 1)
            for row in reader:
                if int(row["k alpha"]) <= max_block_size:
                    ahsvp_strategies_[int(row["k alpha"])] = tuple([int(row["k pre"])] * 4)

        block_sizes = []
        for k in range(lower_bound, max_block_size + 1, step_size):
            if ahsvp_strategies_[k] is not None:
                block_sizes.append(k)

        target = approx_svp_time
    else:

        block_sizes = range(lower_bound, max_block_size + 1, step_size)
        target = svp_time

    if dump_filename is None:
        dump_filename = (
            "../data/approx-hsvp-fplll-observations"
            ",{alpha:.2f},{c:.2f},"
            "[{bkz_strategies},{ahsvp_strategies}].sobj"
        ).format(
            bkz_strategies=os.path.basename(bkz_strategies),
            ahsvp_strategies=os.path.basename(ahsvp_strategies) if ahsvp_strategies else "",
            alpha=alpha,
            c=c,
        )

    for block_size in block_sizes:
        return_queue = Queue()
        result = OrderedDict([("total time", None)])

        traces = []
        # 2. run `chunk` processes in parallel
        for chunk in chunk_iterator(range(samples), jobs):
            processes = []
            for i in chunk:
                seed = i
                param = BKZ.Param(
                    block_size=block_size, strategies=list(bkz_strategies_), flags=BKZ.VERBOSE | BKZ.GH_BND
                )
                param["c"] = c
                if alpha > 1:
                    param["ahsvp preprocessing block sizes"] = ahsvp_strategies_[block_size]
                    param["alpha"] = alpha
                if jobs > 1:
                    process = Process(target=target, args=(seed, param, return_queue))
                    processes.append(process)
                    process.start()
                else:
                    traces.append(target(seed, param, None))

            active_children()

            if jobs > 1:
                for process in processes:
                    traces.append(return_queue.get())

        preprocessing_block_sizes = [trace.data["preprocessing_block_sizes"] for trace in traces][0]
        total_time = sum([float(trace.data["walltime"]) for trace in traces]) / samples
        alpha_res = sum([trace.data["alpha"] for trace in traces]) / samples
        repeat_count = sum([trace.data["repeat"] for trace in traces]) / samples
        enum_nodes = (
            sum([sum([float(enum["#enum"]) for enum in trace.find_all("enumeration")]) for trace in traces]) / samples
        )

        logger.info(
            "= block size: %3d, m: %3d, t: %10.3fs, preproc: %s, #: %3d, log(#enum): %6.1f, alpha = %.3f",
            block_size,
            samples,
            total_time,
            preprocessing_block_sizes,
            repeat_count,
            log(enum_nodes, 2),
            alpha_res,
        )

        result["total time"] = total_time
        result["preprocessing block sizes"] = preprocessing_block_sizes
        result["alpha"] = alpha
        result["#enum"] = enum_nodes
        result["traces"] = traces

        results[block_size] = result

        if results[block_size]["total time"] > 1.0 and samples > max(8, 2 * jobs):
            samples //= 2
            if samples < jobs:
                samples = jobs

        pickle.dump(results, open(dump_filename, "wb"))

    return results
