# -*- coding: utf-8 -*-
"""
Simulate FastEnum BKZ variants
"""
from collections import OrderedDict
from math import ceil, lgamma, log, pi
from contextlib import contextmanager

from fpylll import BKZ, GSO, IntegerMatrix, LLL, Pruning, Enumeration, EnumerationError
from fpylll.tools.bkz_stats import Accumulator, Node, Tracer, dummy_tracer, pretty_dict
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from utils import k_alphaf
from time import process_time


def gh_normalizer_log2(d, plain=False):
    """
    Return the log2() of normalization factor for the Gaussian heuristic.

    :param d: dimensions
    :param plain: if ``True`` do not deviate from plain formula for small dimensions

    """

    rk = (
        0.789527997160000,
        0.780003183804613,
        0.750872218594458,
        0.706520454592593,
        0.696345241018901,  # noqa
        0.660533841808400,
        0.626274718790505,
        0.581480717333169,
        0.553171463433503,
        0.520811087419712,
        0.487994338534253,
        0.459541470573431,
        0.414638319529319,
        0.392811729940846,
        0.339090376264829,
        0.306561491936042,
        0.276041187709516,
        0.236698863270441,
        0.196186341673080,
        0.161214212092249,
        0.110895134828114,
        0.0678261623920553,
        0.0272807162335610,
        -0.0234609979600137,
        -0.0320527224746912,
        -0.0940331032784437,
        -0.129109087817554,
        -0.176965384290173,
        -0.209405754915959,
        -0.265867993276493,
        -0.299031324494802,
        -0.349338597048432,
        -0.380428160303508,
        -0.427399405474537,
        -0.474944677694975,
        -0.530140672818150,
        -0.561625221138784,
        -0.612008793872032,
        -0.669011014635905,
        -0.713766731570930,
        -0.754041787011810,
        -0.808609696192079,
        -0.859933249032210,
        -0.884479963601658,
        -0.886666930030433,
    )

    if plain or d > 45:
        log_vol = log(pi, 2) / 2 * d - lgamma(d / 2.0 + 1) / log(2.0)
    else:
        log_vol = -rk[-d] * d + sum(rk[-d:])
    return log_vol


class BKZSimulationTreeTracer(Tracer):
    """
    A tracer for tracing simulations.
    """

    def __init__(self, instance, verbosity=False, root_label="bkz"):
        """
        Create a new tracer instance.

        :param instance: BKZ-like object instance
        :param verbosity: print information, integers >= 0 are also accepted
        :param root_label: label to give to root node

        """

        Tracer.__init__(self, instance, verbosity)
        self.trace = Node(root_label)
        self.current = self.trace

    def enter(self, label, **kwds):
        """
        Enter new context with label

        :param label: label

        """
        self.current = self.current.child(label)
        self.reenter()

    def reenter(self, **kwds):
        """
        Reenter current context.
        """
        self.current.data["cost"] = self.current.data.get("cost", Accumulator(1, repr="sum"))

    def inc_cost(self, cost):
        self.current.data["cost"] += Accumulator(cost, repr="sum")

    def exit(self, **kwds):
        """
        When the label is a tour then the status is printed if verbosity > 0.
        """
        node = self.current
        label = node.label

        if label[0] == "tour":
            data = basis_quality([2 ** (2 * r_) for r_ in self.instance.r])
            for k, v in data.items():
                if k == "/":
                    node.data[k] = Accumulator(v, repr="max")
                else:
                    node.data[k] = Accumulator(v, repr="min")

        if self.verbosity and label[0] == "tour":
            nf = 64 / 2 / 10 ** 9
            report = OrderedDict()
            report["i"] = label[1]
            report["cputime"] = float(node.sum("cost")) * nf
            report["walltime"] = report["cputime"]
            report["preproc"] = float(node.sum("cost", label="preprocessing")) * nf
            report["svp"] = float(node.sum("cost", label="enumeration")) * nf
            report["#enum"] = node.sum("cost", label="enumeration")
            report["lll"] = float(node.sum("cost", label="lll")) * nf
            report["pruner"] = float(node.sum("cost", label="pruner")) * nf
            report["r_0"] = node["r_0"]
            report["/"] = node["/"]
            report["δ"] = node["rhf"]
            print(pretty_dict(report))

        self.current = self.current.parent


class BKZQualitySimulation(object):
    """
    Simulate quality of BKZ reduction.
    """

    def __init__(self, A, preprocessing_levels=1, preprocessing_cutoff=45):
        """
        Create a new BKZ Simulation object.

        :param A: An integer matrix, a GSO object or a list of squared Gram-Schmidt norms.
        :param preprocessing_levels: how many levels of preprocessing to simulate (slow!)

        .. note :: Our internal representation is log2 norms of Gram-Schmidt vectors (not squared).

        """
        if isinstance(A, GSO.Mat):
            A.update_gso()
            r = A.r()
            self.r = [log(r_, 2) / 2.0 for r_ in r]
        elif isinstance(A, LLL.Reduction):
            A.M.update_gso()
            r = A.M.r()
            self.r = [log(r_, 2) / 2.0 for r_ in r]
        elif isinstance(A, IntegerMatrix):
            M = GSO.Mat(LLL.reduction(A))
            M.update_gso()
            r = M.r()
            self.r = [log(r_, 2) / 2.0 for r_ in r]
        else:
            try:
                self.r = [log(r_, 2) / 2.0 for r_ in A]
            except TypeError:
                raise TypeError("Unsupported type '%s'" % type(A))

        self.preprocessing_levels = preprocessing_levels
        self.preprocessing_cutoff = preprocessing_cutoff

    @contextmanager
    def descent(self, preproc):
        """
        Context for limiting decent downward when preprocessing.

        This will return ``skip=True`` when recursion should be skipped.

        """
        if not hasattr(self, "level"):
            self.level = 0

        self.level += 1
        skip = (preproc <= self.preprocessing_cutoff) or (self.preprocessing_levels <= self.level)
        try:
            yield skip
        finally:
            self.level -= 1

    @classmethod
    def alphaf(cls, params, k, default=1.00):
        """
        Pick α

        :param params: BKZ parameters
        :param k: ignored
        :param default: default value

        .. note: Our strategy is to have the same `α` for all blocks, regardless of size, on a level.
           Different levels may have different `α`s.

        """

        try:
            # NOTE we are *ignoring* k here
            return params.strategies[params.block_size]["alpha"]
        except KeyError:
            return 1.00

    @classmethod
    def k_tailf(cls, params, space):
        """
        Pick actual dimension used.

        :param params: BKZ parameters
        :param space: dimensions available

        ..  note : Our strategy is to use k_α if we have the space, otherwise we decrease
            linearly towards the end.

        """
        k = params.block_size
        k_alpha = k_alphaf(k, cls.alphaf(params, k))

        if space >= k_alpha:
            return k

        return max(min(space, ceil(k - (k_alpha - space) / 2.0)), 2)

    def enumeration_context_sizef(self, kappa, k, params):
        """
        A variant of the ``k_alphaf`` function that takes the remaining space into account.

        :param kappa: current index
        :param k: SVP parameter
        :param params: BKZ parameters

        """

        return min(k_alphaf(k, self.alphaf(params, k)), len(self.r) - kappa)

    def __call__(self, params, min_row=0, max_row=-1, tracer=None, **kwds):
        """
        Simulate quality of BKZ reduction.

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :param kwds: added to parameters
        :returns: Squared Gram-Schmidt norms.

        """
        self.level = 0
        i = 0

        if tracer is None:
            tracer = BKZSimulationTreeTracer(
                self, root_label="BKZQualitySimulation", verbosity=params.flags & BKZ.VERBOSE
            )

        params = params.new(**kwds)  # import kwds

        while True:
            with tracer.context(("tour", i)):
                clean = self.tour(params, min_row, max_row, tracer=tracer)
            i += 1
            if clean or k_alphaf(params.block_size, self.alphaf(params, params.block_size)) >= len(self.r):
                break  # HKZ
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break

        tracer.exit()
        self.trace = tracer.trace

        return tuple([2 ** (2 * r_) for r_ in self.r])

    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        """
        One tour of BKZ.

        :param params: BKZ parameters
        :param min_row: start processing at min_row (inclusive)
        :param max_row: stop processing at max_row (exclusive)
        :param tracer: tracer object
        :returns: whether the basis remained untouched or not

        """
        if max_row == -1:
            try:
                max_row = len(self.r)
            except AttributeError:
                max_row = self.M.d

        block_sizes = []
        clean = True

        for kappa in range(min_row, max_row - 1):
            k = self.k_tailf(params, max_row - kappa)
            block_sizes.append(k)
            clean &= self.svp_reduction(kappa, k, params, tracer=tracer)

        try:
            tracer.current.data["block_sizes"] = block_sizes
        except AttributeError:
            pass

        return clean

    def svp_reduction(self, kappa, k, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param k: SVP parameter
        :param params: BKZ parameters
        :param tracer: tracer object

        """
        clean = True
        context_size = self.enumeration_context_sizef(kappa, k, params)

        with tracer.context("preprocessing"):
            clean &= self.svp_preprocessing(kappa, k, params, tracer=tracer, end=kappa + context_size)

        with tracer.context("enumeration"):
            solution = self.svp_call(kappa, k, params, tracer=tracer)

        with tracer.context("postprocessing"):
            clean &= self.svp_postprocessing(kappa, context_size, solution, tracer=tracer)

        return clean

    def lll(self, start, end, tracer=dummy_tracer):
        """
        Simulate LLL on ``r[start:end]``

        :param start: first index to be touched
        :param end: last index to be touched (exclusive)

        """
        d = end - start
        if d <= 1:
            return

        delta_0 = log(1.0219, 2)
        alpha = delta_0 * (-2 * d / float(d - 1))
        rv = sum(self.r[start:end]) / d

        if basis_quality([2 ** (2 * r_) for r_ in self.r[start:end]])["rhf"] > 1.0219:
            self.r[start:end] = [(i * alpha + delta_0 * d) + rv for i in range(d)]

    def svp_preprocessing(self, kappa, k, params, tracer=dummy_tracer, end=None):
        """

        :param kappa: SVP start index
        :param k: SVP parameter
        :param params: BKZ parameters
        :param tracer: tracer object
        :param end: preprocess until this index (exclusive)

        """
        if end is None:
            end = kappa + k

        clean = True

        with tracer.context("lll"):
            self.lll(kappa + 1, end, tracer)

        if not params.strategies[k].preprocessing_block_sizes:
            return clean

        for k_ in params.strategies[k].preprocessing_block_sizes:
            with self.descent(k_) as skip:
                if not skip:
                    pre_params = params.new(block_size=k_, flags=BKZ.GH_BND)
                    clean &= self.tour(pre_params, kappa, end, tracer=tracer)

        return clean

    def svp_call(self, kappa, k, params, tracer=dummy_tracer):
        """
        Return log norm as predicted by Gaussian heuristic.

        :param kappa: SVP start index
        :param k: SVP parameter
        :param params: BKZ parameters
        :param tracer: ignored

        """
        alpha = self.alphaf(params, k)
        context_size = self.enumeration_context_sizef(kappa, k, params)
        log_vol = sum(self.r[kappa : kappa + context_size])
        normalizer = gh_normalizer_log2(context_size)
        return log(alpha, 2) + (log_vol - normalizer) / context_size

    def svp_postprocessing(self, kappa, block_size, solution, tracer=dummy_tracer):
        """
        Insert vector and distribute additional weight equally.

        :param kappa: SVP start index
        :param block_size:  SVP dimension
        :param solution: log norm of the found vector
        :param tracer: ignored

        """
        clean = True
        if solution < self.r[kappa]:
            clean = False
            delta = (self.r[kappa] - solution) / (block_size - 1)
            self.r[kappa] = solution
            for j in range(kappa + 1, kappa + block_size):
                self.r[j] += delta

        return clean


class BKZSimulation(BKZQualitySimulation):
    """
    Simulate quality and cost of Procrastinating-BKZ reduction.
    """

    def __init__(self, A):
        """
        Create a new simulation object

        :param A: An integer matrix, a GSO object or a list of squared Gram-Schmidt norms.

        """
        super(BKZSimulation, self).__init__(A, preprocessing_levels=1024, preprocessing_cutoff=19)

    def get_pruning(self, kappa, k, params, tracer=dummy_tracer):
        from fpylll.tools.quality import gaussian_heuristic

        strategy = params.strategies[k]
        alpha = strategy["alpha"]
        context_size = self.enumeration_context_sizef(kappa, k, params)
        gh_radius = gaussian_heuristic([2 ** (2 * r_) for r_ in self.r[kappa : kappa + context_size]])

        return alpha ** 2 * gh_radius, strategy.get_pruning(alpha ** 2 * gh_radius, gh_radius)

    def lll(self, start, end, tracer=dummy_tracer):
        """
        Simulate LLL on ``r[start:end]``

        :param start: first index to be touched
        :param end: last index to be touched (exclusive)

        """
        BKZQualitySimulation.lll(self, start, end, tracer=tracer)
        d = end - start
        if d <= 1:
            return
        try:
            tracer.inc_cost(cost=d ** 3)
        except AttributeError:
            pass

    def svp_preprocessing(self, kappa, k, params, tracer=dummy_tracer, end=None):
        """
        """
        if end is None:
            end = kappa + k

        with tracer.context("lll"):
            self.lll(kappa + 1, end, tracer)

        return super().svp_preprocessing(kappa, k, params, tracer, end=end)

    def svp_reduction(self, kappa, k, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param k: SVP parameter
        :param params: BKZ parameters
        :param tracer: tracer object

        """

        clean = True
        context_size = self.enumeration_context_sizef(kappa, k, params)
        remaining_trials = 1.0

        while remaining_trials > 0.0:
            with tracer.context("preprocessing"):
                clean &= self.svp_preprocessing(kappa, k, params, tracer=tracer, end=kappa + context_size)

            with tracer.context("enumeration"):
                solution = self.svp_call(kappa, k, params, tracer=tracer)
                radius, pr = self.get_pruning(kappa, k, params, tracer)
                tracer.inc_cost(params.strategies[k]["total_cost"])

            remaining_trials -= pr.expectation

            with tracer.context("postprocessing"):
                clean &= self.svp_postprocessing(kappa, context_size, solution, tracer=tracer)

        return clean


class BKZReduction(object):

    __init__ = BKZ2.__init__
    __call__ = BKZ2.__call__
    svp_postprocessing = BKZ2.svp_postprocessing
    randomize_block = BKZ2.randomize_block
    get_pruning = BKZ2.get_pruning

    tour = BKZSimulation.tour
    k_tailf = BKZSimulation.k_tailf
    alphaf = BKZSimulation.alphaf

    def enumeration_context_sizef(self, kappa, k, params):
        """
        A variant of the ``k_alphaf`` function that takes the remaining space into account.

        :param kappa: current index
        :param k: SVP parameter
        :param params: BKZ parameters

        """
        k_alpha = k_alphaf(k, self.alphaf(params, k))
        return min(k_alpha, self.M.d - kappa)

    def svp_preprocessing(self, kappa, k, params, tracer=dummy_tracer, end=None):
        clean = True

        if end is None:
            end = kappa + k

        lll_start = kappa if params.flags & BKZ.BOUNDED_LLL else 0
        with tracer.context("lll"):
            self.lll_obj(lll_start, lll_start, end)
            if self.lll_obj.nswaps > 0:
                clean = False

        for k_ in params.strategies[k].preprocessing_block_sizes:
            pre_params = params.new(block_size=k_, flags=BKZ.GH_BND)
            clean &= self.tour(pre_params, kappa, end, tracer=tracer)

        return clean

    def svp_reduction(self, kappa, k, params, tracer=dummy_tracer):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """
        from fpylll.util import gaussian_heuristic

        if params.strategies[k]["alpha"] == 1.0:
            return BKZ2.svp_reduction(self, kappa, k, params, tracer)

        self.lll_obj.size_reduction(0, kappa + 1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)
        nodes_per_second = 2.0 * 10 ** 9 / 64.0  # NOTE: This is a somewhat rough approximation
        rerandomize, preproc_cost, coefficients = False, None, None
        context_size = self.enumeration_context_sizef(kappa, k, params)

        for _ in range(k):  # should be enough
            with tracer.context("preprocessing"):
                if preproc_cost is None:
                    preproc_cost = -process_time()
                if rerandomize:
                    with tracer.context("randomization"):
                        self.randomize_block(
                            kappa + 1, kappa + context_size, density=params.rerandomization_density, tracer=tracer
                        )
                with tracer.context("reduction"):
                    self.svp_preprocessing(kappa, k, params, tracer=tracer, end=kappa + context_size)
                if preproc_cost < 0:
                    preproc_cost += process_time()
                    preproc_cost *= nodes_per_second

            with tracer.context("pruner"):
                if coefficients is None:
                    strategy = params.strategies[k]
                    alpha = strategy["alpha"]
                    self.lll_obj.size_reduction(0, kappa + context_size)  # HACK clean up
                    gh_radius = gaussian_heuristic(self.M.r(kappa, kappa + context_size))
                    target_norm = alpha ** 2 * gh_radius
                    pruner = Pruning.Pruner(
                        target_norm,
                        preproc_cost,
                        [self.M.r(kappa, kappa + context_size)],
                        target=1,
                        metric=Pruning.EXPECTED_SOLUTIONS,
                    )
                    coefficients = pruner.optimize_coefficients([1.0] * context_size)

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration", enum_obj=enum_obj, full=k == params.block_size):
                    max_dist, solution = enum_obj.enumerate(
                        kappa, kappa + context_size, target_norm, 0, pruning=coefficients
                    )[0]
                with tracer.context("postprocessing"):
                    self.svp_postprocessing(kappa, context_size, solution, tracer=tracer)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            self.lll_obj.size_reduction(0, kappa + 1)
            if self.M.get_r(kappa, kappa) <= target_norm:
                break

        self.lll_obj.size_reduction(0, kappa + context_size)  # HACK clean up
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2 ** (new_first_expo - old_first_expo)
        return clean


class ProcrastinatingBKZQualitySimulation(BKZQualitySimulation):
    """
    Simulate quality of Procrastinating-BKZ reduction.
    """

    @classmethod
    def cf(cls, params, block_size):
        """
        Pick overshooting parameter.

        :param params: BKZ parameters
        :param k: ignored

        .. note: Our strategy is to have the same `c` for all blocks, regardless of size, on a level.
           Different levels may have different `c`s.

        """
        try:
            # NOTE we are *ignoring* k here
            return params.strategies[params.block_size]["c"]
        except KeyError:
            return params["c"]

    @classmethod
    def k_tailf(cls, params, space):
        """
        Compute SVP parameter `k'` used for enumeration

        We are going roughly for a shape like this:

        - k while it fits
        - then reduce by roughly a rate of 1/2 to keep the cost constant
        - eventually we drop down to zero

            ----------------
                            | \
                            |    \
                            |       \
                            |        |\
                            |        | \
                            |        |  \
                            |        |   \

        :param params: to extract α and c
        :param space: dimensions available

        """
        k = params.block_size
        k_alpha = k_alphaf(k, cls.alphaf(params, k))
        context_size = ceil((1 + cls.cf(params, k)) * k_alpha)

        # easy part, everything fits
        if space >= context_size:
            return k

        return max(min(space, ceil(k - (context_size - space) / 2.0)), 2)

    def preprocessing_context_sizef(self, kappa, k, params):
        """
        Preprocessing context size taking the remaining space into account.

        :param kappa: current index
        :param k: SVP parameter
        :param params: BKZ parameters

        """
        k_alpha = k_alphaf(k, self.alphaf(params, k))
        context_size = ceil((1 + self.cf(params, k)) * k_alpha)
        return min(context_size, len(self.r) - kappa)

    def svp_reduction(self, kappa, k, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param block_size: SVP dimension
        :param params: BKZ parameters
        :param tracer: tracer object

        """
        clean = True
        with tracer.context("preprocessing"):
            context_size = self.preprocessing_context_sizef(kappa, k, params)
            clean &= self.svp_preprocessing(kappa, k, params, tracer=tracer, end=kappa + context_size)

        with tracer.context("enumeration"):
            solution = self.svp_call(kappa, k, params, tracer=tracer)

        with tracer.context("postprocessing"):
            context_size = self.enumeration_context_sizef(kappa, k, params)
            clean &= self.svp_postprocessing(kappa, context_size, solution, tracer=tracer)

        return clean


class ProcrastinatingBKZSimulation(BKZSimulation):
    """
    A simulator simulating both quality and time.
    """

    cf = ProcrastinatingBKZQualitySimulation.cf
    k_tailf = ProcrastinatingBKZQualitySimulation.k_tailf
    preprocessing_context_sizef = ProcrastinatingBKZQualitySimulation.preprocessing_context_sizef

    def svp_reduction(self, kappa, k, params, tracer=dummy_tracer):
        """
        Preprocessing, oracle call, postprocessing

        :param kappa: SVP start index
        :param k: SVP parameter
        :param params: BKZ parameters
        :param tracer: tracer object

        """

        clean = True
        remaining_trials = 1.0

        while remaining_trials > 0.0:
            with tracer.context("preprocessing"):
                context_size = self.preprocessing_context_sizef(kappa, k, params)
                clean &= self.svp_preprocessing(kappa, k, params, tracer=tracer, end=kappa + context_size)

            with tracer.context("enumeration"):
                solution = self.svp_call(kappa, k, params, tracer=tracer)
                radius, pr = self.get_pruning(kappa, k, params, tracer)
                tracer.inc_cost(params.strategies[k]["total_cost"])

            remaining_trials -= pr.expectation

            with tracer.context("postprocessing"):
                context_size = self.enumeration_context_sizef(kappa, k, params)
                clean &= self.svp_postprocessing(kappa, context_size, solution, tracer=tracer)

        return clean


class ProcrastinatingBKZReduction(BKZReduction):

    tour = ProcrastinatingBKZSimulation.tour
    k_tailf = ProcrastinatingBKZSimulation.k_tailf
    alphaf = ProcrastinatingBKZSimulation.alphaf
    cf = ProcrastinatingBKZQualitySimulation.cf

    def preprocessing_context_sizef(self, kappa, k, params):
        """
        Preprocessing context size taking the remaining space into account.

        :param kappa: current index
        :param k: SVP parameter
        :param params: BKZ parameters

        """
        k_alpha = k_alphaf(k, self.alphaf(params, k))
        context_size = ceil((1 + self.cf(params, k)) * k_alpha)
        return min(context_size, self.M.d - kappa)

    def svp_reduction(self, kappa, k, params, tracer=dummy_tracer):
        """

        :param kappa:
        :param block_size:
        :param params:
        :param tracer:

        """
        from fpylll.util import gaussian_heuristic

        if params.strategies[k]["alpha"] == 1.0:
            return BKZ2.svp_reduction(self, kappa, k, params, tracer)

        self.lll_obj.size_reduction(0, kappa + 1)
        old_first, old_first_expo = self.M.get_r_exp(kappa, kappa)
        nodes_per_second = 2.0 * 10 ** 9 / 64.0  # NOTE: This is a somewhat rough approximation
        rerandomize, preproc_cost, coefficients = False, None, None
        enumeration_context_size = self.enumeration_context_sizef(kappa, k, params)
        preprocessing_context_size = self.preprocessing_context_sizef(kappa, k, params)

        for _ in range(k):  # should be enough
            with tracer.context("preprocessing"):
                if preproc_cost is None:
                    preproc_cost = -process_time()
                if rerandomize:
                    with tracer.context("randomization"):
                        self.randomize_block(
                            kappa + 1,
                            kappa + preprocessing_context_size,
                            density=params.rerandomization_density,
                            tracer=tracer,
                        )
                with tracer.context("reduction"):
                    self.svp_preprocessing(kappa, k, params, tracer=tracer, end=kappa + preprocessing_context_size)
                if preproc_cost < 0:
                    preproc_cost += process_time()
                    preproc_cost *= nodes_per_second

            with tracer.context("pruner"):
                if coefficients is None:
                    strategy = params.strategies[k]
                    alpha = strategy["alpha"]
                    self.lll_obj.size_reduction(0, kappa + enumeration_context_size)  # HACK clean up
                    gh_radius = gaussian_heuristic(self.M.r(kappa, kappa + enumeration_context_size))
                    target_norm = alpha ** 2 * gh_radius
                    pruner = Pruning.Pruner(
                        target_norm,
                        preproc_cost,
                        [self.M.r(kappa, kappa + enumeration_context_size)],
                        target=1,
                        metric=Pruning.EXPECTED_SOLUTIONS,
                    )
                    coefficients = pruner.optimize_coefficients([1.0] * enumeration_context_size)

            try:
                enum_obj = Enumeration(self.M)
                with tracer.context("enumeration", enum_obj=enum_obj, full=k == params.block_size):
                    max_dist, solution = enum_obj.enumerate(
                        kappa, kappa + enumeration_context_size, target_norm, 0, pruning=coefficients
                    )[0]
                with tracer.context("postprocessing"):
                    self.svp_postprocessing(kappa, enumeration_context_size, solution, tracer=tracer)
                rerandomize = False

            except EnumerationError:
                rerandomize = True

            self.lll_obj.size_reduction(0, kappa + 1)
            if self.M.get_r(kappa, kappa) <= target_norm:
                break

        self.lll_obj.size_reduction(0, kappa + enumeration_context_size)  # HACK clean up
        new_first, new_first_expo = self.M.get_r_exp(kappa, kappa)

        clean = old_first <= new_first * 2 ** (new_first_expo - old_first_expo)
        return clean


def bkz_simulatef(cls, init_kwds=None, call_kwds=None):
    """
    Turn simulation class into a callable.

    :param cls: a Simulation class
    :param init_kwds: keywords passed to ``__init__``
    :param call_kwds: keywords passed to ``__call__``

    """

    if init_kwds is None:
        init_kwds = {}
    if call_kwds is None:
        call_kwds = {}

    def bkz_simulate(r, params):
        bkz = cls(r, **init_kwds)
        r = bkz(params, **call_kwds)
        return r, None

    return bkz_simulate
