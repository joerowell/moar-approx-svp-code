#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulator for α-HSVP costs.

"""

import pickle
import logging
import csv
from typing import NamedTuple

from bkz import BKZQualitySimulation, ProcrastinatingBKZQualitySimulation, bkz_simulatef
from collections import namedtuple, OrderedDict
from fpylll import BKZ, FPLLL, Pruning, load_strategies_json
from fpylll.fplll.bkz_param import Strategy as Strategy
from fpylll.util import gaussian_heuristic
from math import log, ceil, sqrt
from multiprocessing import Pool

from utils import SuppressStream, mkdict, k_alphaf


class SConf:
    """
    One stop place to configure the simulation.
    """

    k_low = 50  # start of simulation
    alpha_history_len = 8  # length of history of best αs
    preproc_loops = 4  # number of BKZ-like tours used for preprocessing
    start_delta = -8  # δ compared to previous optimal value
    stop_delta = 32  # how deep to search, we use early abort, though
    step_size = 2  # consider very `step_size`'d dimension
    steps_per_iteration = 16  # number of steps to consider in one iteration
    early_abort_time_delta = 3.00  # We stop when the time is worse by this factor compared to best
    best_fudge = 1.5  # we have a bias towards smaller values of α
    dump_filename = (
        "../data/approx-hsvp-simulations,qary,"
        "{alpha_low:.2f}-{alpha_high:.2f},"
        "{c:4.2f},"
        "{preproc_strategy}"
        "{rho}"
        ".sobj"
    )

    levels_filename = ("../data/approx-hsvp-simulation-levels,qary,"
                       "{alpha_low:.2f}--{alpha_high:.2f},"
                       "{c:4.2f},"
                       "{preproc_strategy}",
                       "{rho}",
                       ".sobj"
    )
                       


def sample_r(d):
    """
    Sample squared Gram-Schmidt norms of an LLL reduced lattice in dimension d.

    :param d: lattice dimension

    """
    q = 2 ** 30
    return [1.0219 ** (2 * (d - 2 * i - 1)) * q for i in range(d)]


def lll_cost(d):
    """
    Cost of LLL in dimension `d` in enumeration nodes.

    :param d: lattice dimension

    .. note:: We are ignoring the bit-size of the input here.

    """
    return d ** 3


def preprocess(r, c, k_, strategies, max_loops=1, bkz_simulate=None):
    """
    Perform preprocessing with quality `GH(k')^(1/k')`

    :param r: squared Gram-Schmidt norms
    :param c: overshooting parameter ≥ 0
    :param k_: preprocessing parameter
    :param strategies: recursive strategies
    :param max_loops: number of preprocessing tours
    :param bkz_simulate: simulation implementation

    """

    def get_cost(k_):
        return strategies[k_]["total_cost"]

    params = BKZ.EasyParam(k_, strategies=strategies, max_loops=max_loops)
    r = bkz_simulate(r, params)[0]

    D = len(r)
    cost = lll_cost(D)  # we first run LLL

    if k_ == 2:
        return r, cost

    if c == 0:
        cls = BKZQualitySimulation
    elif c > 0:
        cls = ProcrastinatingBKZQualitySimulation
    else:
        raise ValueError("c < 0 does not make sense")

    for kappa in range(D - 3):
        k_ = cls.k_tailf(params, D - kappa)
        cost += max_loops * get_cost(k_)

    return r, cost


PruningData = namedtuple("PruningData", ("coefficients", "total_cost", "single_enum_cost", "single_enum_cost_levels", "nsolutions"))


def pruning_parameters(r, preprocessing_cost, alpha=1.0, greedy=False, rho=None):
    """
    Compute pruning coefficients for `r`.

    :param r: squared Gram-Schmidt norms
    :param preprocessing_cost: cost of achieving basis of quality similar to r
    :param alpha: relaxation parameter ≥ 1
    :param greedy: use Greedy pruning strategy
    :param rho: pruning as in Theorem 3, LN20

    """
    gh = gaussian_heuristic(r)
    target_norm = alpha ** 2 * gh

    for float_type in ("d", "ld", "dd", 212, 318, 424):
        if isinstance(float_type, int):
            FPLLL.set_precision(float_type)
        try:
            with SuppressStream():
                pruner = Pruning.Pruner(
                    target_norm,
                    preprocessing_cost,
                    [r],
                    target=1,
                    metric=Pruning.EXPECTED_SOLUTIONS,
                    float_type=float_type if not isinstance(float_type, int) else "mpfr",
                    flags=Pruning.HALF if greedy else Pruning.GRADIENT | Pruning.HALF,
                )

                if rho:
                    coefficients = [1.0] * (len(r) // 2) + [sqrt(rho)] * (len(r) - len(r) // 2)
                else:
                    coefficients = pruner.optimize_coefficients([1.0] * len(r))

                # We extract both the per level costs and the overall cost
                single_cost, per_level_costs = pruner.single_enum_cost(coefficients, True)                    
                return PruningData(
                    [(alpha ** 2, coefficients)],
                    preprocessing_cost + pruner.repeated_enum_cost(coefficients),
                    single_cost, per_level_costs,
                    pruner.measure_metric(coefficients),
                )
        except RuntimeError:
            pass

    raise RuntimeError("Ran out of precision for k={k}, α={alpha:.2f}".format(k=len(r), alpha=alpha))


class Cost(NamedTuple):
    """
    Enumeration cost to achieve `RHF = GH(k)^{1/(k-1)}``
    """

    k: int
    alpha: float
    k_alpha: int
    c: float
    total_cost: float
    k_pre: int
    single_enum: float
    single_enum_cost_levels : (float)
    preprocessing_cost: float
    solutions: float
    strategy: Strategy

    def __repr__(self):
        return (
            "{k:4d} :: "
            "cost {cost:5.1f}, "
            "α: {alpha:.2f}, k_α: {k_alpha:4d}, "
            "k': {k_pre:4d}, "
            "single enum: {single_enum:5.1f}, "
            "sol: {solutions:4.2f}, "
            "rep: 2^{repeat:.2f}"
        ).format(
            k=self.k,
            k_alpha=self.k_alpha,
            alpha=self.alpha,
            cost=log(self.total_cost, 2),
            single_enum=log(self.single_enum, 2),
            k_pre=self.k_pre,
            solutions=self.solutions,
            repeat=log(self.total_cost / (self.single_enum + self.preprocessing_cost), 2),
        )


def cost_kernel(opts):
    """
    :param alpha: relaxation parameter ≥ 1
    :param c: overshooting parameter ≥ 0
    :param k: compute costs for `rhf = GH(k)^(1/(k-1))`
    :param k_pre: preprocess with `rhf = GH(k_{pre})^(1/(k_{pre}-1))`
    :param preproc_loops: number of preprocessing tours
    :param strategies: preprocessing strategies

    """
    alpha = opts["alpha"]
    c = opts["c"]
    k = opts["k"]
    k_pre = opts["k_pre"]
    preproc_loops = opts["preproc_loops"]
    k_alpha = k_alphaf(k, alpha)
    # NOTE: We insist on even k_α because we use Pruning.HALF
    k_alpha += k_alpha % 2

    # We are considering a lattice of dimension (1+c)⋅k_α
    D = ceil((1 + c) * k_alpha)
    r = sample_r(D)

    if c == 0.00:
        f = bkz_simulatef(BKZQualitySimulation, init_kwds={"preprocessing_levels": 1})
    elif c > 0:
        f = bkz_simulatef(
            ProcrastinatingBKZQualitySimulation, init_kwds={"preprocessing_levels": 1}, call_kwds={"c": c}
        )
    else:
        raise ValueError("c={c} not supported.".format(c=c))

    r, preprocessing_cost = preprocess(
        r=r, c=c, k_=k_pre, strategies=opts["strategies"], max_loops=preproc_loops, bkz_simulate=f
    )

    pp = pruning_parameters(r[:k_alpha], preprocessing_cost, alpha=alpha, greedy=opts["greedy"], rho=opts["rho"])

    strategy = Strategy(
        k,
        alpha=alpha,
        c=c,
        preprocessing_block_sizes=[k_pre] * preproc_loops if k_pre > 2 else [],
        pruning_parameters=pp.coefficients,
        total_cost=pp.total_cost,
    )

    return Cost(
        **{
            "k": k,
            "alpha": alpha,
            "k_alpha": k_alpha,
            "c": c,
            "total_cost": pp.total_cost,
            "k_pre": k_pre,
            "single_enum": pp.single_enum_cost,
            "single_enum_cost_levels": pp.single_enum_cost_levels,
            "preprocessing_cost": preprocessing_cost,
            "solutions": pp.nsolutions,
            "strategy": strategy,
        }
    )


def bootstrap_strategies(d, input_strategies=None, input_costs=None, c=0.00):
    """
    We bootstrap the search by using the assumption that for small block sizes α=1.00 is optimal.

    :param d: bootstrap until this dimension (exclusive)
    :param input_strategies: SVP strategies
    :param input_costs: SVP costs
    :param c: overshooting parameter c ≥ 0

    """
    if input_strategies is None:
        if c == 0.00:
            input_strategies = load_strategies_json(b"../data/fplll-strategies-one-tour-strombenzin.json")
        elif c > 0.00:
            input_strategies = load_strategies_json(b"../data/fplll-block-strategies,strategizer,2.json")
        else:
            raise ValueError("c < 0")

    if input_costs is None:
        input_costs = OrderedDict()
        if c == 0.00:
            fh = open("../data/fplll-observations,qary,[one-tour-strombenzin.json].csv", "r")
        elif c > 0.00:
            fh = open("../data/fplll-observations,qary,[fplll-block-strategies,strategizer,2.json].csv", "r")
        else:
            raise ValueError("c={c} not supported.".format(c=c))

        reader = csv.DictReader(fh, delimiter=",")
        for row in reader:
            input_costs[int(row["d"])] = row

    output_strategies = []

    for k in range(2):
        output_strategies.append(Strategy(k))

    for k in range(2, d):
        strat = Strategy(
            k,
            preprocessing_block_sizes=input_strategies[k].preprocessing_block_sizes,
            pruning_parameters=input_strategies[k].pruning_parameters,
            alpha=1.0,
            c=c,
            total_cost=float(input_costs[k]["total time"]) * 10 ** 9 * 2.6 / 64,
        )
        output_strategies.append(strat)

    return output_strategies


def early_abort(current, best):
    """
    Abort the search for preprocessing parameters if this function returns ``True``

    :param current: current candicate
    :param best: best found so far

    """
    return current.total_cost > SConf.early_abort_time_delta * best.total_cost


def find_best(data, min_alpha=None):
    """

    Find entry with lowest total cost.

    :param data: a dictionary, indexed by α values of costs
    :param min_alpha: αs < min_alpha will be skipped

    """

    best = None

    for alpha in sorted(data):
        if data[alpha] is None:
            continue
        if min_alpha is not None and alpha < min_alpha:
            continue
        if best is None:
            best = data[alpha]
        elif best.total_cost > SConf.best_fudge * data[alpha].total_cost:
            best = data[alpha]
    return best


def increase_level_maybe(preproc_level, alpha_history, best_alpha, alphas, doincrease=True):
    """
    Decide if α should be increased for the next ranks.

    This is done using a rolling history of best αs.

    :param preproc_level: current preproc_level (index into ``alphas``)
    :param alpha_history: rolling history of best αs so far
    :param best_alpha: current best α
    :param alphas: all available αs
    :param doincrease: actually increase the level?

    """
    alpha_history = alpha_history[1:] + [best_alpha]
    if (
        doincrease
        and all(alpha_ > alphas[preproc_level] for alpha_ in alpha_history)
        and preproc_level < len(alphas) - 1
    ):
        # we make it harder to increase α as we increase the ranks
        alpha_history = alpha_history + alpha_history
        preproc_level += 1

    return preproc_level, alpha_history


def simulate(
    k_high: "compute up to ``RHF = GH(k_high)^(1/(k_high-1))`` (inclusive).",
    alphas: "An iterable of alpha values",
    c: "overshooting parameter" = 0.00,
    preproc_strategy: 'either some `α ∈ αs` or "best"' = "best",
    dump_filename: """results are regularly written to this filename, if ``None``
               then a filename will be chosen.""" = None,
    jobs: "number of cores to use in parallel" = 1,
    greedy: "use Greedy pruning strategy" = False,
    rho: "fix ρ as in [LN20,Theorem 3]" = None,
):
    """
    Estimate cost of enumeration using `α-GH(k_α)^{1/(k_α-1)}` oracles.
    """

    alphas = list(alphas)

    if preproc_strategy != "best" and preproc_strategy not in alphas:
        raise ValueError(
            "preproc_strategy={preproc_stategy} but {preproc_stategy} ∉ {alphas}".format(
                preproc_strategy=preproc_strategy, alphas=alphas
            )
        )

    if preproc_strategy == "best":
        preproc_level = 0
    else:
        preproc_level = alphas.index(preproc_strategy)

    alpha_history = [alphas[preproc_level]] * SConf.alpha_history_len

    if dump_filename is None:
        dump_filename = (SConf.dump_filename).format(
            alpha_low=alphas[0],
            alpha_high=alphas[-1],
            c=c,
            preproc_strategy=preproc_strategy if preproc_strategy == "best" else "%.2f" % preproc_strategy,
            rho=",%s" % rho if rho is not None else "",
        )

    strategies = bootstrap_strategies(SConf.k_low)

    if jobs > 1:
        workers = Pool(jobs)

    costs = OrderedDict()

    for k in range(SConf.k_low, k_high + 1, SConf.step_size):

        tasks_in_class = mkdict(alphas, "list")

        for alpha in alphas:
            try:
                start = max(costs[k - SConf.step_size][alpha]["k_pre"] + SConf.start_delta, 2)
            except KeyError:
                start = 2
            stop = min(start + SConf.step_size * SConf.stop_delta, k)

            for k_pre in range(start, stop, SConf.step_size):
                tasks_in_class[alpha].append(
                    {
                        "k": k,
                        "k_pre": k_pre,
                        "alpha": alpha,
                        "c": c,
                        "greedy": greedy,
                        "rho": rho,
                        "preproc_loops": SConf.preproc_loops,
                        "strategies": strategies,
                    }
                )

        best_in_class = mkdict(alphas, None)
        done_in_class = mkdict(alphas, False)
        costs[k] = mkdict(alphas, None)
        costs[k]["best"] = None

        while not all(done_in_class.values()):
            tasks = []
            for alpha in reversed(alphas):
                if done_in_class[alpha]:
                    continue
                for _ in range(SConf.steps_per_iteration):
                    try:
                        tasks.append(tasks_in_class[alpha].pop(0))
                    except IndexError:
                        done_in_class[alpha] = True
                        break

            results_in_class = mkdict(alphas, "OrderedDict")

            if jobs == 1:
                for result in [cost_kernel(task) for task in tasks]:
                    results_in_class[result.alpha][result.k_pre] = result
            else:
                for result in workers.map(cost_kernel, tasks):
                    results_in_class[result.alpha][result.k_pre] = result
            del tasks

            for alpha in sorted(results_in_class):
                for result in sorted(results_in_class[alpha].values(), key=lambda x: x.k_pre):
                    if best_in_class[alpha] is None or result.total_cost < best_in_class[alpha].total_cost:
                        best_in_class[alpha] = result

                    if early_abort(result, best_in_class[alpha]):
                        done_in_class[alpha] = True

        for alpha in best_in_class:
            costs[k][alpha] = best_in_class[alpha]._asdict()
            logging.debug("DEBUG {best}".format(best=best_in_class[alpha]))

        best = find_best(best_in_class, min_alpha=alphas[preproc_level])
        costs[k]["best"] = best._asdict()
        logging.info(
            "{best}, α_pre: {preproc:.2f}, α[ℓ_pre]: {alpha:.2f}, avg(α_best): {alphah:.3f}".format(
                best=best,
                preproc=strategies[best.k_pre]["alpha"],
                alpha=alphas[preproc_level],
                alphah=sum(alpha_history) / len(alpha_history),
            )
        )

        # we need a strategy for all block sizes not just for those covered here
        for k_ in range(k, k + SConf.step_size):
            strategy_ = costs[k][alphas[preproc_level]]["strategy"]
            strategy_ = strategy_.dict()
            strategy_["block_size"] = k_
            strategies.append(Strategy(**strategy_))

        assert strategies[k].block_size == k

        preproc_level, alpha_history = increase_level_maybe(
            preproc_level, alpha_history, best.alpha, alphas, doincrease=preproc_strategy == "best"
        )

        if dump_filename:
            pickle.dump(costs, open(dump_filename, "wb"))
            pickle.dump(strategies, open(dump_filename.replace(".sobj", "-strategies.sobj"), "wb"))
        
    return costs, strategies
