import os, sys, pickle, numpy, platform, argparse, math, time
import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from util import *

VERBOSE = 0


def util_print_output_info(pr, c):
    allcost = pr.repeated_enum_cost(c)
    singlecost, details = pr.single_enum_cost(c, True)
    succprob = pr.measure_metric(c)

    # optimized coefficients
    if (VERBOSE):
        print("# Output    all_cost  = ", allcost)
        print("# Output single_cost  = ", singlecost)
        print("# Output single_prob  = ", succprob)

    return allcost, singlecost, succprob


# does the optimization
def optimize_cost_prob(L2, R2, precost, targetprob, coeff_old, c):

    if (VERBOSE):
        print("# Mode: overall optimiztion ")
        print("# Input R^2  = ")
        print(L2)
        print("# Input   bound  = ", R2)
        print("# Input precost  = ", precost)
        print("# Input  target  = ", targetprob)

    if (len(coeff_old) == len(L2) - 1):
        coeff_old = list(coeff_old)
        coeff_old.append(coeff_old[-1])

    #initial_c = ([1. for _ in range(len(L2))])
    if (len(L2) < 140 + c * 100):
        ft = "double"
    elif (len(L2) < 200 + c * 100):
        ft = "dd"
    else:
        ft = "mpfr"
        PRE = round(len(L2) / 1.65 / 10.0) * 10
        FPLLL.set_precision(PRE)
        print("## setting floating mpfr ", PRE)

    #print ("# Input precost  = ", precost)
    pr = Pruning.Pruner(R2, precost, [L2], targetprob, \
                            metric=Pruning.EXPECTED_SOLUTIONS, \
                            #metric=Pruning.PROBABILITY_OF_SHORTEST, \
                            flags=Pruning.ZEALOUS, float_type=ft)
    c_new = pr.optimize_coefficients(coeff_old)
    allcost = pr.repeated_enum_cost(c_new)
    singlecost, details = pr.single_enum_cost(c_new, True)
    succprob = pr.measure_metric(c_new)

    if (succprob >= 1.0):
        allcost += precost

    # print result
    #util_print_output_info (pr, c_new)

    return allcost, singlecost, succprob, c_new


def main_enum_preprocess(M, n, k_alpha, k, alpha, c):

    # 1. a heavy preprocessing so that the expected num of solution is > 1
    diff = 30
    kk = k - diff
    print("# n: %d, k_a: %d, k: %d, alpha: %f (LLL done)" % (n, k_alpha, k,
                                                             alpha))
    flags = BKZ.AUTO_ABORT | BKZ.MAX_LOOPS  #|BKZ.VERBOSE

    precost = 0
    trials = 0
    while (1):

        print("# start BKZ preprocessing beta = %d" % kk)
        par = BKZ.Param(
            kk, strategies=BKZ.DEFAULT_STRATEGY, max_loops=10, flags=flags)

        bkz = BKZ2(M)
        _ = bkz(par)

        pretime = float(bkz.trace.data["walltime"])
        precost += pretime * 20000000.

        # get L2
        M.update_gso()
        input_L2 = [M.get_r(i, i) for i in range(k_alpha)]

        # get GH
        logr = [log(M.get_r(i, i)) / 2.0 for i in range(k_alpha)]
        log_GH = return_log_GH_simple(logr)
        input_R2 = (e**log_GH)**2 * alpha**2
        input_targetprob = 1.0
        coeff = [1. for x in range(k_alpha)]

        allcost, singlecost, succprob, coeff = \
          optimize_cost_prob (input_L2, input_R2, precost, input_targetprob, coeff, c)

        print ("# [BKZ done] |b1| = %f, R2 = %f, precost = %f, succ = %f, sglcost = %f, allcost = %f" % \
                   (log(input_L2[0])/2.0, log(input_R2)/2.0, precost, succprob, singlecost, allcost))

        trials += 1
        if (succprob > 0.1):
            return 1, input_R2, coeff, singlecost, succprob

        if (trials > 50):
            return 0, input_R2, coeff, singlecost, succprob

        kk += 4


def main_enum(n, k_alpha, k, alpha, c, seed):

    FPLLL.set_random_seed(seed)

    # 1. initialize and LLL
    A = IntegerMatrix.random(n, "qary", bits=n, k=n // 2)
    A = LLL.reduction(A)
    M = GSO.Mat(A)
    M.update_gso()

    # 2. find a good preprocessing such that exp_solution > 1 and radius > current b1
    flag, input_R2, coeff, est_sglcost, succprob = \
      main_enum_preprocess (M, n, k_alpha, k, alpha, c)

    # 3. output some info before enum
    M.update_gso()
    input_L2 = [M.get_r(i, i) for i in range(k_alpha)]
    logr = [log(M.get_r(i, i)) / 2.0 for i in range(k_alpha)]
    log_GH = return_log_GH_simple(logr)
    print ("# [EnumInfo] |b1| = %f, R2 = %f, succ = %f, sglcost = %f" % \
               (log(input_L2[0])/2.0, log(input_R2)/2.0, succprob, est_sglcost))

    # 4. finally start enumeration
    Eobj = Enumeration(M)
    try:
        solution, max_dist = Eobj.enumerate(
            0, k_alpha, input_R2, 0, pruning=coeff)[0]
        sglcost = Eobj.get_nodes()
        #print ("done --> nodes is", Eobj.get_nodes())
        ratio = (est_sglcost / sglcost) / succprob
        ratio2 = (est_sglcost / sglcost)
        print ("# [Enumd GOOD] sglcost %f, est_sglcost %f, ra1 %f, ra2 %f, |b1| %f" \
                   % (sglcost, est_sglcost, ratio, ratio2, log(solution)/2.0))
    except EnumerationError:
        sglcost = Eobj.get_nodes()
        ratio = (est_sglcost / sglcost) / succprob
        ratio2 = (est_sglcost / sglcost)
        print ("# [Enumd BAD] sglcost %f, est_sglcost %f, ra1 %f, ra2 %f" \
                   % (sglcost, est_sglcost, ratio, ratio2))

    return


def k_alphaf(k, alpha):
    """
    Return k_α.
    """
    rhf = rhff(k, 1.0)
    for k_alpha in range(k, 100 * k, 1):
        if rhff(k_alpha, alpha) <= rhf:
            return k_alpha
    raise ValueError("k_α failed for {k},{alpha:.2f}".format(k=k, alpha=alpha))


def main():

    # parse argument
    args = parse_options()
    print("###################################### ")
    print("# [Info]: in enumeration mode")
    print("# [Args] k: %s" % args.k)
    print("# [Args] alpha: %s" % args.alpha)
    print("# [Args] c: %s" % args.c)
    print("# [Args] seed: %s" % args.seed)
    print("###################################### ")

    # start process
    k = int(args.k)
    seed = int(args.seed)
    alpha = float(args.alpha)
    c = float(args.c)

    # get k_alpha and n
    k_alpha = k_alphaf(k, alpha)
    n = round((1 + c) * k_alpha)
    print("# n = %d, k_a = %d" % (n, k_alpha))
    main_enum(n, k_alpha, k, alpha, c, seed)

    return


# auxililary
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k')
    parser.add_argument('-alpha')
    parser.add_argument('-c')
    parser.add_argument('-seed')
    args = None
    args = parser.parse_args()
    if args is None:
        exit(0)
    return args


# python main_enum.py -k 80 -alpha 1.00 -c 0.0 -seed 1
if __name__ == '__main__':
    main()
