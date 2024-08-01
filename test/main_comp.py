import os, sys, pickle, numpy, platform, argparse, math, time
import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
from util import *

OPTION_METHOD = 1
VERBOSE = 0

_ = FPLLL.set_precision(200)


def util_print_output_info(pr, c):
    allcost = pr.repeated_enum_cost(c)
    singlecost, details = pr.single_enum_cost(c, True)
    succprob = pr.measure_metric(c)

    # optimized coefficients
    if (VERBOSE):
        print("# Output c = ")
        print(c)
        print("# Output    all_cost  = ", allcost)
        print("# Output single_cost  = ", singlecost)
        print("# Output single_prob  = ", succprob)

    return allcost, singlecost, succprob


# just compute, no optimization at all.
def compute_cost_prob(L2, R2, precost, targetprob, rho):

    if (VERBOSE):
        print("# Mode: no optimiztion ")
        print("# Input R^2  = ")
        print(L2)
        print("# Input   bound  = ", R2)
        print("# Input precost  = ", precost)
        print("# Input  target  = ", targetprob)

    pr = Pruning.Pruner(R2, precost, [L2], targetprob, \
                            metric=Pruning.EXPECTED_SOLUTIONS, \
                            flags=Pruning.ZEALOUS, float_type="mpfr")

    k = len(L2)

    # input special c
    c = ([1. for _ in range(k)])
    for i in range(k // 2, k):
        c[i] = sqrt(rho)

    # print result
    allcost, singlecost, succprob = util_print_output_info(pr, c)

    return allcost, singlecost, succprob


def main_comp(k, alpha, rho):

    # BKZ-k preprocessed
    if (OPTION_METHOD == 1):
        log_L = get_BKZ_GH_div2(k, k, tailmode=1)
    else:
        log_L = get_GSA(k, k, verbose=0)
    # un-normalize
    L2 = [(e**x)**2 for x in log_L]
    # this is GH!
    #log_R = return_log_GH_simple (log_L)
    #inputR = (e**log_R)**2

    input_R2 = L2[0] * alpha**2
    input_targetprob = 0.99
    input_precost = 1.0 * k**3
    #util_optimize_coefficients (L2, input_R2, input_precost, input_targetprob)

    allcost, singlecost, succprob = compute_cost_prob(
        L2, input_R2, input_precost, input_targetprob, rho)

    return allcost, singlecost, succprob


def main():

    # parse argument
    args = parse_options()
    print("###################################### ")
    print("# [Args] k: %s" % args.k)
    print("# [Args] alpha: %s" % args.alpha)
    print("# [Args] rho: %s" % args.rho)
    print("###################################### ")

    # start process
    k = int(args.k)
    alpha = float(args.alpha)
    rho = float(args.rho)
    assert (alpha >= 1.0)
    assert (rho <= 1.0)

    # main functiona
    for kk in range(k, k + 1):
        big_alpha = 100
        while (big_alpha <= 130):
            each_alpha = big_alpha / 100.0

            allcost, singlecost, succprob = main_comp(kk, each_alpha, rho)
            print("# k =", kk, " alpha =", each_alpha,
                  " cost =", log(allcost) / log(2.0), " sglcst =",
                  log(singlecost) / log(2.0), " prob =", succprob)
            if (succprob > 1.0):
                print("# Error, succprob too large. Not applicable.")

            big_alpha += 5

    return


# auxililary
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k')
    parser.add_argument('-alpha')
    parser.add_argument('-rho')
    args = None
    args = parser.parse_args()
    if args is None:
        exit(0)
    return args


# python main_comp.py -k 100 -rho 0.01 -alpha 1.00
if __name__ == '__main__':
    main()
