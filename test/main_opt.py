import os, sys, pickle, numpy, platform, argparse, math, time
import numpy as np
from fpylll import IntegerMatrix, GSO, LLL, Pruning, FPLLL
# other packages
#from matplotlib import pyplot as plt
from util import *

OPTION_METHOD = 1
VERBOSE = 0


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


# does the optimization
def optimize_cost_prob(L2, R2, precost, targetprob, coeff_old):

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
    if (len(L2) < 140):
        ft = "double"
    elif (len(L2) < 200):
        ft = "dd"
    else:
        ft = "mpfr"
        PRE = round(len(L2) / 1.65 / 10.0) * 10
        FPLLL.set_precision(PRE)
        print("## setting floating mpfr ", PRE)

    #print ("# Input precost  = ", precost)


    pr = Pruning.Pruner(R2, precost, [L2], targetprob, \
                            metric=Pruning.EXPECTED_SOLUTIONS, \
                            flags=Pruning.GRADIENT, float_type=ft)
    c_new = pr.optimize_coefficients(coeff_old)
    allcost = pr.repeated_enum_cost(c_new)
    singlecost, details = pr.single_enum_cost(c_new, True)
    succprob = pr.measure_metric(c_new)

    #print (c_new)

    if (succprob >= 1.0):
        allcost += precost

    # print result
    util_print_output_info(pr, c_new)

    return allcost, singlecost, succprob, c_new


def main_opt(k, alpha, coeff_old, c):

    n = round((1 + c) * k)
    #print ("# [Info] n = (1+c)*k = ", n)

    # BKZ-k preprocessed
    if (OPTION_METHOD == 1):
        log_L = get_BKZ_GH_div2(n, k, tailmode=1)
    else:
        log_L = get_GSA(n, k, verbose=0)

    # only first k such entries and normalize (for easier debugging)
    log_L = log_L[0:k]
    log_L = normalize_GSO_unitary(log_L)

    # compute logGH of first k entries
    log_R = return_log_GH_simple(log_L)
    inputR = (e**log_R)**2

    # un-normalize
    L2 = [(e**x)**2 for x in log_L]

    input_R2 = inputR * alpha**2
    input_targetprob = 1.0
    input_precost = 1.0 * n**3

    allcost, singlecost, succprob, coeff_new = optimize_cost_prob (L2, input_R2, input_precost, \
                                                                input_targetprob, coeff_old)

    return allcost, singlecost, succprob, coeff_new


def main():

    # parse argument
    args = parse_options()
    print("###################################### ")
    print("# [Info]: in optimization mode")
    print("# [Args] k: %s" % args.k)
    print("# [Args] alpha: %s" % args.alpha)
    print("# [Args] trials: %s" % args.num)
    print("# [Args] c: %s" % args.c)
    print("###################################### ")

    # start process
    k = int(args.k)
    num = int(args.num)
    alpha = float(args.alpha)
    c = float(args.c)
    assert (alpha >= 1.0)

    n = round((1 + c) * k)
    #print ("# [Info] n = (1+c)*k = ", n)

    # this ceoff is resued when num > 1. We are enumerating over [1, k] in all cases.
    coeff = ([1. for _ in range(k)])

    for kk in range(k, k + num):

        big_alpha = 100
        while (big_alpha <= 130):
            each_alpha = big_alpha / 100.0
            #print (kk, each_alpha)

            allcost, singlecost, succprob, coeff = main_opt(
                kk, each_alpha, coeff, c)

            if (succprob > 1.0):
                print("# k =", kk, " alpha =", each_alpha, " n =", n,
                      " cost =", log(allcost) / log(2.0), " sglcst =",
                      log(singlecost) / log(2.0), ", prob =", succprob, " BAD")
            else:
                print("# k =", kk, " alpha =", each_alpha, " n =", n,
                      " cost =", log(allcost) / log(2.0), " sglcst =",
                      log(singlecost) / log(2.0), ", prob =", succprob)

            big_alpha += 5

    return


# auxililary
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k')
    parser.add_argument('-alpha')
    parser.add_argument('-num')
    parser.add_argument('-c')
    args = None
    args = parser.parse_args()
    if args is None:
        exit(0)
    return args


# python main_opt.py -k 80 -num 1 -c 0.0 -alpha 1.00
if __name__ == '__main__':
    main()
