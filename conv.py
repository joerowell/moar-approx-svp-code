# -*- coding: utf-8 -*-
import csv
import pickle
import os
from collections import OrderedDict


import math
from utils import read_csv


def preproc_fit(
    filename: ".csv filename to fit",
    low_index: "start fitting at this index" = 50,
    high_index: "stop fitting at this index (exclusive)" = 100,
    columns: "csv columns to select" = ("d", "dprime"),
):
    """
    """
    from scipy.optimize import curve_fit

    def f(x, a, b):
        return a * x + b

    X, Y = read_csv(filename, columns=columns, read_range=range(low_index, high_index))
    r = tuple(curve_fit(f, X, Y)[0])
    print(r)
    return r


def prob_fit(
    filename: ".csv filename to fit",
    low_index: "start fitting at this index" = 50,
    high_index: "stop fitting at this index (exclusive)" = 100,
    columns: "csv columns to select" = ("d", "probability"),
):
    """
    """
    from scipy.optimize import curve_fit

    def f(x, a, b):
        return a * x + b

    X, Y = read_csv(
        filename, columns=columns, read_range=range(low_index, high_index), ytransform=lambda y: math.log(y, 2)
    )
    r = tuple(curve_fit(f, X, Y)[0])
    print(r)
    return r


def cost_simulation(
    input_filename: ".sobj filename to convert", output_filename_template: "template for output filenames" = None
):
    """
    Convert output of ``./cli cost_simulation`` to ``.csv`` suitable to produce LaTeX plots and tables.
    """
    if output_filename_template is None:
        output_filename_template = "../data/approx-hsvp-simulations,qary,{alpha},{c:.2f},{preproc_strategy}{tag}.csv"

    costs = pickle.load(open(input_filename, "rb"))

    costs_transposed = dict()

    headers = None

    if not input_filename.endswith(".sobj"):
        raise ValueError("{input_filename} not supported".format(input_filename=input_filename))
    input_filename = input_filename.replace(".sobj", "")
    input_filename = os.path.basename(input_filename).split(",")
    c = input_filename[3]
    preproc_strategy = input_filename[4]
    if len(input_filename) > 5:
        tag = input_filename[5]
    else:
        tag = ""

    c = float(c)
    if preproc_strategy != "best":
        preproc_strategy = float(preproc_strategy)

    for k in costs:
        for alpha in costs[k]:
            if alpha not in costs_transposed:
                costs_transposed[alpha] = OrderedDict()
            costs_transposed[alpha][k] = costs[k][alpha]
            headers = costs[k][alpha].keys()

    headers_ = []
    for header in headers:
        header = header.replace("_", " ")
        if header == "total cost":
            headers_.append("log(total cost)")
        headers_.append(header)
        if header == "total cost":
            headers_.append("speedup")
            headers_.append("log(speedup)")
    headers = headers_

    ret = OrderedDict()

    for alpha in costs_transposed.keys():
        fn = output_filename_template.format(
            alpha="%.2f" % alpha if isinstance(alpha, float) else alpha,
            c=c,
            tag=",%s" % tag if tag else "",
            preproc_strategy="%.2f" % preproc_strategy if isinstance(preproc_strategy, float) else preproc_strategy,
        )

        ret[alpha] = fn

        with open(fn, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(headers)

            for k, row in costs_transposed[alpha].items():
                row_ = [None] * len(headers)
                for key, value in row.items():
                    if key == "total_cost":
                        row_[headers.index("log(total cost)")] = math.log2(float(value))
                    row_[headers.index(key.replace("_", " "))] = value
                row_[headers.index("speedup")] = costs[k][1.0]["total_cost"] / costs[k][alpha]["total_cost"]
                row_[headers.index("log(speedup)")] = math.log(row_[headers.index("speedup")], 2)
                csvwriter.writerow(row_)
    return ret


def call_sobj_csv(filename: ".sobj filename to convert"):
    """
    Convert output of ``./call.py`` to ``.csv`` suitable to produce LaTeX plots and tables.
    """
    costs = pickle.load(open(filename, "rb"))

    with open(filename.replace(".sobj", ".csv"), "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(["k alpha", "k pre", "total time", "total cost"])

        for beta, cost in costs.items():
            try:
                dprime = cost["strategy"].preprocessing_block_sizes
                dprime = dprime[0] if dprime else 2
            except KeyError:
                if "betaprime" in cost:
                    dprime = cost["betaprime"]
                elif "preprocessing block size" in cost:
                    dprime = cost["preprocessing block size"]
                elif "preprocessing block sizes" in cost:
                    dprime = cost["preprocessing block sizes"][0]
                else:
                    raise KeyError
            dprime = int(dprime)
            csvwriter.writerow([beta, dprime, cost["total time"], round(cost["#enum"])])


def verify_csv(filename):
    input_data = pickle.load(open(filename, "rb"))

    headers = ["i"]
    output_data = dict()

    for key in input_data.keys():
        if not key.startswith("r_"):
            continue
        headers.append(key)
        for i, r_ in enumerate(input_data[key]):
            if i not in output_data:
                output_data[i] = OrderedDict()
            output_data[i][key] = math.log(r_, 2)

    with open(filename.replace(".sobj", ".csv"), "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for i in sorted(output_data.keys()):
            data = [i]
            for key in output_data[i]:
                data.append(output_data[i][key])
            writer.writerow(data)


# def run_estimate(LC):
#     from estm import estimate, gh
#     from sage.all import var, log, find_root

#     x, ca, cb = var("x", "ca", "cb")
#     cost, crossovers = estimate(c=0.00)
#     LC["/formulas/over2e/all"] = "\\frac{{k \\log k}}{{2\\me}} - {cost1:.3f}\\,k + {cost2:.2f}".format(
#         cost1=-cost[ca], cost2=cost[cb]
#     )
#     LC["/functions/over2e/all"] = "0.1839*x*log2(x) - {cost1:.3f}*x + {cost2:.2f}".format(
#         cost1=-cost[ca], cost2=cost[cb]
#     )
#     for alpha, rhf, k, kalpha in crossovers:
#         LC["/crossovers/over2e/alpha{alpha:3d}/rhf".format(alpha=round(100 * alpha))] = "{rhf:.4f}".format(rhf=rhf)
#         LC["/crossovers/over2e/alpha{alpha:3d}/kalpha".format(alpha=round(100 * alpha))] = "{kalpha}".format(
#             kalpha=kalpha
#         )
#         LC["/crossovers/over2e/alpha{alpha:3d}/k".format(alpha=round(100 * alpha))] = "{k}".format(k=k)

#     cost, crossovers = estimate(c=0.25)
#     LC["/formulas/over8/all"] = "\\frac{{k \\log k}}{{8}} - {cost1:.3f}\\,k + {cost2:.2f}".format(
#         cost1=-cost[ca], cost2=cost[cb]
#     )
#     LC["/functions/over8/all"] = "0.125*x*log2(x) - {cost1:.3f}*x + {cost2:.2f}".format(cost1=-cost[ca], cost2=cost[cb])

#     k = ceil(find_root((0.5 * (0.125 * x * log(x, 2) + cost[ca] * x + cost[cb]) - 0.265 * x), 50, 1000))

#     LC["/crossovers/over8/quantum/k"] = k
#     LC["/crossovers/over8/quantum/km1"] = k - 1
#     LC["/crossovers/over8/quantum/rhf"] = "%.4f" % float(gh(k) ** (1 / (k - 1)))

#     for alpha, rhf, k, kalpha in crossovers:
#         LC["/crossovers/over8/alpha{alpha:3d}/rhf".format(alpha=round(100 * alpha))] = "{rhf:.4f}".format(rhf=rhf)
#         LC["/crossovers/over8/alpha{alpha:3d}/kalpha".format(alpha=round(100 * alpha))] = "{kalpha}".format(
#             kalpha=kalpha
#         )
#         LC["/crossovers/over8/alpha{alpha:3d}/k".format(alpha=round(100 * alpha))] = "{k}".format(k=k)

#     return LC
