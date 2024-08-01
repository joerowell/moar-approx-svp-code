# -*- coding: utf-8 -*-

## Did you know that the above is useless in Python 3? (I only learned this today!)
## https://docs.python.org/3/howto/unicode.html#the-string-type


## I hate globals, but here it is useful because it means that you have to read this header
## This is just the scaling factor for the curve to convert cycles into nodes.
## Convert cycles into nodes for (say) LLL is kinda weird, because we're equating the running
## time of two distinct algorithms.
## However, if you think about enumeration as repeated applications of Babai's nearest plane plus
## some other overhead it makes more sense :)

## Anyway: here we stick with the convention that the one node is approximately 64 cycles.
## You can recompute this for your machine using fpylll goodness (Well, the set_mdc file of the pruner)
g_node_scaling_factor = 64
import numpy as np


def process_in(names: [str]):
    # Define a dictionary for the "," pairs
    # The goal here is just to collapse all of the results into one big result set.
    scores = {}

    for f in names:
        file_handle = open(f, "r")
        for line in file_handle:
            ## We now have a line, so we need to split it across the ","
            split_list = line.split(",")
            ## Check to make sure we don't introduce bad results
            assert len(split_list) == 2
            key = int(split_list[0])
            val = int(split_list[1])

            ## We divide through by the cycles here to make sure we've done it for all inputs
            if key in scores:
                scores[key].append(val / g_node_scaling_factor)
            else:
                scores[key] = [val / g_node_scaling_factor]
    return scores


if __name__ == "__main__":

    # Yes, supplying these like this is definitely hacky.
    file_list = [
        "lll_cost0",
        "lll_cost1",
        "lll_cost2",
        "lll_cost3",
        "lll_cost4",
        "lll_cost5",
        "lll_cost6",
        "lll_cost7",
        "lll_cost8",
        "lll_cost9",
    ]

    out = process_in(file_list)

    # We're now going to write to lll_costs.csv
    # The header is going to be "rank, cost0,...,costN, average"
    out_file = open("lll_costs.csv", "w")
    header = "rank,"
    for i in range(0, len(file_list)):
        header += "cost" + str(i) + ","
    header += "average,\n"
    out_file.write(header)

    # We now are going to write the costs into a nice, continuous file
    for key in out:
        # To begin, we'll build a string that starts with the name
        line = str(key) + ","
        average = 0

        if len(out[key]) != 10:
            print("For " + str(key) + " there's not 10 entries!")

        # Yes there's probably a more Pythonic way to write this
        for i in out[key]:
            line += "{:.5f}".format(np.log2(i)) + ","
            average += i

        # Compute the average and append
        line += "{:.5f}".format(np.log2(average / len(out[key]))) + ","
        out_file.write(line + "\n")
