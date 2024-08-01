from hwcounter import Timer, count, count_end
from concurrent.futures import ThreadPoolExecutor
from fpylll import IntegerMatrix, LLL, BKZ, GSO, Pruning
from fpylll.fplll.bkz_param import Strategy
from fpylll.fplll.pruner import PruningParams
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

import math


def linear_pruning_strategy(block_size, level):
    ## This code was stolen from:
    ## https://github.com/fplll/fpylll/blob/master/docs/example-custom-pruning.rst

    if level > block_size - 1:
        raise ValueError
    if block_size < 5:
        raise ValueError

    ## However, we don't need to import BKZ etc because we already have it
    preprocessing = 3
    strategies1 = [Strategy(i) for i in range(6)]
    for b in range(6, block_size + 1):
        if block_size == b:
            pr = PruningParams.LinearPruningParams(block_size, level)
            s = Strategy(b, [preprocessing], [pr])
        else:
            s = Strategy(b, [preprocessing])
        strategies1.append(s)
    param = BKZ.Param(block_size=block_size, strategies=strategies1)
    return param


def reduce_basis(A, params):
    """
    Given an input lattice A, first SVP reduce the lattice and then record the cost of running LLL.
    :param A: the input lattice
    :param params: the BKZ parameters
    :return: the time taken to LLL reduce the basis after SVP reduction, in cycles
    """

    block_size = min(math.ceil(len(A[0])/2), 50)
    M = GSO.Mat(A)
    bkz = BKZ2(M)

    ## Finally, reduce and count
    A_bkz = bkz.svp_reduction(0, block_size=block_size, params=params)
    start = count()
    LLL.Reduction(M)
    return count_end() - start


def pump(rank, threads, nr):

    """
    Produce timings for running LLL on an SVP-reduced lattice basis.
    
    This function works in the following fashion: we sample 'nr' random full-rank 
    q-ary lattices with rank `rank`.  Once sampled, we SVP reduce each lattice using BKZ2, and then time
    how long the LLL reduction takes on these SVP-reduced lattices.
    This work is divded amongst `thread` many threads, using the Pool class.
    
    The timings for each of these reductions is carried out using the CPUs hardware counters,
    via the hwcounter module. At the end, all timings are returned in a list.
    :param rank: the lattice rank
    :param threads: the number of threads to use
    :param nr: the number of lattice bases to sample and process.
    """

    lattices = [IntegerMatrix.random(rank, "qary", bits=30, k=3) for _ in range(rank)]

    # Set-up the worker threads to process things in parallel
    # The goal here is to relatively evenly divide the bases in lattice across
    # n-many results
    # We do this using Python's handy ThreadPoolExecutor

    timings = []
    executor = ThreadPoolExecutor(threads)
    futures = []

    ## This choice of level is really arbitrary: the point here is to force linear pruning
    block_size = math.ceil(rank / 2)
    params = linear_pruning_strategy(block_size, level=math.ceil(block_size / 2))

    for i in range(0, nr):
        futures.append(executor.submit(reduce_basis, lattices[i], params))

    ## With these processing, we iterate over the list and continuously check if the work is done
    ## To prevent race conditions, we make sure to move into a new list (to prevent splitting the list
    ## in an inconsistent state).

    while len(futures) != 0:
        # Create a temp list and move the futures
        temp = []
        for i in range(len(futures)):
            if futures[i].done():
                timings.append(futures[i].result())
            else:
                temp.append(futures[i])

        futures = temp
    return timings


if __name__ == "__main__":
    ### Set the number of threads globally
    ### For sanity we set nr_threads = nr_bases
    ### This corresponds to one job per thread
    nr_threads = 10
    nr_bases = nr_threads

    low_index = 50
    high_index = 150

    filename = "lll_costs_timings.csv"

    out_file = open(filename, "w")
    header = "rank,"
    for i in range(0, nr_bases):
        header += "cost" + str(i) + ","
    header += "average,\n"
    out_file.write(header)
    out_file.close()

    for rank in range(low_index, high_index):
        # Hopefully this will help you know where we are
        print("Current rank:" + str(rank))
        timings = pump(rank, nr_threads, nr_bases)
        # Open at the end of each loop
        # forces the vm to dump the lines : we can then recover from crashes more easily.
        out_file = open(filename, "a")
        line = str(rank) + ","
        total = 0

        for i in timings:
            line += str(i) + ","
            total += i

        line += str(total / len(timings))
        out_file.write(line + "\n")
        out_file.close()
