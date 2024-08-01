# -*- coding: utf-8 -*-

# self-defined stuff
from util import *


# class
class Simulator(object):
    """
    An implementation of the simulator
    """

    # init
    def __init__(self, gso_bi, l_format=2):
        if gso_bi is None:
            raise TypeError("L must be a list of initial GSO's")

        # some variables
        self.n = len(gso_bi)
        self.l_format = l_format
        self.l = [gso_bi[i] / l_format for i in range(self.n)]
        self.touched = [True] * self.n
        self.new_touched = [False] * self.n
        self.changed_in_this_tour = False
        self.rk = HKZ45_LN
        self.verbose = False

    # run with params
    def __call__(self, params, min_row=0, max_row=-1):
        return

    # each tour
    def tour(self, params, min_row=0, max_row=-1):
        return

    def svp_reduction(self, start, bs):
        return

    def svp_reduction_last45(self, min_row, max_row):
        return
