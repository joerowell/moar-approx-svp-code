# -*- coding: utf-8 -*-

from util import *
from simulator import Simulator as Base_Simulator


# class
class BKZ_Simulator_CN(Base_Simulator):

    # init
    def __init__(self, gso_bi, l_format=2, verbose=0):
        Base_Simulator.__init__(self, gso_bi, l_format)
        self.not_changed = True
        self.l_old = [gso_bi[i] / l_format for i in range(self.n)]
        self.verbose = verbose

    # run with params
    def __call__(self, params, min_row=0, max_row=-1):

        # (log of) multiplier in front of det^(1/n)
        self.c = []
        if (0):
            for d in range(1, params.block_size + 1):
                vol = sphere_vol_exact(d, 1)
                extra_common = 0
                common = -log(vol) / d + extra_common
                self.c.append(common)
        else:
            for d in range(1, params.block_size + 1):
                log_spherevol = SPHERE_LOGVOL[d]
                extra_common = 0
                common = -log_spherevol / d + extra_common
                self.c.append(common)

        # determinant of the input. should be a constant
        self.det = sum(self.l)

        # start tours
        i = 0
        while True:

            # initial HF
            hf = return_Hermite_factor(self.l, 0, self.n)
            if (self.verbose):
                print("# Chen-Nguyen BKZ tour ", i, ", hermite = ", hf,\
                  ", det = ", sum(self.l))

            # start a tour
            self.tour(params, min_row, max_row)
            i += 1

            # abort if no change at all
            if (not self.changed_in_this_tour):
                if (self.verbose):
                    print("# Abort happends since no change at all!")
                break

            #print self.l
            if (self.verbose):
                pretty_print_l(self.l, self.l_format)

            # check if returns
            if params.block_size >= len(self.l):
                break

            # check if max_loop
            if (params.flags & BKZ.MAX_LOOPS) and i >= params.max_loops:
                break
        return

    # each tour
    def tour(self, params, min_row=0, max_row=-1):
        if max_row == -1:
            max_row = self.n

        # not changed in the tour at all. this may be changed in svp_redcution()
        self.changed_in_this_tour = False

        # step 1: first n-45 blocks
        for start in range(min_row, max_row - 45):
            bs = min(params.block_size, max_row - start)
            self.svp_reduction(start, bs)

        # step 2: the last 45-block (use HKZ experiments)
        self.svp_reduction_last45(min_row, max_row)

        # deepcopy
        self.l_old = deepcopy(self.l)

        return

    def svp_reduction(self, start, bs):

        end = start + bs
        GH = return_log_GH(self.l_old, self.l, start, end, self.c)

        # updated ever previously?
        old = self.l_old[start]
        if (self.not_changed):
            if (GH < self.l_old[start]):
                self.l[start] = GH
                self.not_changed = False
        else:
            self.l[start] = GH

        # changed in this tour?
        if (abs(self.l[start] - old) > 1e-6):
            self.changed_in_this_tour = True

        return

    def svp_reduction_last45(self, min_row, max_row):

        # else change
        logdet = sum(self.l_old[:max_row]) - sum(self.l[:max_row - 45])
        K = list(range(max_row - 45, max_row))
        for k, r in zip(K, self.rk):
            self.l[k] = logdet / 45 + r

        return
