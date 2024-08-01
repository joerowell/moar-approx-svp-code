#from util_data import *
from util import *

from mpmath import *
mp.dps = 20

def enum_cost_GH(log_bi, log_R):
    """
    Refers to GNR and HS paper: estimate the number
    of nodes for the full enumeration given log_bi and log_R.
    This is based on Gaussian heuristic.

    INPUT HAS to be log||b_i^*|| without Square!
    """

    # gaussian heuristic minimum
    n = len(log_bi)

    # start estimation
    H = -1e100
    log_prod = 0
    for k in range (0, n):
        log_prod = log_prod + log_bi[n-k-1]
        log_numerator = (k+1) * log_R + SPHERE_LOGVOL[k+1]
        #log_numerator = (k+1) * log_R + log_sphere_vol_exact (k+1, 1)        
        log_cost = max(log_numerator - log_prod - 0.693147180559945, 0)

        if (log_cost > H):
            H = log_cost

    return H  # log(max_cost)


def enum_cost_GH_all(log_bi, log_R):
    """
    Same as above but full enumeration (not max of level)
    """

    # gaussian heuristic minimum
    n = len(log_bi)

    # start estimation
    H = mpf(0.0)
    log_prod = 0
    for k in range (0, n):
        log_prod = log_prod + log_bi[n-k-1]
        log_numerator = (k+1) * log_R + SPHERE_LOGVOL[k+1]
        log_cost = max(log_numerator - log_prod - 0.693147180559945, 0)
        H = H + e**log_cost

    return log(H)
