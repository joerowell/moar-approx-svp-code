##########################
# some auxiliary functions
##########################

import sys
from copy import deepcopy
from random import expovariate
from math import pi,e,log,sqrt,lgamma
from scipy.special import zeta
from scipy.special import gamma
import numpy as np
from fpylll import BKZ
import random

from util_data import *


def sphere_vol_exact (n, R):
    return pi**(n/2.0) * R**n / gamma(n/2.0+1.0)


def log_sphere_vol_exact (n, R):
    return (n/2.0)*log(pi) + n*log(R) - lgamma(n/2.0+1.0)


# sphere_vol_exact_test (30)
def sphere_vol_exact_test (n):
    for i in range(2, n):
        voln = sphere_vol_exact (i, 1)
        print("# n: ", i, ", vol: ", voln, ", 1/vol^(1/n): ", \
            1/voln**(1/float(i)))


# (log of) multiplier in front of det^(1/n)
def return_log_GH_multiplier (bs):
    c = []
    for d in range(1, bs+1):
        #vol = sphere_vol_exact (d, 1)
        log_spherevol = SPHERE_LOGVOL[d]
        extra_common = 0
        common = -log_spherevol/d + extra_common
        c.append(common)
    return c


# return GH (deterministic). l1 is the old; l2 is the new
def return_log_GH (l1, l2, start, end, c, dual=False):
    bs = end - start
    logdet = sum(l1[:end]) - sum(l2[:start])
    if (dual):
        GH_0 = logdet/bs - c[bs-1]
    else:
        GH_0 = logdet/bs + c[bs-1]
    return GH_0


# return b1 given rhf. l1b is the old; l2 is the new
def return_log_b1_approx (l1, l2, start, end, rhf):
    bs = end - start
    logdet = sum(l1[:end]) - sum(l2[:start])
    GH_0 = logdet/bs + log(rhf)*bs
    return GH_0


# return GH (deterministic)
def return_log_GH_simple (l):
    logdet = sum(l)
    n = len(l)
    log_spherevol = SPHERE_LOGVOL[n]
    c = -log_spherevol/n
    b1 = logdet/n + c
    return b1


def return_random_sphere(dim):
    vec = [random.gauss(0, 1) for i in range(dim)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]


def return_random_ball(dim):
    """
    vec = return_random_sphere(dim)
    vec_norm = [x * uu for x in vec]
    mag = sum(x**2 for x in vec_norm) ** .5
    """
    u = random.uniform(0, 1)
    uu = u**(1/float(dim))
    return uu


# return sphere point length on unit ball
def return_log_random_point (radius, dim):
    ratio = return_random_ball(dim)
    return ratio * radius


def return_Hermite_factor (l, start, end):
    bs = end - start
    logdet = sum(l[:end]) - sum(l[:start])
    hf = (l[start] - logdet / bs) / bs
    return (e)**hf


def need_renew_this_block (old_touched, start, endp1):
    to_be_changed = False
    for k in range(start, endp1):
        to_be_changed = (to_be_changed or old_touched[k])
    return to_be_changed


def pretty_print_l (l, l_format):
    print(str(["{0:0.3f}".format(x*l_format) for x in l]).replace("'", ""))


def get_current_slope (l, start, end):
    n = end - start
    i_mean = (n - 1) * 0.5 + start
    x_mean = 0
    for i in range(start, end):
        x_mean += l[i]
    x_mean /= n
    v1 = 0
    v2 = 0
    for i in range(start, end):
        v1 += (i - i_mean) * (l[i] - x_mean)
        v2 += (i - i_mean) * (i - i_mean)
    return v1 / v2


def lll_cost(d):
    return 3*log(d,2.0)


def get_HKZ_GH_core (n):
    assert(n<=100000)
    logl = [0.0] * n
    logl[n-1] = 0.0
    logdet = 0.0
    for k in range(n-2, -1, -1):
        d = n - k
        log_spherevol = SPHERE_LOGVOL[d]
        logl[k] = (logdet - log_spherevol) / (d-1.0)
        logdet += logl[k]
    nor_logl = normalize_GSO_unitary(logl)
    logl2 = [2 * x for x in nor_logl]
    return logl2


def get_HKZ_GH_core_slower (n):
    logl = [0.0] * n
    logl[n-1] = 0.0
    logdet = 0.0
    for k in range(n-2, -1, -1):
        d = n - k
        log_spherevol = log_sphere_vol_exact (d, 1)
        logl[k] = (logdet - log_spherevol) / (d-1.0)
        logdet += logl[k]
    nor_logl = normalize_GSO_unitary(logl)
    logl2 = [2 * x for x in nor_logl]
    return logl2


def get_BKZ_GH_div2 (n, beta, tailmode=1):
    l2 = get_BKZ_GH(n, beta, tailmode)
    l = [x/2.0 for x in l2]
    return l


def get_BKZ_GH(n, beta, tailmode=1):
    assert(beta >= 45) # the behavior of beta<45 is hard
    assert(n >= beta)
    if (tailmode == 0):
        logl2_tail = get_HKZ_GH_core_slower(45)
    else:
        logl_tail = HKZ45_LN
        logl2_tail = [2 * x for x in logl_tail]
    logl = [0.0] * n
    logl_tail = [x/2.0 for x in logl2_tail]
    logl[-45:] = logl_tail
    logdet = sum(logl_tail)
    for k in range(n-46, -1, -1):
        d = min(n-k, beta)
        log_spherevol = SPHERE_LOGVOL[d]
        if (n > k+beta):
            logdet -= logl[k+beta]
        logl[k] = (logdet - log_spherevol) / (d-1.0)
        logdet += logl[k]
    nor_logl = normalize_GSO_unitary(logl)
    logl2 = [2 * x for x in nor_logl]
    return logl2


def get_LLL_GSA (n):
    delta = 1.02
    logr = log(delta*delta)
    logl = [0.0] * n
    logl[n-1] = 0.0
    for k in range(n-2, -1, -1):
        logl[k] = logl[k+1] + logr
    nor_logl = normalize_GSO_unitary(logl)
    logl2 = [2 * x for x in nor_logl]
    return logl2
    
    
def normalize_GSO_unitary(logl):
    log_det = sum(logl)
    n = len(logl)
    nor_log_det = [0.0] * n
    for i in range(n):
        nor_log_det[i] = logl[i] - log_det / n
    return nor_log_det


def return_SDBKZ_logb1(n, beta):
    log_spherevol = SPHERE_LOGVOL[beta]
    exponent = -1.0 * (n-1) / beta / (beta - 1)
    logb1 = exponent * log_spherevol
    return logb1


def return_SDBKZ_GSA_ratio(beta):
    log_spherevol = SPHERE_LOGVOL[beta]
    exponent = -1.0 / beta / (beta - 1)
    return exponent * log_spherevol * 2.0


def get_SDBKZ_GSA(n, beta,verbose=1):
    logb1 = return_SDBKZ_logb1(n, beta)
    logratio = return_SDBKZ_GSA_ratio(beta)
    logL = []
    for i in range(n-beta):
        logL.append(logb1)
        logb1 = logb1 - logratio
    carry = (0 - sum(logL))/beta
    taillog2 = get_BKZ_GH(beta, beta, tailmode=1)
    taillog = [x/2.0 for x in taillog2]
    for i in range(len(taillog)):
        logL.append (taillog[i] + carry)
    logL = normalize_GSO_unitary(logL)        
    return logL

def get_GSA(n, beta, verbose=1):
    logb1 = return_SDBKZ_logb1(n, beta)
    logratio = return_SDBKZ_GSA_ratio(beta)
    logL = []
    for i in range(n):
        logL.append(logb1)
        logb1 = logb1 - logratio
    logL = normalize_GSO_unitary(logL)
    return logL


def rhff(k, alpha):
    """
    `α⋅GH(k)^{1/(k-1)}`

    :param k:
    :param alpha:

    """
    from math import log, exp

    small = (
        (0, 0),
        (1, 0),
        (2, -0.08841287960306055),
        (3, -0.07544412913470162),
        (4, -0.07544412913470162),
        (5, -0.07544412913470162),
        (6, -0.07048974749276288),
        (7, -0.06694511236665464),
        (8, -0.06693551836216481),
        (9, -0.06514479158624617),
        (10, -0.06504048330773132),
        (11, -0.06300742595211455),
        (12, -0.06292516413568243),
        (13, -0.06144932038223158),
        (14, -0.061335194884657974),
        (15, -0.05979787485927308),
        (16, -0.059698538727778944),
        (17, -0.05854990651097318),
        (18, -0.05842140906014142),
        (19, -0.05743190747788606),
        (20, -0.057204890112602075),
        (21, -0.05625926852349446),
        (22, -0.0561373418825689),
        (23, -0.05513018652148212),
        (24, -0.055035755076112546),
        (25, -0.05444317674587422),
        (26, -0.054169276891314226),
        (27, -0.05393347149410525),
        (28, -0.05377470641416448),
        (29, -0.05373663760840141),
        (30, -0.05367325837865427),
        (31, -0.05368859276991003),
        (32, -0.053211946661946876),
        (33, -0.05289686076369957),
        (34, -0.05256983259115176),
        (35, -0.05208132942258301),
        (36, -0.051994765423887495),
        (37, -0.05088847622411832),
        (38, -0.05074262422310026),
        (39, -0.050366640049666535),
        (40, -0.05005110125528332),
        (41, -0.05273129943387767),
        (42, -0.052517722118313626),
        (43, -0.05205856884875133),
        (44, -0.051824674774980664),
        (45, -0.05141429845211216),
        (46, -0.05118951330085285),
        (47, -0.050749690271035014),
        (48, -0.05048220060429904),
        (49, -0.050113727094377634),
        (50, -0.04986388279651427),
    )

    if k <= 2:
        return 1.0219
    elif k <= 40:
        return alpha * exp(-small[k][1] / 4)
    else:
        return exp((log(alpha) + lghf(k)) / (k - 1))

    
def lghf(d):
    """
    `log(GH(d))`

    :param d:

    """
    from scipy.special import loggamma
    from math import sqrt, pi, log

    d = float(d)
    return float(loggamma(d / 2 + 1) / d - log(sqrt(pi)))
    
