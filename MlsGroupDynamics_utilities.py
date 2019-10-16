#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on March 22 2019
Last Update March 22 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

Contains varies functions used in MLS model code and in figure code

"""
import math
import numpy as np
import scipy.stats as st
from numba import jit, f8, i8, vectorize
from numba.types import UniTuple
from scipy import special
from numba.extending import get_cython_function_address
import ctypes
import scipy.optimize as opt

# create numba compatible inverse normal cdf function
addr = get_cython_function_address("scipy.special.cython_special", "ndtri")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
ndtri_fn = functype(addr)

"""
 General functions
"""


# %%random sample based on propensity
@jit(i8(f8[::1], f8), nopython=True)
def select_random_event(propensity_vec, randNum):
    # calculate cumulative propensities
    cumPropensity = propensity_vec.cumsum()
    # rescale uniform random number [0,1] to total propensity
    randNumScaled = randNum * cumPropensity[-1]
    # create index vector
    index = np.arange(cumPropensity.size)
    # select group
    id_group = index[(cumPropensity > randNumScaled)][0]

    return id_group


# %%create matrix with random numbers, excluding 0 and 1
def create_randMat(num_t, num_rand):
    notDone = True
    while notDone:
        randMat = np.random.random((num_t, num_rand))
        containsNo0 = (~np.any(randMat == 0))
        containsNo1 = (~np.any(randMat == 1))
        if containsNo0 & containsNo1:
            notDone = False

    return randMat


# %% Set figure size in cm
def set_fig_size_cm(fig, w, h):
    cmToInch = 0.393701
    wInch = w * cmToInch
    hInch = h * cmToInch
    fig.set_size_inches(wInch, hInch)
    return None


# %%calculate timestep to have max 1 host level event per step
@jit(f8(f8, f8[:]), nopython=True)
def calc_max_time_step(max_host_prop, dtVec):
    max_p2 = 0.01
    # calc P(2 events)
    p0 = np.exp(-max_host_prop * dtVec)
    p1 = max_host_prop * dtVec * np.exp(-max_host_prop * dtVec)
    p2 = 1 - p0 - p1
    # choose biggest dt with constrained that P(2 events) < maxP2
    dtMax = dtVec[p2 < max_p2].max()
    dtMax = max(dtMax, dtVec.min())
    return dtMax


# %% fast implementation of normal distribution
#JIT compatible normal cdf    
@jit(f8(f8), nopython=True)
def norm_cdf(x):
    'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# inverse normal cdf
def norm_inv_cdf(x):
    return - math.sqrt(2.0) * special.erfcinv(2.0 * x)

# Cython version of Scipy ndtri function (inv cdf)
@jit(f8(f8), nopython=True)
def ndtri_in_njit(x):
    return ndtri_fn(x)

#JIT compatible inverse cdf    
@jit(f8(f8), nopython=True)
def norm_inv_cdf_jit(x):
    return ndtri_in_njit(x)


# convert rand num with uniform distribution to one with trunc norm distribution
    #fast JIT compatible version
@jit(f8(f8, f8, f8, f8, f8), nopython=True)
def trunc_norm_fast(exp, std, minV, maxV, rnd):
    minX = norm_cdf((minV - exp) / std)
    maxX = norm_cdf((maxV - exp) / std)
    rndRescaled = rnd * (maxX - minX) + minX
    rndTrunc = norm_inv_cdf_jit(rndRescaled) * std + exp
    if rndTrunc < minV:
        rndTrunc = minV
    elif rndTrunc > maxV:
        rndTrunc = maxV
    return rndTrunc


# convert rand num with uniform distribution to one with trunc norm distribution
#Non JIT version
def trunc_norm(exp, std, min=-np.inf, max=np.inf, type='lin'):
    # log transform data if needed
    if type == 'log':
        exp = math.log10(exp)
        min = math.log10(min)
        max = math.log10(max)
    elif type != 'lin':
        raise Exception('problem with trunc_norm only support lin or log mode')
    # convert bounds to to standard normal distribution
    minT = (min - exp) / std
    maxT = (max - exp) / std
    # draw from truncated normal distribution and transfrom to desired mean and var
    newT = st.truncnorm.rvs(minT, maxT) * std + exp
    # log transfrom if needed
    if type == 'log':
        newT = 10 ** newT
    return newT



# %% Model sampling functions
#calculate moving average of time vector
@jit(UniTuple(f8, 2)(f8[:], i8, i8), nopython=True)
def calc_moving_av(f_t, curr_idx, windowLength):
    # get first time point
    start_idx = max(0, curr_idx - windowLength + 1)
    movingAv = f_t[start_idx:curr_idx].mean()
    movindStd = f_t[start_idx:curr_idx].std()

    return (movingAv, movindStd)


# calculate moving median of time vector
@jit(f8(f8[:], i8, i8), nopython=True)
def calc_moving_med(f_t, curr_idx, windowLength):
    # get first time point
    start_idx = max(0, curr_idx - windowLength + 1)
    movingMed = np.median(f_t[start_idx:curr_idx])

    return movingMed


# calculate rms error of time vector
@jit(f8(f8[:], i8, i8), nopython=True)
def calc_rms_error(mav_t, curr_idx, windowLength):
    # get first time point
    start_idx = max(0, curr_idx-windowLength+1)
    # get time points to process
    localsegment = mav_t[start_idx:curr_idx]
    # calc rms error
    av = localsegment.mean()
    errorSquared = (localsegment-av)**2
    meanErrorSquared = errorSquared.mean()
    rms_err = math.sqrt(meanErrorSquared)

    return rms_err


#%% convert value with continuos categorial states to labeled categorial states
def make_categorial(vector):
    elements = np.unique(vector)
    indexVec = np.arange(elements.size)
    cat_vector = np.zeros(vector.size)
    for idx in range(vector.size):
        cat_vector[idx] = indexVec[elements == vector[idx]]
    return cat_vector


