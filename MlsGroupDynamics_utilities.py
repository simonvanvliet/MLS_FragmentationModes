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
@jit(i8(f8[:], f8), nopython=True)
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
@jit(f8[:, :](i8, i8), nopython=True)
def create_randMat(num_t, num_rand):
    notDone = True
    while notDone:
        randMat = np.random.random((num_t, num_rand))
        containsNo0 = (~np.any(randMat == 0))
        containsNo1 = (~np.any(randMat == 1))
        if containsNo0 & containsNo1:
            notDone = False

    return randMat


# %% Model sampling functions
#calculate moving average of time vector
@jit(UniTuple(f8, 2)(f8[:], i8, i8), nopython=True)
def calc_moving_av(f_t, curr_idx, windowLength):
    # get first time point
    start_idx = max(0, curr_idx - windowLength + 1)
    movingAv = f_t[start_idx:curr_idx].mean()
    movindStd = f_t[start_idx:curr_idx].std()

    return (movingAv, movindStd)


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


