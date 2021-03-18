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
from numba import jit, f8, i8
from numba.types import UniTuple

# import os
# os.environ["NUMBA_DISABLE_JIT"] = '0'


"""
 General functions
"""

def set_model_par(model_par, settings):
    """[Sets model parameters based on settings dictionary]
    
    Arguments:
        model_par {[dictionary]} -- Dictionary with base parameters to modify
        
        settings {[dictionary]} -- [keys are names of model parameters, values are parameter values]
    
    Returns:
        [dictionary] -- [modified parameter dictionary]
    """
    model_par_local = model_par.copy()
    for key, val in settings.items():
        model_par_local[key] = val
    return model_par_local


@jit(UniTuple(i8,2)(i8, UniTuple(i8,2)), nopython=True)
def flat_to_2d_index(flatIndex, shape):
    """
    converts flattend index to 2D indices for 'C' order arrays
    warning: no error checking, only use if you are sure input is C array
    """    
    idx0 = int(np.floor(flatIndex / shape[1]))
    idx1 = int(flatIndex % shape[1])
    return (idx0, idx1)

@jit(UniTuple(i8,3)(i8, UniTuple(i8,3)), nopython=True)
def flat_to_3d_index(flatIndex, shape):
    """
    converts flattend index to 3D indices for 'C' order arrays
    warning: no error checking, only use if you are sure input is C array
    """
    nPlane = shape[1]*shape[2]
    idx0 = int(np.floor(flatIndex / nPlane))
    idx12 = flatIndex % nPlane
    idx1 = int(np.floor(idx12 / shape[2]))
    idx2 = int(idx12 % shape[2])
    return (idx0,idx1,idx2)

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

# %%random sample based on propensity, for 2D propensity input
@jit(UniTuple(i8,2)(f8[:, :], f8), nopython=True)
def select_random_event_2D(propensity_vec, randNum):
    # calculate cumulative propensities
    cumPropensity = propensity_vec.cumsum()
    # rescale uniform random number [0,1] to total propensity
    randNumScaled = randNum * cumPropensity[-1]
    # create index vector
    index = np.arange(cumPropensity.size)
    # select group
    id_group = index[(cumPropensity > randNumScaled)][0]
    idx = flat_to_2d_index(id_group, propensity_vec.shape)
    
    return idx

# %%random sample based on propensity, for 3D propensity input
@jit(i8(f8[:, :, :], f8), nopython=True)
def select_random_event_3D(propensity_vec, randNum):
    # calculate cumulative propensities
    cumPropensity = propensity_vec.cumsum()
    # rescale uniform random number [0,1] to total propensity
    randNumScaled = randNum * cumPropensity[-1]
    # create index vector
    index = np.arange(cumPropensity.size)
    # select group
    id_group = index[(cumPropensity > randNumScaled)][0]
    return id_group

@jit(i8(f8, i8))
def truncated_poisson(expect_value, cutoff):
    searching = True
    while searching:
        randNum = int(np.random.poisson(expect_value))
        if randNum <= cutoff:
            searching = False
    return randNum


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
    if windowLength < 1:
        windowLength = 1
    start_idx = max(0, curr_idx - windowLength+1)
    movingAv = f_t[start_idx:curr_idx+1].mean()
    if windowLength > 1:
        movindStd = f_t[start_idx:curr_idx+1].std()
    else:
        movindStd = np.nan

    return (movingAv, movindStd)


# calculate rms error of time vector
@jit(f8(f8[:], i8, i8), nopython=True)
def calc_rms_error(mav_t, curr_idx, windowLength):
    # get first time point
    if windowLength < 1:
        windowLength = 1
    start_idx = max(0, curr_idx-windowLength+1)
    # get time points to process
    localsegment = mav_t[start_idx:curr_idx+1]
    # calc rms error
    av = localsegment.mean()
    errorSquared = (localsegment-av)**2
    meanErrorSquared = errorSquared.mean()
    rms_err = math.sqrt(meanErrorSquared)

    return rms_err


