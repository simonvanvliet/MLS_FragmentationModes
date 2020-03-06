#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:47:37 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import MlsGroupDynamics_utilities as util
import math
from numba import jit, f8, i8
from numba.types import UniTuple





@jit(i8(f8[:, :], f8), nopython=True)
def select_random_event_2D(propensity_vec, randNum):
    # calculate cumulative propensities
    cumPropensity = propensity_vec.cumsum()
    # rescale uniform random number [0,1] to total propensity
    randNumScaled = randNum * cumPropensity[-1]
    # create index vector
    index = np.arange(cumPropensity.size)
    # select group
    id_group = index[(cumPropensity > randNumScaled)][0]
    return id_group

@jit(UniTuple(i8,2)(i8, UniTuple(i8,2)), nopython=True)
def flat_to_2d_index(flatIndex, shape):
    """
    converts flattend index to 3D indices for 'C' order arrays
    warning: no error checking, only use if you are sure input is C array
    """    
    idx0 = int(np.floor(flatIndex / shape[1]))
    idx1 = int(flatIndex % shape[1])
    
    return (idx0,idx1)


#find trait of affected cell
def test(loc,size=(100,100)):
    
    a = np.zeros(size)
    shape = a.shape
    a[loc] = 1
    
    b= np.random.random(1)
    
    if len(size) == 2:
        idX = util.select_random_event_2D(a, b.item())
    else:
        cellID = util.select_random_event_3D(a, b.item())
        idX = util.flat_to_3d_index(cellID, shape)
        
    print(idX)
    return idX

test((4, 49,5), size=(8, 100,100))