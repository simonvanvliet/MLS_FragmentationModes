#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:15:00 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""


import numpy as np
from numba.types import Tuple, UniTuple
from numba import jit, f8, i8

mutR_frac =0.01
mutR_size =0.01
fracIdx = 0.5
sizeIdx = 0.25
nBinOffsprSize = 100
nBinOffsprFrac = 100

rand = np.random.random(2)


@jit(UniTuple(i8, 2)(f8, f8, i8, i8, f8[:, ::1]), nopython=True)
def mutate_group(mutR_frac, mutR_size, fracIdx, sizeIdx, rand):
    #check for mutation in offspring size
    if rand[0].item() < mutR_size / 2:  # offspring size mutates to lower value
        offsprSizeIdx = max(0, sizeIdx - 1)
    elif rand[0].item() < mutR_size:  # offspring size mutates to lower value
        offsprSizeIdx = min(nBinOffsprSize - 1, sizeIdx + 1)
    else:
        offsprSizeIdx = sizeIdx

    #check for mutation in offspring fraction
    if rand[1].item() < mutR_frac / 2:  # offspring size mutates to lower value
        offsprFracIdx = max(0, fracIdx - 1)
    elif rand[1].item() < mutR_frac:  # offspring size mutates to lower value
        offsprFracIdx = min(nBinOffsprFrac - 1, fracIdx + 1)
    else:
        offsprFracIdx = fracIdx

    #make sure we stay inside allowed trait space
    
    return (offsprFracIdx, offsprSizeIdx)