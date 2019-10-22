#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:47:58 2019

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import matplotlib.pyplot as plt
import scipy.stats as stat
import numpy as np
from numba import jit, void, f8, i8

from scipy import special
from numba.extending import get_cython_function_address
import ctypes
import time


# create numba compatible inverse poisson cdf function
addr = get_cython_function_address("scipy.special.cython_special", "pdtrik")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
pdtrik_fn = functype(addr)

## Cython version of Scipy ndtri function (inv cdf)
#@jit(f8(f8), nopython=True)
#def pdtrik_in_njit(x):
#    return pdtrik_fn(x)

#JIT compatible inverse cdf    
#@jit(f8(f8, i8), nopython=True)
def pois_inv_cdf_jit(p, mu):
    return pdtrik_fn(p, mu)


#import scipy.special.pdtrik



mu = int(3)
N = int(1E6)



    
@jit(f8(f8), nopython=True)
def draw_rand_poisson(expectation):
    rndPois = np.random.poisson(expectation)
    return rndPois


#@jit(f8(i8), nopython=True)
def create_r1(mu):
    rndPois = stat.poisson.rvs(mu)
    return rndPois

#@jit(f8(f8, i8), nopython=True)
def create_r2(rndUni, mu):
    rndPois = stat.poisson.ppf(rndUni, mu)    
    return rndPois
    

def process_r1():
    rnd = np.zeros(N)    
    for i in range(N):
        rnd[i] = create_r1(mu)
    return rnd

def process_r3():
    rnd = np.zeros(N)    
    for i in range(N):
        rnd[i] = create_r3(mu)
    return rnd


def process_r2():
    rnd = np.zeros(N)  
    rndUni = np.random.random(N)
    for i in range(N):
        rnd[i] = create_r2(rndUni[i],mu)
    return rnd




start = time.time()
rnd1 = process_r1()
end = time.time()
print("Elapsed time r1 = %s" % (end - start))

    
start = time.time()
rnd2 = process_r2()
end = time.time()
print("Elapsed time r2 = %s" % (end - start))

start = time.time()
rnd3 = process_r3()
end = time.time()
print("Elapsed time r3 = %s" % (end - start))



binEdges = np.arange(-0.5,5*mu)
binCenter = np.arange(0,5*mu)

binCount1, _ = np.histogram(rnd1, bins=binEdges)
binCount2, _ = np.histogram(rnd2, bins=binEdges)
binCount3, _ = np.histogram(rnd3, bins=binEdges)


fig = plt.figure()
plt.plot(binCenter, binCount1, label='poission')
plt.plot(binCenter, binCount2, label='uniform')
plt.plot(binCenter, binCount3, label='np poisson')

plt.legend()
