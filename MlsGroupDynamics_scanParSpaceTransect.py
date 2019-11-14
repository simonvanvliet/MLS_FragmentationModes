#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

Last Update Oct 22 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import itertools
import MlsGroupDynamics_main as mls
import datetime
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import os

"""============================================================================
Define parameters
============================================================================"""

override_data = True #set to true to force re-calculation
numCore = 20 #number of cores to run code on
numThread = 2 #number o threads per core
#where to store output?
mainName = 'vanVliet_scan'

#setup variables to scan
offspr_sizeVec = np.arange(0.01, 0.5, 0.01)
mu_vec = np.array([1e-4, 1e-3, 1e-2])
type_vec = np.arange(1,5)
tau_vec = np.array([10, 100, 1000])
assymetry_vec = np.array([1, 2, 4])

#set other parameters
model_par = {
    # solver settings
    "maxT":             3000,  # total run time
    "minT":             400,   # min run time
    "sampleInt":        1,     # sampling interval
    "mav_window":       200,   # average over this time window
    "rms_window":       200,   # calc rms change over this time window
    "rms_err_trNCoop":  2E-2,  # when to stop calculations
    "rms_err_trNGr":    0.1,  # when to stop calculations
    # settings for initial condition
    "init_groupNum":    100,  # initial # groups
    # initial composition of groups (fractions)
    "init_fCoop":       1,
    "init_groupDens":   100,  # initial total cell number in group  
    # settings for individual level dynamics
    "indv_NType":       1,
    "indv_cost":        0.1,  # cost of cooperation
    "indv_K":           100,  # total group size at EQ if f_coop=1
    "indv_mutationR":   1E-3,  # mutation rate to cheaters
    # difference in growth rate b(j+1) = b(j) / asymmetry
    "indv_asymmetry":    1,
    # setting for group rates
    'gr_Sfission':       0.,    # fission rate = (1 + gr_Sfission * N)/gr_tau
    'gr_Sextinct':      0.,    # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
    'gr_K':             100,   # total carrying capacity of cells
    'gr_tau':           100,   # relative rate individual and group events
    # settings for fissioning
    'offspr_size':      0.5,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
}

#setup name to save files
parName = '_cost%.0g_indvK%.0e_grK%.0g_sFis%.0g_sExt%.0g' % (
    model_par['indv_cost'], 
    model_par['indv_K'], model_par['gr_K'],
    model_par['gr_Sfission'], model_par['gr_Sextinct'])
dataFileName = mainName + parName 


"""============================================================================
Define functions
============================================================================"""
#set model parameters for fission mode
def set_fission_mode(offspr_size, indv_NType, indv_mutationR, indv_asymmetry, gr_tau):
    #copy model par (needed because otherwise it is changed in place)
    offspr_frac = 1 - offspr_size
    model_par_local = model_par.copy()
    model_par_local['offspr_size'] = offspr_size
    model_par_local['offspr_frac'] = offspr_frac
    model_par_local['indv_NType'] = indv_NType
    model_par_local['indv_mutationR'] = indv_mutationR
    model_par_local['indv_asymmetry'] = indv_asymmetry
    model_par_local['gr_tau'] = gr_tau

    return model_par_local

# run model
def run_model():
    #create model paremeter list for all valid parameter range
    # *x unpacks variables stored in tuple x e.g. if x = (a1,a2,a3) than f(*x) = f(a1,a2,a3)
    # itertools.product creates all possible combination of parameters
    modelParList = [set_fission_mode(*x)
                   for x in itertools.product(*(offspr_sizeVec, type_vec, mu_vec, assymetry_vec, tau_vec))]

    modelParList = modelParList[0:4]
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))

    with parallel_backend("loky", inner_max_num_threads=numThread):
        results = Parallel(n_jobs=nJobs, verbose=10, timeout=1.E8)(
            delayed(mls.single_run_finalstate)(par) for par in modelParList)
    
    np.savez(dataFileName, results=results,
             offspr_sizeVec = offspr_sizeVec,
             mu_vec = mu_vec,
             type_vec = type_vec,
             tau_vec = tau_vec,
             assymetry_vec = assymetry_vec,
             modelParList = modelParList, date=datetime.datetime.now())
    
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_model()

