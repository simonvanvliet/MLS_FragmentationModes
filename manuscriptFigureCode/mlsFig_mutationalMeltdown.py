#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

Code for Figure 3E & Figure 5 & Figure S3
- For each strategy, maximum mutation rate, varies fission slope, no migration

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import sys
sys.path.insert(0, '..')

from mainCode import MlsGroupDynamics_main as mls
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
#import plotParScan

"""============================================================================
Define parameters
============================================================================"""

#running with minT = 200, Ktot = 4E4, maxPopSize = 1E5

#SET nr of cores to use
nCore = 20

#SET OUTPUT FILENAME
fileName = 'mutationalMeltdown'

#SET nr of replicates
nReplicate = 7

#set  mutation rates to try
mu_Vec = np.logspace(0,-7,20) 

#setup 2D parameter grid
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07)

#set model mode settings (slope and migration rate)
#mode_set = np.array([[4, 2, 0.1, 0,    0,    0, 0],   # here we run for migration rates > 0 as well
#                     [0, 0,   0, 0, 1e-2, 1e-1, 1]])
mode_set = np.array([[4, 2, 0.1, 0],                   # here we run for migration rate = 0 only
                     [0, 0,   0, 0]])
modeNames = ['gr_SFis', 'indv_migrR']
mode_vec = np.arange(mode_set.shape[1])

#set other parameters to scan
parNames = ['gr_CFis','indv_NType'] #parameter keys
par0_vec = np.array([0.05]) #parameter values - 2020-07-13 changed from 0.01 to 0.05
par1_vec = np.array([1, 2, 3, 4]) #parameter values

#SET rest of model parameters
model_par = {
          #time and run settings
        "maxRunTime":       900,     #max cpu time in seconds
        "maxT":             5000,  # total run time
        "maxPopSize":       1E5,  #stop simulation if population exceeds this number
        "minT":             200,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       200,    # average over this time window
        "rms_window":       200,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-2,   # when to stop calculations
        "rms_err_trNGr":    5E-2,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    100,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   50,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       1,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,   # cost of cooperation
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,      # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0.01,
        'gr_SFis':          0,     # measured in units of 1 / indv_K
        'grp_tau':          1,     # constant multiplies group rates
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            4E4,    # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.01,  # offspr_size <= 0.5 and
        'offspr_frac':      0.01,    # offspr_size < offspr_frac < 1-offspr_size'
        # extra settings
        'run_idx':          1,
        'replicate_idx':    1,
        'perimeter_loc':    0
    }


"""============================================================================
Define functions
============================================================================"""

#set model parameter
def set_model_par(model_par, settings):
    #copy dictionary (needed, otherwise changed in place)
    model_par_local = model_par.copy()

    #set model parameters
    for key, val in settings.items():
        model_par_local[key] = val

    return model_par_local


# run model
def create_model_par_list(model_par):
   #create model paremeter list for all valid parameter range
    modelParList = []
    run_idx = 0

    #create model parameter list for all valid parameter range
    for mode in mode_vec:
        for par0 in par0_vec:
            for par1 in par1_vec:
                run_idx += 1
                for repIdx in range(nReplicate):
                    for offspr_size in offspr_size_Vec:
                        for offspr_frac in offspr_frac_Vec:
                            inBounds = offspr_frac >= offspr_size and \
                                    offspr_frac <= (1 - offspr_size)
                            if inBounds:
                                settings = {'gr_SFis'      : mode_set[0, mode],
                                            'indv_migrR'   : mode_set[1, mode],
                                            parNames[0]    : par0,
                                            parNames[1]    : par1,
                                            'offspr_size'  : offspr_size,
                                            'offspr_frac'  : offspr_frac,
                                            'run_idx'      : run_idx,
                                            'replicate_idx': repIdx+1,
                                            }
                                curPar = set_model_par(model_par, settings)
                                modelParList.append(curPar)
    return modelParList

#run mutational meltdown scan
def run_meltdown_scan(model_par):
    #input parameters to store
    parList = ['run_idx','replicate_idx',
               'indv_NType', 'indv_asymmetry', 'indv_cost',
               'indv_mutR','indv_migrR', 'gr_SFis', 'gr_CFis',
               'offspr_size','offspr_frac',
               'indv_K', 'K_tot',
               'delta_indv', 'delta_tot', 'delta_size', 'delta_grp', 'K_grp']

    stateVar = ['maxMu', 'NTot','fCoop','NGrp']

    dTypeList = [(x, 'f8') for x in parList] + [(x, 'f8') for x in stateVar]
    dType = np.dtype(dTypeList)

    outputMat = np.full(1, np.nan, dType)
    for par in parList:
        outputMat[par] = model_par[par]

    hasMeltdown = True
    idx = 0
    #reduce mutation rate till community can survive
    while hasMeltdown:
        model_par['indv_mutR'] = mu_Vec[idx]
        output = mls.run_model_steadyState_fig(model_par)

        #if community survives, found max mutation burden
        if output['NTot'][-1] > 0:
            hasMeltdown = False
            outputMat['maxMu'] = model_par['indv_mutR']
            outputMat['NTot'] = output['NTot_mav']
            outputMat['fCoop'] = output['fCoop_mav']
            outputMat['NGrp'] = output['NGrp_mav']

        else:
            idx += 1

        #end when lowest possible mutation rate has been reached
        if idx >= mu_Vec.size:
            hasMeltdown = False

    return outputMat

#run model code
def run_model(nCore):
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)

    # run model, use parallel cores 
    nJobs = min(len(modelParList), nCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(run_meltdown_scan)(par) for par in modelParList)

    #store output to disk
    fileNameTemp = fileName + '_temp' + '.npy'
    np.save(fileNameTemp, results)

    #convert to pandas dataframe and export
    fileNameFull = fileName + '.pkl'
    outputComb = np.hstack(results)
    df = pd.DataFrame.from_records(outputComb)
    df.to_pickle(fileNameFull)

    return None

#run parscan
if __name__ == "__main__":
    run_model(nCore)
