#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-07-03

Code for figure X
- Triangle showing, for each strategy, the number of cells and number of groups at equilibrium.
Varies complexity of community by changing NType or Asymmetry

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

============================================================================
Run Model and plot results
============================================================================"""

import sys
sys.path.insert(0, '..')

#load code
from mainCode import MlsGroupDynamics_main as mls
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

"""============================================================================
SET MODEL SETTINGS
============================================================================"""

#SET nr of cores to use
nCore = 40

#SET OUTPUT FILENAME
fileName = 'scanComplexity'

#setup 2D parameter grid
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07)

#set model mode settings (slope and migration rate)
mode_set = np.array([[4, 3, 2, 1, 2, 2, 2],
                     [1, 1, 1, 1, 2, 3, 4]])
modeNames = ['indv_NType', 'indv_asymmetry']
mode_vec = np.arange(mode_set.shape[1])

#SET fission rates to scan
gr_CFis_vec = np.array([0.05])# changed from 0.01 in 2020-08-07

#SET nr of replicates
nReplicate = 5

#SET rest of model parameters
model_par = {
          #time and run settings
        "maxT":             5000,  # total run time
        "maxPopSize":       1000000,  #stop simulation if population exceeds this number
        "minT":             2500,    # min run time
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
        'K_tot':            1E5,    # carrying capacity of total individuals
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
CODE TO MAKE FIGURE
============================================================================"""


#set model parameters for fission mode
def set_model_par(model_par, settings):
    #copy model par (needed because otherwise it is changed in place)
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

    for mode in mode_vec:
        for gr_CFis in gr_CFis_vec:
            run_idx += 1
            for repIdx in range(nReplicate):
                for offspr_size in offspr_size_Vec:
                    for offspr_frac in offspr_frac_Vec:
                        inBounds = offspr_frac >= offspr_size and \
                                offspr_frac <= (1 - offspr_size)

                        if inBounds:
                            settings = {'indv_NType'     : mode_set[0, mode],
                                        'indv_asymmetry' : mode_set[1, mode],
                                        'gr_CFis'        : gr_CFis,
                                        'offspr_size'    : offspr_size,
                                        'offspr_frac'    : offspr_frac,
                                        'run_idx'        : run_idx,
                                        'replicate_idx'  : repIdx+1,
                                        }
                            curPar = set_model_par(model_par, settings)
                            modelParList.append(curPar)

    return modelParList

# run model code
def run_model():
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)

    # run model, use parallel cores
    nJobs = min(len(modelParList), nCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.run_model_steadyState_fig)(par) for par in modelParList)

    #store output to disk
    fileNameTemp = fileName + '_temp' + '.npy'
    np.save(fileNameTemp, results)

    #convert to pandas dataframe and export
    fileNameFull = fileName + '.pkl'
    dfSet = [pd.DataFrame.from_records(npa) for npa in results]
    df = pd.concat(dfSet, axis=0, ignore_index=True)
    df.to_pickle(fileNameFull)

    return None

#run parscan
if __name__ == "__main__":
    run_model()
