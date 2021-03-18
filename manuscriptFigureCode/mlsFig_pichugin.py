#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 23 2020
Edited July 2 2020

Creates data for SI Figure S5 (comparison with Pichugin et al)
Scans model parameters within full 2D parameter space
Runs version of model with Pichugin et al 2017 like rate function  


Output stored on disk as a Pandas dataframe
To run code on UBC Zoology cluster use /Linux/anaconda3/bin/python MlsGroupDynamics_fig_pichugin.py

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

from mainCode import MlsGroupDynamics_pichugin as mls
from joblib import Parallel, delayed
import numpy as np
import pandas as pd


"""============================================================================
SET MODEL SETTINGS
============================================================================"""

#SET nr of cores to use
numCore = 40 # SET TO MAX NR OF CORES TO USE 
# SET TO POPULATION SIZE AT WHICH TO END SIMULATIONS
endPopSize = 1E5 
#SET TO POPULATION SIZE WHERE TO START FIT
startFit = 2E4 
#SET name of output
fileName = 'Data_FigSX_Pichugin'
#SET nr of replicates
nReplicate = 5

#SET 2D parameter grid
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07) 

#SET other parameters to scan
parNames = ['alpha_b','indv_K'] #parameter keys
par0_vec = np.array([40, 20, 10, 1, 0.1, 0.05, 0.025]) #parameter values
par1_vec = np.array([20]) #parameter values

#SET model parameters
model_par = {
        #time and run settings
        "maxT":             100,  # total run time
        "maxPopSize":       endPopSize,  #stop simulation if population exceeds this number
        "startFit":         startFit,  #start fit of growth rate here
        "sampleInt":        0.05, # sampling interval
        "mav_window":       1,   # average over this time window
        "rms_window":       1,   # calc rms change over this time window
        # settings for initial condition
        "init_groupNum":    100,  # initial # groups
        "init_fCoop":       1,   # DO NOT CHANGE, code only works if init_fCoop = 1
        "init_groupDens":   10,  # initial total cell number in group
        # settings for individual level dynamics
        "indv_NType":       1,  # DO NOT CHANGE, code only works if indv_NType = 1
        "indv_mutR":        0,  # DO NOT CHANGE, code only works if indv_mutR = 0
        "indv_migrR":       0,  # mutation rate to cheaters
        # group size control
        "indv_K":           20,# max group size
        # setting for group rates
        # fission rate
        'gr_CFis':          1E6, # when group size >= Kind this is fission rate
        'alpha_b':          1,
        'grp_tau':          1,
        # settings for fissioning
        'offspr_size':      0.1,  # offspr_size <= 0.5 and
        'offspr_frac':      0.9,  # offspr_size < offspr_frac < 1-offspr_size'
        # extra settings
        'run_idx':          1,
        'replicate_idx':    1
    }
  

"""============================================================================
Define functions
============================================================================"""

#set model parameters for fission mode
def set_model_par(model_par, settings):
    #copy model par (needed because otherwise it is changed in place)
    model_par_local = model_par.copy()
    
    #set model parameters
    for key, val in settings.items():
        model_par_local[key] = val
                               
    return model_par_local

#csv export code
#load data
def export_growthrate_csv(results, mainName):
    # process and store output
    statData = np.vstack(results)
    #save main data file
    dataName = mainName + '.csv'
    header = ','.join(statData.dtype.names)
    np.savetxt(dataName, statData.view(np.float64), delimiter=',', header=header, comments='')
    return None

# create lists of model parameters to scan
def create_model_par_list(model_par):
   #create model paremeter list for all valid parameter range
    modelParList = []
    run_idx = 0

    for par0 in par0_vec:
        for par1 in par1_vec:
            run_idx += 1
            for offspr_size in offspr_size_Vec:
                for offspr_frac in offspr_frac_Vec:
                    for repIdx in range(nReplicate):
                        inBounds = offspr_frac >= offspr_size and \
                                    offspr_frac <= (1 - offspr_size)
                        if inBounds:                    
                            settings = {parNames[0]    : par0,
                                        parNames[1]    : par1,
                                        'run_idx'      : run_idx,
                                        'replicate_idx': repIdx+1,
                                        'offspr_size'  : offspr_size,
                                        'offspr_frac'  : offspr_frac,
                                        }

                            curPar = set_model_par(model_par, settings)
                            modelParList.append(curPar)
    return modelParList

# run model code
def run_model():
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)
                
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.single_run_trajectories)(par) for par in modelParList)
    
    #store output to disk 
    fileNameTemp = fileName + '_temp' + '.npy'
    np.save(fileNameTemp, results)
    
    #convert to pandas dataframe and export
    fileNameFull = fileName + '.pkl'
    outputComb = np.reshape(results, (-1))
    df = pd.DataFrame.from_records(outputComb)
    df.to_pickle(fileNameFull)
    
    return results

#run parscan
if __name__ == "__main__":
    results = run_model()    

                            
