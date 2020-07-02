#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Juen 23 2020

Creates data for SI Figure S[X] (comparison with Pichugin et al)
Scans model parameters within full 2D parameter space
Runs version of model with Pichugin et al 2017 like rate function  



Output stored on disk in csv file format
Format:
csv file is organized in blocks of three lines, each block corresponds to a single run.
The first line in a block shows the sampling time
The second line in a block shows the total population size
The third line in a block shows the number of groups
The first few columns give the parameters values used for that run (see header)
All other columns show the data as function of time

to run code use python3

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import MlsGroupDynamics_pichugin as mls
from joblib import Parallel, delayed
import numpy as np
import itertools

"""============================================================================
Define parameters
============================================================================"""

#SET VARIABLE SETTINGS
numCore = 40 # SET TO MAX NR OF CORES TO USE 
endPopSize = 1E5 # SET TO POPULATION SIZE AT WHICH TO END SIMULATIONS
startFit = 2E4 #SET TO POPULATION SIZE WHERE TO START FIT

#set name of output
mainName = 'Data_FigSX_Pichugin'

#set 2D parameter grid
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07) 

#set other parameters to scan
parNames = ['alpha_b','indv_K'] #parameter keys
par0_vec = np.array([50, 20, 10, 1, 0.1, 0.05, 0.02]) #parameter values
par1_vec = np.array([20]) #parameter values

#set model parameters
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
    
    # itertools.product creates all possible combination of parameters
    for parValues in itertools.product(*(par0_vec, par1_vec)):   
        for offspr_size in offspr_size_Vec:
            for offspr_frac in offspr_frac_Vec:
                inBounds = offspr_frac >= offspr_size and \
                            offspr_frac <= (1 - offspr_size)
                if inBounds:                    
                    settings = {parNames[0]    : parValues[0],
                                parNames[1]    : parValues[1],
                                'offspr_size'  : offspr_size,
                                'offspr_frac'  : offspr_frac,
                                }

                    curPar = set_model_par(model_par, settings)
                    modelParList.append(curPar)
    return modelParList

# run model code
def run_model(mainName, model_par, numCore):
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)
            
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.single_run_trajectories)(par) for par in modelParList)
    
    #store output to disk as .npz 
    dataFilePath = mainName + '_python' + '.npz'
    np.savez(dataFilePath, results   = results)

    #store output as csv
    export_growthrate_csv(results, mainName)

    return results

#run parscan
if __name__ == "__main__":
    results = run_model(mainName, model_par, numCore)    

                            
