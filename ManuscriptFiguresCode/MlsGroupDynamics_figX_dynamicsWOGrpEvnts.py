#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-06-29

Code for figure X
Runs model without group events and outputs temporal dynamics. 

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

============================================================================
Run Model and plot results
============================================================================"""

#load code
import MlsGroupDynamics_main as mls
#import pandas as pd
import numpy as np
from joblib import Parallel, delayed

"""============================================================================
SET MODEL SETTINGS
============================================================================"""

#SET nr of cores to use
nCore = 40

#SET OUTPUT FILENAME
fileName = 'dynamicsWOGrpEvents'

#SET mutation rates to scan
mutR_vec = np.array([1E-2, 1E-3])

#SET XY Coordinates in parameter space
xyLoc_vec = np.array([[0.05,0.05],[0.05,0.5],[0.05,0.95],[0.25,0.5],[0.45,0.5]])
numInit = xyLoc_vec.shape[0]

#SET nr of replicates
nReplicate = 10

#SET rest of model parameters
model_par = {
          #time and run settings
        "maxT":             2500,  # total run time
        "maxPopSize":       40000,  #stop simulation if population exceeds this number
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
        'gr_CFis':          0,
        'gr_SFis':          0,     # measured in units of 1 / indv_K
        'grp_tau':          0,     # constant multiplies group rates
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
    
    for mutR in mutR_vec:
        for initLocIdx in range(numInit):
            run_idx += 1
            for repIdx in range(nReplicate):
                #implement local settings    
                settings = {'indv_mutR'      : mutR,
                            'run_idx'        : run_idx,
                            'replicate_idx'  : repIdx+1,
                            'offspr_sizeInit': xyLoc_vec[initLocIdx, 0],
                            'offspr_fracInit': xyLoc_vec[initLocIdx, 1]}

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
        delayed(mls.run_model_dynamics_fig)(par) for par in modelParList)
    
    #store output to disk 
    fileNameTemp = fileName + '_temp' + '.npy'
    fileNameFull = fileName + '.pkl'
    np.save(fileNameTemp, results)
    
    # #convert to pandas dataframe and export
    # outputComb = np.hstack(results)
    # df = pd.DataFrame.from_records(outputComb)
    # df.to_pickle(fileNameFull)
    
    return None

#run parscan
if __name__ == "__main__":
    statData = run_model()    




    
    
#output, _, _ = mls.run_model(model_par)
#
## convert to pandas dataframe and export
#df = pd.DataFrame.from_records(output)
#dataName = fileName + '.pkl'
#df.to_pickle(dataName)
#



