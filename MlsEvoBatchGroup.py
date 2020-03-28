#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:05 2020

Code runs multiple evolution run, each run is stored on disk independently
Results can be plotted using plotEvolutionBatch an plotEvolutionBatchMovie

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""
import MlsGroupDynamics_evolve_groups as mlsg
import MlsGroupDynamics_utilities as util
import numpy as np
from joblib import Parallel, delayed

""" 
SET SETTINGS
"""
#SET mainName is appended to file name
mainName = 'group_evolution_March9'
#SET number of cores to use
numCore = 4;
#SET group fission rates to scan
gr_Sfission_Vec = np.array([4])
#SET parName and par0Vec to scan over any parameter of choice
par0Name = 'indv_tau'
par0Vec = np.array([0.01,0.1,1])
#SET initial locations of evolution runs
init_Aray = np.array([[0.05,0.05],[0.05,0.5],[0.05,0.95],[0.25,0.5],[0.45,0.5]])
numInit = init_Aray.shape[0]

#SET Population size
K_tot_def = 8000

#SET Model default settings
model_par = {
    #time and run settings
    "maxT":             60000,   # total run time
    "sampleInt":        250,      # sampling interval
    "mav_window":       2500,    # average over this time window
    "rms_window":       2500,    # calc rms change over this time window
    # settings for initial condition
    "init_groupNum":    100,     # initial # groups
    "init_fCoop":       1,
    "init_groupDens":   20,     # initial total cell number in group
    # settings for individual level dynamics
    # complexity
    "indv_NType":       2,
    "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
    # mutation load
    "indv_cost":        0.01,  # cost of cooperation
    "indv_migrR":       0,   # mutation rate to cheaters
    # set mutation rates
    'mutR_type':        1E-3,
    'mutR_size':        1E-2, 
    'mutR_frac':        1E-2, 
    'indv_tau' :        0.1,
    # group size control
    "indv_K":           100,     # total group size at EQ if f_coop=1
    "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
    # setting for group rates
    # fission rate
    'gr_CFis':          1/100,
    'gr_SFis':          2,
    # extinction rate
    'delta_grp':        0,      # exponent of denisty dependence on group #
    'K_grp':            0,    # carrying capacity of groups
    'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
    'K_tot':            5000,   # carrying capacity of total individuals
    'delta_size':       0,      # exponent of size dependence
    # initial settings for fissioning
    'offspr_sizeInit':  0.25,  # offspr_size <= 0.5 and
    'offspr_fracInit':  0.5  # offspr_size < offspr_frac < 1-offspr_size'
    }
  
""" 
Function definitions
"""        

def run_batch():
    """[Runs batch of evolution experiments]
    Returns:
        [list] -- [results of evolution runs]
    """
    #create list with model settings to run
    modelParList = []
    for gr_SFis in gr_Sfission_Vec:
        for par0 in par0Vec:
            for ii in range(numInit):
                #increase total carrying capacity if fission slope is 0
                #needed to prevent extinction
                if gr_SFis == 0:
                    K_tot = K_tot_def * 10
                else:
                    K_tot = K_tot_def
                #implement local settings    
                settings = {'gr_SFis'  : gr_SFis,
                            par0Name   : par0,
                            'K_tot'    : K_tot,
                            'offspr_sizeInit': init_Aray[ii, 0],
                            'offspr_fracInit': init_Aray[ii, 1]}
                #add settings to list
                modelParCur = util.set_model_par(model_par, settings)
                modelParList.append(modelParCur)
    
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mlsg.single_run_save)(par, mainName) for par in modelParList)

    return results

#run parscan
if __name__ == "__main__":
    results = run_batch()
  

