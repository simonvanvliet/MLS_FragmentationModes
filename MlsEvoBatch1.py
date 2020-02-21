#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:05 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import MlsGroupDynamics_evolve as mls
import numpy as np
from joblib import Parallel, delayed

mainName = 'evolution_Feb19'
numCore = 30;


gr_Sfission_Vec = np.array([0, 2, 4])
indv_KVec = np.array([100, 200])
init_Aray = np.array([[0.05,0.06],[0.05,0.5],[0.05,0.94],[0.25,0.5],[0.45,0.5]])
numInit = init_Aray.shape[0]
K_tot_def = 5000

model_par = {
    #time and run settings
    "maxPopSize":       0,
    "maxT":             5000,   # total run time
    "minT":             5000,   # min run time
    "sampleInt":        20,      # sampling interval
    "mav_window":       100,    # average over this time window
    "rms_window":       100,    # calc rms change over this time window
    "rms_err_trNCoop":  0,   # when to stop calculations
    "rms_err_trNGr":    0,   # when to stop calculations
    # settings for initial condition
    "init_groupNum":    10,     # initial # groups
    "init_fCoop":       1,
    "init_groupDens":   50,     # initial total cell number in group
    # settings for individual level dynamics
    # complexity
    "indv_NType":       2,
    "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
    # mutation load
    "indv_cost":        0.01,  # cost of cooperation
    "indv_migrR":       0,   # mutation rate to cheaters
    # set mutation rates
    'mutR_type':        1E-3,
    'mutR_size':        2E-2, 
    'mutR_frac':        2E-2, 
    # group size control
    "indv_K":           100,     # total group size at EQ if f_coop=1
    "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
    # setting for group rates
    # fission rate
    'gr_Cfission':      1/100,
    'gr_Sfission':      2,
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
  
        
    
def set_model_par(settings):
    model_par_local = model_par.copy()
    for key, val in settings.items():
        model_par_local[key] = val
    return model_par_local



modelParList = []

def run_batch():
    for gr_Sfission in gr_Sfission_Vec:
        for indv_K in indv_KVec:
            for ii in range(numInit):
                                
                if gr_Sfission == 0:
                    K_tot = K_tot_def * 4
                else:
                    K_tot = K_tot_def
                    
                settings = {'gr_Sfission' : gr_Sfission,
                            'indv_K' : indv_K,
                            'K_tot'  : K_tot,
                            'offspr_sizeInit': init_Aray[ii, 0],
                            'offspr_fracInit': init_Aray[ii, 1]}
                
                
                modelParCur = set_model_par(settings)
                modelParList.append(modelParCur)
    
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.single_run_save)(par, mainName) for par in modelParList)

    return results


#run parscan and make figure
if __name__ == "__main__":
    results = run_batch()
    fileName = mainName + '_batch.npz'
    np.savez(fileName, results=results)
    

