#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:05 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import MlsGroupDynamics_scanParSpace as mls
import numpy as np

mainName = 'scan2D_Feb15'
numCore = 40;


gr_Sfission_Vec = np.array([0, 2, 4])
indv_KVec = np.array([25, 50, 100, 200])
K_tot_def = 20000

model_par = {
        #time and run settings
        "maxT":             10000,  # total run time
        "maxPopSize":       30000,  #stop simulation if population exceeds this number
        "minT":             200,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       400,    # average over this time window
        "rms_window":       400,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,   # when to stop calculations
        "rms_err_trNGr":    5E-1,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    10,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   10,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,  # cost of cooperation
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,   # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_Cfission':      1/100,
        'gr_Sfission':      1/50,
        # extinction rate
        'delta_grp':        0,      # exponent of denisty dependence on group #
        'K_grp':            0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            K_tot_def,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.8  # offspr_size < offspr_frac < 1-offspr_size'
    }
  
        
    
def set_model_par(settings):
    model_par_local = model_par.copy()
    for key, val in settings.items():
        model_par_local[key] = val
    return model_par_local

def run_batch():

    for gr_Sfission in gr_Sfission_Vec:
        for indv_K in indv_KVec:
            if gr_Sfission == 0:
                K_tot = K_tot_def * 5
            else:
                K_tot = K_tot_def
                
            settings = {'gr_Sfission' : gr_Sfission,
                        'indv_K' : indv_K,
                        'K_tot'  : K_tot}
            
            print(settings)
            
            modelParCur = set_model_par(settings)
            _ = mls.load_or_run_model(mainName, modelParCur, numCore)
            
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_batch()
    

