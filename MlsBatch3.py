#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:05 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import MlsGroupDynamics_scanParSpace as mls
import numpy as np

model_par = {
        #time and run settings
        "maxT":             10000,  # total run time
        "maxPopSize":       20000,  #stop simulation if population exceeds this number
        "minT":             250,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       400,    # average over this time window
        "rms_window":       400,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,   # when to stop calculations
        "rms_err_trNGr":    5E-1,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    50,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   20,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.05,  # cost of cooperation
        "indv_mutationR":   1E-3,   # mutation rate to cheaters
        # group size control
        "indv_K":           50,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_Sfission':      0,
        'gr_Cfission':      1/100,
        # extinction rate
        'delta_group':      1,      # exponent of denisty dependence on group #
        'K_group':          1000,    # carrying capacity of groups
        'delta_tot':        0,      # exponent of denisty dependence on total #indvidual
        'K_tot':            4000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
    }


mainName = 'scan2D_Jan23Mut'

deltaIndv_Vec = np.array([1])
delta_group_Vec = np.array([1])
fissionSlope_Vec = np.array([0])
mu_Vec = np.array([1E-4, 1E-3, 1E-2, 5E-2])


def set_model_par(settings):
    model_par_local = model_par.copy()
    for key, val in settings.items():
        model_par_local[key] = val
    return model_par_local

def run_batch():
    for delta_group in delta_group_Vec:
        for deltaIndv in deltaIndv_Vec:
            for fissionSlope in fissionSlope_Vec:
                for mu in mu_Vec:
                
                    if delta_group==0:
                        dTot = 1
                        dGrp = 0
                    else:
                        dTot= 0
                        dGrp = 1
                        
                    if fissionSlope==0:
                        fissC = 1/100
                    else:
                        fissC = 0
                        
                        
                    settings = {'delta_indv': deltaIndv, 'delta_tot' : dTot, 'delta_group' : dGrp,
                                'gr_Sfission' : fissionSlope, 'gr_Cfission': fissC,
                                'indv_mutationR' : mu}
                    
                    modelParCur = set_model_par(settings)
                    _ = mls.load_or_run_model(mainName, modelParCur)
            
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_batch()
    

