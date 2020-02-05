#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:05 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import MlsGroupDynamics_scanParSpace as mls
import numpy as np

mainName = 'scan2D_Jan30Part1'

model_mode_Vec = np.array([2, 3]) #np.arange(4)
cost_Vec = np.array([0.01, 0.1])
Kindv_Vec = np.array([100])
mu_Vec = np.array([1E-3, 5E-3, 1E-2, 5E-2, 1E-1])
NType_Vec = np.array([1, 2, 3, 4, 5])
Assym_Vec = np.array([1])


K_group_def = 2000
K_tot_def = 120000

model_par = {
        #time and run settings
        "maxT":             10000,  # total run time
        "maxPopSize":       40000,  #stop simulation if population exceeds this number
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
        "indv_cost":        0.1,  # cost of cooperation
        "indv_mutationR":   1E-3,   # mutation rate to cheaters
        # group size control
        "indv_K":           0,     # total group size at EQ if f_coop=1
        "delta_indv":       0,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        'model_mode':       0,
        # fission rate
        'slope_coef':       1,
        'gr_Sfission':      0,
        'gr_Cfission':      1/100,
        # extinction rate
        'delta_group':      0,      # exponent of denisty dependence on group #
        'K_group':          K_group_def,    # carrying capacity of groups
        'delta_tot':        0,      # exponent of denisty dependence on total #indvidual
        'K_tot':            K_tot_def,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
    }


def set_model_mode(model_par, mode, K_indv, slope_coef=1):
    model_par['delta_size'] = 0
    model_par['model_mode'] = mode
    
    if mode == 0:
        model_par['delta_group'] = 1
        model_par['delta_tot'] = 0
        model_par['delta_indv'] = 1
        model_par['gr_Sfission'] = 0
        model_par['K_group'] = K_group_def       
        model_par['K_tot'] = 0
    elif mode == 1:
        model_par['delta_group'] = 1
        model_par['delta_tot'] = 0     
        model_par['delta_indv'] = 0
        model_par['gr_Sfission'] = slope_coef / K_indv
        model_par['K_group'] = K_group_def / 12        
        model_par['K_tot'] = 0
    elif mode == 2:
        model_par['delta_group'] = 0
        model_par['delta_tot'] = 1
        model_par['delta_indv'] = 1
        model_par['gr_Sfission'] = 0
        model_par['K_tot'] = K_tot_def       
        model_par['K_group'] = 0
    elif mode == 3:
        model_par['delta_group'] = 0
        model_par['delta_tot'] = 1     
        model_par['delta_indv'] = 0
        model_par['gr_Sfission'] = slope_coef / K_indv
        model_par['K_tot'] = K_tot_def / 12        
        model_par['K_group'] = 0       
    else:
        print('unkown model_mode, choose from [0:3]')
        raise ValueError
   
    return None
        
    
def set_model_par(settings):
    model_par_local = model_par.copy()
    set_model_mode(model_par_local, settings['model_mode'], settings['indv_K'])
    for key, val in settings.items():
        model_par_local[key] = val
    return model_par_local

def run_batch():
    for model_mode in model_mode_Vec:
        for cost in cost_Vec:
            for Kindv in Kindv_Vec:
                for mu in mu_Vec:
                    for NType in NType_Vec:
                        settings = {'model_mode': model_mode, 
                                    'indv_cost' : cost, 
                                    'indv_mutationR' : mu,
                                    'indv_K' : Kindv,
                                    'indv_NType' : NType}
                        
                        modelParCur = set_model_par(settings)
                        _ = mls.load_or_run_model(mainName, modelParCur)
                    
                                
        
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_batch()
    

