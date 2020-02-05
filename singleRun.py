#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 2019

Last Update Oct 22 2019

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
import plotSingleRun as pltRun
import time

#set model parameters
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
        "indv_cost":        0.01,  # cost of cooperation
        "indv_mutationR":   1E-3,   # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_Sfission':      0,
        'gr_Cfission':      1/100,
        # extinction rate
        'delta_group':      0,      # exponent of denisty dependence on group #
        'K_group':          100,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            75000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.4,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
    }


model_mode = 1
K_group_def = 1000
K_tot_def = 60000
K_indv_def = 100

def set_model_mode(model_par, mode):
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
        model_par['gr_Sfission'] = 1 / K_indv_def
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
        model_par['gr_Sfission'] = 1 / K_indv_def
        model_par['K_tot'] = K_tot_def / 12        
        model_par['K_group'] = 0       
    else:
        print('unkown model_mode, choose from [0:3]')
        raise ValueError
   
    return None



#run model

# run code
start = time.time()
set_model_mode(model_par, model_mode)
output, distFCoop, distGrSize = mls.run_model(model_par)
pltRun.plot_single_run(model_par, output, distFCoop, distGrSize)

end = time.time()

# print timing
print("Elapsed time run 1 = %s" % (end - start))



