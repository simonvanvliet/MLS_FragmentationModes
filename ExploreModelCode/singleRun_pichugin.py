#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 23 2020
Run single model of model with Pichugin et al 2017 like rate function  

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

============================================================================
Run Model and plot results
============================================================================"""

#load code
import MlsGroupDynamics_pichugin as mls
import plotSingleRun as pltRun
import time

#set model parameters
model_par = {
        #time and run settings
        "maxT":             100,  # total run time
        "maxPopSize":       1E5,  #stop simulation if population exceeds this number
        "startFit":         2E4,  #start fit of growth rate here
        "sampleInt":        0.05, # sampling interval
        "mav_window":       1,   # average over this time window
        "rms_window":       1,   # calc rms change over this time window
        # settings for initial condition
        "init_groupNum":    50,  # initial # groups
        "init_fCoop":       1,   # DO NOT CHANGE, code only works if init_fCoop = 1
        "init_groupDens":   20,  # initial total cell number in group
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
        # settings for fissioning
        'offspr_size':      0.05,  # offspr_size <= 0.5 and
        'offspr_frac':      0.05,  # offspr_size < offspr_frac < 1-offspr_size'
    }
  



#run model

# run code
start = time.time()
output = mls.run_model(model_par)
pltRun.plot_single_run(model_par, output, plotTypeCells='log', plotTypeGroup='log')
end = time.time()

# print timing
print("Elapsed time run 1 = %s" % (end - start))

output2 = mls.single_run_trajectories(model_par)


