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
        "maxT":             10000,  # total run time
        "minT":             250,   # min run time
        "sampleInt":        1,     # sampling interval
        "mav_window":       400,   # average over this time window
        "rms_window":       400,   # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,  # when to stop calculations
        "rms_err_trNGr":    5E-1,  # when to stop calculations
        # settings for initial condition
        "init_groupNum":    20,  # initial # groups
        # initial composition of groups (fractions)
        "init_fCoop":       1,
        "init_groupDens":   10,  # initial total cell number in group
        # settings for individual level dynamics
        "indv_NType":       2,
        "indv_cost":        0.5,  # cost of cooperation
        "indv_K":           50,  # total group size at EQ if f_coop=1
        "indv_mutationR":   5E-2,  # mutation rate to cheaters
        "delta_indv":       1, # zero if death rate is simply 1/k, one if death rate decreases with group size
        # difference in growth rate b(j+1) = b(j) / asymmetry
        "indv_asymmetry":   1,
        # setting for group rates
        # fission rate
        'gr_Sfission':      1,
        'Nmin':             0., # fission rate is zero below Nmin
        'group_offset':     0., # fission rate is linear with slope gr_Sfission and intercept "offset" above Nmin
        # extinction rate
        'gr_Sextinct':      0.,
        'gr_K':             100,   # carrying capacity of groups
        'gr_tau':           100,   # relative rate individual and group events
        'delta_group':      0., 
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
    }


#run model

# run code
start = time.time()
output, distFCoop, distGrSize = mls.run_model(model_par)
pltRun.plot_single_run(model_par, output, distFCoop, distGrSize)

end = time.time()

# print timing
print("Elapsed time run 1 = %s" % (end - start))



