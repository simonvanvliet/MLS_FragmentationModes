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
import matplotlib
import matplotlib.pyplot as plt

#set model parameters
model_par = {
        # solver settings
        "maxT":             1000,  # total run time
        "minT":             200,   # min run time
        "sampleInt":        1,     # sampling interval
        "mav_window":       200,   # average over this time window
        "rms_window":       200,   # calc rms change over this time window
        "rms_err_trNCoop":    2E-2,  # when to stop calculations
        "rms_err_trNGr":    1E-1,  # when to stop calculations
        # settings for initial condition
        "init_groupNum":    10,  # initial # groups
        # initial composition of groups (fractions)
        "init_fCoop":       1,
        "init_groupDens":   50,  # initial total cell number in group
        # settings for individual level dynamics
        "indv_NType":       2,
        "indv_cost":        0.01,  # cost of cooperation
        "indv_K":           50,  # total group size at EQ if f_coop=1
        "indv_mutationR":   1E-3,  # mutation rate to cheaters
        # difference in growth rate b(j+1) = b(j) / asymmetry
        "indv_asymmetry":    1,
        # setting for group rates
        # fission rate = (1 + gr_Sfission * N)/gr_tau
        'gr_Sfission':      0.,
        # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
        'gr_Sextinct':      0.,
        'gr_K':             20,   # carrying capacity of groups
        'gr_tau':           100,   # relative rate individual and group events
        # settings for fissioning
        'offspr_size':      0.5,  # offspr_size <= 0.5 and
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



