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

#set model parameters
model_par = {
    # solver settings
    "maxT":             3000,  # total run time
    "minT":             250,   # min run time
    "sampleInt":        1,     # sampling interval
    "mav_window":       100,   # average over this time window
    "rms_window":       100,   # calc rms change over this time window
    "rms_err_treshold": 2E-20,  # when to stop calculations
    # settings for initial condition
    "init_groupNum":    100,  # initial # groups
    # initial composition of groups (fractions)
    "init_groupComp":   [0.5, 0, 0.5, 0],
    "init_groupDens":   100,  # initial total cell number in group
    # settings for individual level dynamics
    "indv_cost":        0.05,  # cost of cooperation
    "indv_deathR":      0.001,  # death rate individuals
    "indv_mutationR":   1E-2,  # mutation rate to cheaters
    "indv_interact":    1,  # 0 1 to turn off/on crossfeeding
    # setting for group rates
    'gr_Sfission':      0.,    # fission rate = (1 + gr_Sfission * N)/gr_tau
    # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
    'gr_Sextinct':      0.,
    'gr_K':             5E3,   # total carrying capacity of cells
    'gr_tau':           100,   # relative rate individual and group events
    # settings for fissioning
    'offspr_size':      0.5,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'

}

#run model
mls.single_run_with_plot(model_par)
