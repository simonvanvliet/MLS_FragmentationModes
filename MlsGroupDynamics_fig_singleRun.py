#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-06-26

Run single model and export to csv

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
import pandas as pd

#SET OUTPUT FILENAME
fileName = 'singleRunTest'

#SET model parameters
model_par = {
        #time and run settings
        "maxT":             2000,  # total run time
        "maxPopSize":       50000,  #stop simulation if population exceeds this number
        "minT":             100,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       100,    # average over this time window
        "rms_window":       100,    # calc rms change over this time window
        "rms_err_trNCoop":  5E-2,   # when to stop calculations
        "rms_err_trNGr":    1E-1,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    20,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   75,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.05,   # cost of cooperation
        "indv_mutR":        1E-2,   # mutation rate to cheaters
        "indv_migrR":       1E-3,      # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0,
        'gr_SFis':          10,     # measured in units of 1 / indv_K
        'grp_tau':          1,
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            1E4,    # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.5,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5,    # offspr_size < offspr_frac < 1-offspr_size'
         # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
    }
  




# run model
output, _, _ = mls.run_model(model_par)

# convert to pandas dataframe and export
df = pd.DataFrame.from_records(output)
dataName = fileName + '.pkl'
df.to_pickle(dataName)




