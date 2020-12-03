#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-06-26
Updated 2020-07-02

Run single model and export to Pandas dataframe

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

============================================================================
Run Model and plot results
============================================================================"""

import numpy as np

import sys
sys.path.insert(0, '..')

#load code
from mainCode import MlsGroupDynamics_main as mls
import pandas as pd


migr_vec = np.array([0, 1])
S_vec = np.array([0, 4])
NTypes_vec = np.array([1, 4])

#SET model parameters
model_par = {
        #time and run settings
        "maxT":             2000,  # total run time
        "maxPopSize":       40000,  #stop simulation if population exceeds this number
        "minT":             25000,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       200,    # average over this time window
        "rms_window":       200,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-2,   # when to stop calculations
        "rms_err_trNGr":    5E-2,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    100,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   50,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       1,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,   # cost of cooperation
        "indv_mutR":        1E-7,   # mutation rate to cheaters
        "indv_migrR":       1,      # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0.05,
        'gr_SFis':          0,     # measured in units of 1 / indv_K
        'grp_tau':          1,
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            1E5,    # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.01,  # offspr_size <= 0.5 and
        'offspr_frac':      0.01,    # offspr_size < offspr_frac < 1-offspr_size'
         # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
    }
  
for migr in migr_vec:
    model_par['indv_migrR'] = migr
    for S in S_vec:
        model_par['gr_SFis'] = S
        for NType in NTypes_vec:
            model_par['indv_NType'] = NType
            for i in (1, 2, 3):
                if i == 1:
                    model_par['offspr_size'] = 0.01
                    model_par['offspr_frac'] = 0.01
                    fileName = 'test_migr' + str(model_par['indv_migrR']) + '_S' + str(model_par['gr_SFis']) + '_NTypes' + str(model_par['indv_NType']) + '_s' + str(model_par['offspr_size']) + '_n' + str(model_par['offspr_frac'])
                    # run model
                    output, _, _ = mls.run_model(model_par)
                    # convert to pandas dataframe and export
                    df = pd.DataFrame.from_records(output)
                    dataName = fileName + '.pkl'
                    df.to_pickle(dataName)
                elif i == 2:
                    model_par['offspr_size'] = 0.01
                    model_par['offspr_frac'] = 0.99
                    fileName = 'test_migr' + str(model_par['indv_migrR']) + '_S' + str(model_par['gr_SFis']) + '_NTypes' + str(model_par['indv_NType']) + '_s' + str(model_par['offspr_size']) + '_n' + str(model_par['offspr_frac'])
                    # run model
                    output, _, _ = mls.run_model(model_par)
                    # convert to pandas dataframe and export
                    df = pd.DataFrame.from_records(output)
                    dataName = fileName + '.pkl'
                    df.to_pickle(dataName)
                elif i == 3:
                    model_par['offspr_size'] = 0.49
                    model_par['offspr_frac'] = 0.5
                    fileName = 'test_migr' + str(model_par['indv_migrR']) + '_S' + str(model_par['gr_SFis']) + '_NTypes' + str(model_par['indv_NType']) + '_s' + str(model_par['offspr_size']) + '_n' + str(model_par['offspr_frac'])
                    # run model
                    output, _, _ = mls.run_model(model_par)
                    # convert to pandas dataframe and export
                    df = pd.DataFrame.from_records(output)
                    dataName = fileName + '.pkl'
                    df.to_pickle(dataName)   




