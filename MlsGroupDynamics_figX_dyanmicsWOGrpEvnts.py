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
from joblib import Parallel, delayed


#SET OUTPUT FILENAME
fileName = 'multipleRunTest'

#SET model parameters
model_par = {
        #time and run settings
        "maxT":             5000,  # total run time
        "maxPopSize":       40000,  #stop simulation if population exceeds this number
        "minT":             200,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       200,    # average over this time window
        "rms_window":       200,    # calc rms change over this time window
        "rms_err_trNCoop":  5E-2,   # when to stop calculations
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
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,      # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0.01,
        'gr_SFis':          1,     # measured in units of 1 / indv_K
        'grp_tau':          0,     # constant multiplies group rates
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            3E4,    # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.01,  # offspr_size <= 0.5 and
        'offspr_frac':      0.01,    # offspr_size < offspr_frac < 1-offspr_size'
         # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
    }
  



modelParList = [model_par, model_par]

# run model

results = Parallel(n_jobs=2, verbose=9, timeout=1.E9)(
        delayed(mls.run_model)(par) for par in modelParList)

# process and store output
output, distFCoop, distGrSize = zip(*results)


    
    
#output, _, _ = mls.run_model(model_par)
#
## convert to pandas dataframe and export
#df = pd.DataFrame.from_records(output)
#dataName = fileName + '.pkl'
#df.to_pickle(dataName)
#



