#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 2019

Last Update Oct 22 2019

Run single evolution model and plot result


@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

============================================================================
Run Model and plot results
============================================================================"""

#load code
import sys
sys.path.insert(0, '..')

from mainCode import MlsGroupDynamics_evolve as mlse
import plotSingleRun_evolution as pltRun
import plotEvolutionMovie as evomo
import numpy as np
import time
from datetime import datetime

model_par = {
    #time and run settings
    "maxPopSize":       0,
    "maxT":             200,   # total run time
    "minT":             200,   # min run time
    "sampleInt":        1,      # sampling interval
    "mav_window":       5,    # average over this time window
    "rms_window":       5,    # calc rms change over this time window
    "rms_err_trNCoop":  0,   # when to stop calculations
    "rms_err_trNGr":    0,   # when to stop calculations
    # settings for initial condition
    "init_groupNum":    5,     # initial # groups
    "init_fCoop":       1,
    "init_groupDens":   50,     # initial total cell number in group
    # settings for individual level dynamics
    # complexity
    "indv_NType":       2,
    "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
    # mutation load
    "indv_cost":        0.01,  # cost of cooperation
    "indv_migrR":       0,   # mutation rate to cheaters
    # set mutation rates
    'mutR_type':        1E-6,
    'mutR_size':        1E-1, 
    'mutR_frac':        1E-1, 
    # group size control
    "indv_K":           100,     # total group size at EQ if f_coop=1
    "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
    # setting for group rates
    # fission rate
    'gr_Cfission':      1/100,
    'gr_Sfission':      2,
    'alpha_Fis':        1,
    # extinction rate
    'delta_grp':        0,      # exponent of denisty dependence on group #
    'K_grp':            0,    # carrying capacity of groups
    'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
    'K_tot':            5000,   # carrying capacity of total individuals
    'delta_size':       0,      # exponent of size dependence
    # initial settings for fissioning
    'offspr_sizeInit':  0.25,  # offspr_size <= 0.5 and
    'offspr_fracInit':  0.5  # offspr_size < offspr_frac < 1-offspr_size'
    }
    



#run model

# run code
start = time.time()
now = datetime.now();
output, traitDistr = mlse.run_model(model_par)
pltRun.plot_single_run(model_par, output, traitDistr)
end = time.time()



fileName = 'evolution_' + now.strftime("%y%m%d_%H%M") + '.npz'
movieName = 'evolution_' + now.strftime("%y%m%d_%H%M") + '.mp4'

np.savez(fileName, output=output, traitDistr=traitDistr,
             model_par=[model_par], date=now)

evomo = evomo.create_movie(traitDistr, movieName, fps=25, size=800)

# print timing
print("Elapsed time run 1 = %s" % (end - start))



