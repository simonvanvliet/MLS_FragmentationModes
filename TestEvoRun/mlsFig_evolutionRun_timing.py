#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-11-20
Code runs multiple evolution run, each run is stored on disk independently
Optimized for SLURM array job
Use mlsFig_evolutionFitnessLandscape with same settings to create reference parameter space scans

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""
import sys
sys.path.insert(0, '..')

import MlsGroupDynamics_evolve_time as mls
import MlsGroupDynamics_utilities as util
import numpy as np
import pandas as pd
import itertools
import os

"""
SET SETTINGS
"""

#select which parameter combination to test (integer between 0-29)
run_index = 23

#SET fileName is appended to file name
mainName = 'evoRun-2020-12-02'
#SET group fission rates to scan
gr_Sfission_Vec = np.array([0, 0.1, 4])
#SET parName and par0Vec to scan over any parameter of choice
par0Name = 'indv_NType'
par0Vec = np.array([1, 2])
#SET initial locations of evolution runs
init_Aray = np.array([[0.05,0.05],[0.05,0.5],[0.05,0.95],[0.25,0.5],[0.45,0.5]])
numInit = init_Aray.shape[0]

#SET Population size
K_totSnon0 = 2.5E3 #SFis > 0
K_totS0 = 3E4 #SFis = 0

#SET Model default settings
model_par_def = {
    #time and run settings
    "maxPopSize":       0,
    "maxT":             1E3,    # total run time
    "sampleInt":        1E2,    # sampling interval
    "mav_window":       5E2,    # average over this time window
    "rms_window":       5E2,    # calc rms change over this time window
    # settings for initial condition
    "init_groupNum":    100,    # initial # groups
    "init_fCoop":       1,
    "init_groupDens":   50,     # initial total cell number in group
    # settings for individual level dynamics
    # complexity
    "indv_NType":       1,
    "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
    # mutation load
    "indv_cost":        0.01,   # cost of cooperation
    "indv_migrR":       0,      # migration rate
    # set mutation rates
    'mutR_type':        1E-3,   # mutation rate  between cooperator and cheater
    'mutR_size':        1E-2,   # mutation rate in offsspring size trait value
    'mutR_frac':        1E-2,   # mutation rate in offsspring fraction trait value
    'indv_tau' :        1,      # multipies individual rates
    # group size control
    "indv_K":           100,    # total group size at EQ if f_coop=1
    "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
    # setting for group rates
    # fission rate
    'gr_CFis':          0.01,
    'gr_SFis':          0,      # measured in units of 1 / indv_K
    'grp_tau':          1,      # constant multiplies group rates
    # extinction rate
    'delta_grp':        0,      # exponent of denisty dependence on group #
    'K_grp':            0,      # carrying capacity of groups
    'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
    'K_tot':            2E5,  # carrying capacity of total individuals.
    'delta_size':       0,      # exponent of size dependence
    # initial settings for fissioning
    'offspr_sizeInit':  0.25,   # offspr_size <= 0.5 and
    'offspr_fracInit':  0.5     # offspr_size < offspr_frac < 1-offspr_size'
    }

parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_CFis'       : 'fisC',
                'gr_SFis'       : 'fisS',
                'indv_NType'    : 'nTyp',
                'indv_asymmetry': 'asym',
                'indv_cost'     : 'cost',
                'mutR_type'     : 'muTy',
                'mutR_size'     : 'muSi',
                'mutR_frac'     : 'muFr',
                'indv_migrR'    : 'migR',
                'indv_K'        : 'kInd',
                'K_grp'         : 'kGrp',
                'K_tot'         : 'kTot',
                'offspr_sizeInit':'siIn',
                'offspr_fracInit':'frIn',
                'indv_tau'      : 'tInd'}

parListName = ['indv_NType',
               'gr_SFis',
               'offspr_sizeInit',
               'offspr_fracInit']

"""
Function definitions
"""

modelParList = [x for x in itertools.product(gr_Sfission_Vec, par0Vec, range(numInit))]

def report_number_runs():
    return len(modelParList)

def run_single(runIdx, folder):
    print('start run idx', runIdx)
    #implement local settings
    runIdx = int(runIdx)
    
    if modelParList[runIdx][0] == 0:
        K_tot = K_totS0
    else:
        K_tot = K_totSnon0
    
    
    settings = {'gr_SFis'  : modelParList[runIdx][0],
                par0Name   : modelParList[runIdx][1],
                'K_tot'     : K_tot,
                'offspr_sizeInit': init_Aray[modelParList[runIdx][2], 0],
                'offspr_fracInit': init_Aray[modelParList[runIdx][2], 1]}
    #add settings to list
    model_par = util.set_model_par(model_par_def, settings)

    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    fileName = folder + mainName + parName + '.npz'
    fileNamePkl = folder + mainName + parName + '.pkl'

    #run model and save data to disk
    outputMat, traitDistr = mls.run_model(model_par)
    np.savez(fileName, output=outputMat, traitDistr=traitDistr, model_par=[model_par])
    df = pd.DataFrame.from_records(outputMat)
    df.to_pickle(fileNamePkl)



#run parscan
if __name__ == "__main__":
    run_single(run_index, os.getcwd())
