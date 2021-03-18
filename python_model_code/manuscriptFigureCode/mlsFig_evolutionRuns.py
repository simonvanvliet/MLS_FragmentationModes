#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-11-16
Code for Figure 6
Code runs multiple evolution run, each run is stored on disk independently
Use mlsFig_evolutionFitnessLandscape with same settings to create reference parameter space scans

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""
import sys
sys.path.insert(0, '..')

from mainCode import MlsGroupDynamics_evolve as mls
from mainCode import MlsGroupDynamics_utilities as util
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

""" 
SET SETTINGS
"""
#SET fileName is appended to file name
fileName = 'evolutionRun'
#SET number of cores to use
numCore = 20
#SET group fission rates to scan
gr_Sfission_Vec = np.array([0.1, 4]) # for S = 0, use K = 2E5; for S > 0 (0.1 or 4), use K = 3E4
#SET parName and par0Vec to scan over any parameter of choice
par0Name = 'indv_NType'
par0Vec = np.array([1, 2])
#SET initial locations of evolution runs
init_Aray = np.array([[0.05,0.05],[0.05,0.5],[0.05,0.95],[0.25,0.5],[0.45,0.5]])
numInit = init_Aray.shape[0]


#SET Model default settings
model_par = {
    #time and run settings
    "maxPopSize":       0,
    "maxT":             1E6,    # total run time
    "sampleInt":        1E3,    # sampling interval
    "mav_window":       1E4,    # average over this time window
    "rms_window":       1E4,    # calc rms change over this time window
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
    'K_tot':            1E4,  # carrying capacity of total individuals.
    'delta_size':       0,      # exponent of size dependence
    # initial settings for fissioning
    'offspr_sizeInit':  0.25,   # offspr_size <= 0.5 and
    'offspr_fracInit':  0.5     # offspr_size < offspr_frac < 1-offspr_size'
    }
  
""" 
Function definitions
"""        

def run_single(model_par, mainName):
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
    
    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    fileName = mainName + parName + '.npz'
    fileNamePkl = mainName + parName + '.pkl'
    
    #run model and save data to disk
    try:
        outputMat, traitDistr = mls.run_model(model_par)  
        np.savez(fileName, output=outputMat, traitDistr=traitDistr, model_par=[model_par])
        try:
            df = pd.DataFrame.from_records(outputMat)
            df.to_pickle(fileNamePkl)
        except:
            print("failure with export")
                
    except: 
        print("Failure with run")
 
                
def run_batch():
    """[Runs batch of evolution experiments]
    Returns:
        [list] -- [results of evolution runs]
    """
    #create list with model settings to run
    modelParList = []
    for gr_SFis in gr_Sfission_Vec:
        for par0 in par0Vec:
            for ii in range(numInit):
               
                #implement local settings    
                settings = {'gr_SFis'  : gr_SFis,
                            par0Name   : par0,
                            'offspr_sizeInit': init_Aray[ii, 0],
                            'offspr_fracInit': init_Aray[ii, 1]}
                #add settings to list
                modelParCur = util.set_model_par(model_par, settings)
                modelParList.append(modelParCur)
    
    # run model, use parallel cores     
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(run_single)(par, fileName) for par in modelParList)

    return results

#run parscan
if __name__ == "__main__":
    results = run_batch()

