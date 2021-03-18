#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-11-20

Code runs multiple 2D parameter space scans, each run is stored on disk independently
Use as background to plot results of mlsFig_evolutionRuns 

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""
import sys
sys.path.insert(0, '..')

import numpy as np
import datetime
from joblib import Parallel, delayed
import pandas as pd
from mainCode import MlsGroupDynamics_main as mls
from mainCode import MlsGroupDynamics_utilities as util

""" 
SET SETTINGS
"""
#SET fileName is appended to file name
fileNameBase = 'evolutionFitnessLandscape'

#SET number of cores to use
nCore = 20;

#SET group fission rates to scan
gr_Sfission_Vec = np.array([0.1, 4])

#SET parName and par0Vec to scan over any parameter of choice
par0Name = 'indv_NType'
par0Vec = np.array([1, 2])

#SET initial locations of evolution runs
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07) 

#SET nr of replicates
nReplicate = 5

#SET Population size
K_totSnon0 = 3E3 #SFis > 0
K_totS0 = 2E5 #SFis = 0

#SET Model default settings
model_par_def = {    
    #time and run settings
    "maxT":             5000,  # total run time
    "maxPopSize":       1000000,  #stop simulation if population exceeds this number
    "minT":             400,    # min run time
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
    "indv_cost":        0.01,  # cost of cooperation
    "indv_migrR":       0,   # mutation rate to cheaters
    # set mutation rates
    'indv_mutR':        1E-3,
    'indv_tau':         1,
    # group size control
    "indv_K":           100,     # total group size at EQ if f_coop=1
    "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
    # setting for group rates
    # fission rate
    'gr_CFis':          0.01,
    'gr_SFis':          0,
    'grp_tau':          1,
    # extinction rate
    'delta_grp':        0,      # exponent of denisty dependence on group #
    'K_grp':            0,    # carrying capacity of groups
    'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
    'K_tot':            2E5,   # carrying capacity of total individuals
    'delta_size':       0,      # exponent of size dependence
    # initial settings for fissioning
    'offspr_size':      0.25,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5,  # offspr_size < offspr_frac < 1-offspr_size'
    # extra settings
    'run_idx':          1,
    'replicate_idx':    1,
    'perimeter_loc':    0
    }
 
parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_CFis'       : 'fisC',
                'gr_SFis'       : 'fisS',
                'alpha_b'       : 'alph',
                'indv_NType'    : 'nTyp', 
                'indv_asymmetry': 'asym',
                'indv_cost'     : 'cost', 
                'indv_mutR'     : 'mutR', 
                'indv_migrR'    : 'migR', 
                'indv_K'        : 'kInd', 
                'K_grp'         : 'kGrp', 
                'K_tot'         : 'kTot',
                'model_mode'    : 'mode',
                'slope_coef'    : 'sCof',
                'indv_tau'      : 'tInd'} 
  
 
def create_data_name(mainName, model_par):
    parListName = ['indv_NType', 
               'gr_SFis']
    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    dataFileName = mainName + parName 
    return dataFileName

def create_model_par_list(model_par):
    #create model paremeter list for all valid parameter range
    modelParList = []
    for repIdx in range(nReplicate):
        for offspr_size in offspr_size_Vec:
            for offspr_frac in offspr_frac_Vec:
                inBounds = offspr_frac >= offspr_size and \
                        offspr_frac <= (1 - offspr_size)
            
                if inBounds:
                    settings = {'offspr_size'  : offspr_size,
                                'offspr_frac'  : offspr_frac,
                                'replicate_idx': repIdx+1,
                                }
                    curPar = util.set_model_par(model_par, settings)
                    modelParList.append(curPar)
  
    return modelParList


# run single 2D scan
def run_2Dscan(model_par):
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)

    # run model, use parallel cores 
    nJobs = min(len(modelParList), nCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.run_model_steadyState_fig)(par) for par in modelParList)
    
    #add parameters to filename
    fileNamePar = create_data_name(fileNameBase, model_par)
    
    #store output to disk 
    fileNameTemp = fileNamePar + '_temp' + '.npy'
    np.save(fileNameTemp, results)
    
    #convert to pandas dataframe and export
    fileNameFull = fileNamePar + '.pkl'
    outputComb = np.hstack(results)
    df = pd.DataFrame.from_records(outputComb)
    df.to_pickle(fileNameFull)
    
    return None 
 
          
def run_batch():
    """[Runs batch of 2D parameter scans]
    
    Returns:
        None
    """
    for gr_Sfission in gr_Sfission_Vec:
        for par0 in par0Vec:
            if gr_Sfission == 0:
                K_tot = K_totS0 
            else:
                K_tot = K_totSnon0
            
            #assign settings for current 2D scan    
            settings = {'gr_SFis' : gr_Sfission,
                        par0Name  : par0,
                        'K_tot'   : K_tot}
            modelParCur = util.set_model_par(model_par_def, settings)
            
            #run 2D scan and store to disk
            run_2Dscan(modelParCur)
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_batch()
    

