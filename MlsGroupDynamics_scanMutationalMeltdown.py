#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

Last Update Oct 22 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import MlsGroupDynamics_main as mls
from joblib import Parallel, delayed
import numpy as np
#import plotParScan

"""============================================================================
Define parameters
============================================================================"""

override_data = False #set to true to force re-calculation
numCore = 45 #number of cores to run code on

mainName = 'MutationMeltdown_March9'
numRepeat = 3

#setup variables to scan
mu_Vec = np.logspace(0,-7,29) #8+7n

offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07) 
mode_set = np.array([[8, 2, 0.1, 0,    0,    0, 0],
                     [0, 0,   0, 0, 1e-2, 1e-1, 1]])
modeNames = ['gr_SFis', 'indv_migrR']
mode_vec = np.arange(mode_set.shape[1])
par0_vec = np.array([0.01])
par1_vec = np.array([1, 2, 3, 4])
parNames = ['indv_cost','indv_NType']
K_tot_def = 30000



model_par = {
        #time and run settings
        "maxT":             10000,  # total run time
        "maxPopSize":       30000,  #stop simulation if population exceeds this number
        "minT":             200,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       200,    # average over this time window
        "rms_window":       200,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,   # when to stop calculations
        "rms_err_trNGr":    5E-1,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    100,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   50,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
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
        'gr_CFis':          1/100,
        'gr_SFis':          1/50,
        # extinction rate
        'delta_grp':        0,      # exponent of denisty dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            30000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.8,    # offspr_size < offspr_frac < 1-offspr_size'
         # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
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
                'indv_mutR'     : 'mutR', 
                'indv_migrR'    : 'migR', 
                'indv_K'        : 'kInd', 
                'K_grp'         : 'kGrp', 
                'K_tot'         : 'kTot',
                'run_idx'       : 'indx'}




"""============================================================================
Define functions
============================================================================"""


def create_data_name(mainName, model_par):
    parListName = ['indv_K', 'gr_CFis','K_tot',
                   'indv_asymmetry']

    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    dataFileName = mainName + parName 
        
    return dataFileName

def set_model_par(model_par, settings):
    model_par_local = model_par.copy()
    for key, val in settings.items():
        model_par_local[key] = val
        
    if model_par_local['gr_SFis'] == 0:
        model_par_local['K_tot']  = K_tot_def * 5
    else:
        model_par_local['K_tot'] = K_tot_def    
        
    return model_par_local


# run model
def create_model_par_list(model_par):
   #create model paremeter list for all valid parameter range
    modelParList = []
    run_idx = -1
    
    for mode in mode_vec:            
        for par0 in par0_vec:
            for par1 in par1_vec:                
                for offspr_size in offspr_size_Vec:
                    for offspr_frac in offspr_frac_Vec:
                        inBounds = offspr_frac >= offspr_size and \
                                   offspr_frac <= (1 - offspr_size)
                        if inBounds:
                            settings = {'gr_SFis'     : mode_set[0, mode],
                                        'indv_migrR'  : mode_set[1, mode],
                                        parNames[0]   : par0,
                                        parNames[1]   : par1,
                                        'offspr_size' : offspr_size,
                                        'offspr_frac' : offspr_frac,
                                        'run_idx'     : mode
                                        }
                            curPar = set_model_par(model_par, settings)
                            
                            
                            modelParList.append(curPar)
    return modelParList

def run_meltdown_scan(model_par, numRepeat):
    
    maxMu = np.full(numRepeat, np.nan)
    maxLoad = np.full(numRepeat, np.nan)
    NTot = np.full(numRepeat, np.nan)
    NCoop = np.full(numRepeat, np.nan)
    NCoop = np.full(numRepeat, np.nan)
    NGrp = np.full(numRepeat, np.nan)

    for rr in range(numRepeat):
        #init state
        idx = 0
        hasMeltdown = True
        #reduce mutation rate till community can survive
        while hasMeltdown:
            model_par['indv_mutR'] = mu_Vec[idx]
            output, _ , _ = mls.single_run_finalstate(model_par)
        
            #if community survives, found max mutation burden
            if output['NTot'][-1] > 0:
                hasMeltdown = False
                maxMu[rr] = model_par['indv_mutR']
                maxLoad[rr] = model_par['indv_mutR'] * model_par['indv_cost']
                NTot[rr] = output['NTot_mav']
                NCoop[rr] = output['NCoop_mav']
                NGrp[rr] = output['NGrp_mav']
            else:
                idx += 1
                
            #end when lowest possible mutation rate has been reached
            if idx >= mu_Vec.size:
                hasMeltdown = False
                
        if rr==0:
            outputMat = output  
        else:
            outputMat = np.vstack((outputMat, output))
    
    return (maxMu, maxLoad, NTot, NCoop, NGrp, outputMat)

def run_model(mainName, model_par, numRepeat):
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)
    modelParList = modelParList[:2]
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(run_meltdown_scan)(par, numRepeat) for par in modelParList)

    # process and store output
    maxMu, maxLoad, NTot, NCoop, NGrp, output = zip(*results)
    statData   = np.vstack(output) 
    maxMu      = np.vstack(maxMu)
    maxLoad    = np.vstack(maxLoad)
    NTot       = np.vstack(NTot)
    NCoop      = np.vstack(NCoop)
    NGrp       = np.vstack(NGrp)

    #store output to disk 
    dataFileName = create_data_name(mainName, model_par)
    dataFilePath = dataFileName + '.npz'
    np.savez(dataFilePath, 
             statData    = statData,
             maxMu       = maxMu,
             maxLoad     = maxLoad,
             NTot        = NTot,
             NCoop       = NCoop,
             NGrp        = NGrp,
             numRepeat  = numRepeat,
             offsprSize = offspr_size_Vec, 
             offsprFrac = offspr_frac_Vec,
             mutR       = mu_Vec,
             mode_vec   = mode_vec,
             par0_vec   = par0_vec,
             par1_vec   = par1_vec,
             mode_set   = mode_set,
             modeNames  = modeNames,
             parNames   = parNames,
             parList    = modelParList
             )

    return None

#run parscan and make figure
if __name__ == "__main__":
    statData = run_model(mainName, model_par, numRepeat)




