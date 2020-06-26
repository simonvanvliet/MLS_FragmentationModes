#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

Last Update Oct 22 2019

Scans model parameters within full 2D parameter space
Output stored on disk
Plot with plotScanStates

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
import itertools

"""============================================================================
Define parameters
============================================================================"""

numCore = 44 #number of cores to run code on

#set name of output
mainName = 'April13'

#setup variables to scan

#set 2D parameter grid
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07) 

#set other parameters to scan
parNames = ['gr_SFis', 'delta_indv', 'alpha_Fis'] #parameter keys
par0_vec = np.array([1, 4, 8]) #parameter values
par1_vec = np.array([1, 0]) #parameter values
par2_vec = np.array([1, 2, 5]) #parameter values

#set constant model settings
K_tot_def = 10000
K_tot_multiplier = 6 #if SFis=0 increase K_tot by this factor  


model_par = {
        #time and run settings
        "maxT":             2000,  # total run time
        "maxPopSize":       40000,  #stop simulation if population exceeds this number
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
        "indv_NType":       1,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0,   # cost of cooperation
        "indv_mutR":        0,   # mutation rate to cheaters
        "indv_migrR":       0,      # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0,
        'gr_SFis':          0,
        'alpha_b':          0,
        # extinction rate
        'delta_grp':        0,      # exponent of denisty dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            K_tot_def,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.8,    # offspr_size < offspr_frac < 1-offspr_size'
         # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
    }
  
#set abbreviated parameter names
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
                'run_idx'       : 'indx'}

"""============================================================================
Define functions
============================================================================"""

#create data name from parameter values
def create_data_name(mainName, model_par):
    #set parameters and order to include in file name
    parListName = ['indv_K', 'gr_CFis','K_tot',
                   'indv_NType', 'indv_asymmetry']
    #create name string
    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    dataFileName = mainName + parName 
        
    return dataFileName

#set model parameters for fission mode
def set_model_par(model_par, settings):
    #copy model par (needed because otherwise it is changed in place)
    model_par_local = model_par.copy()
    
    #set model parameters
    for key, val in settings.items():
        model_par_local[key] = val
        
    #adjust K_tot if needed
    if model_par_local['gr_SFis'] == 0:
        model_par_local['K_tot']  = K_tot_def * K_tot_multiplier
    else:
        model_par_local['K_tot'] = K_tot_def
                               
    return model_par_local

# run model
def create_model_par_list(model_par):
   #create model paremeter list for all valid parameter range
    modelParList = []
    run_idx = -1
    
    # itertools.product creates all possible combination of parameters
    for parValues in itertools.product(*(par0_vec, par1_vec, par2_vec)):
        run_idx = -1
   
        for offspr_size in offspr_size_Vec:
            for offspr_frac in offspr_frac_Vec:
                inBounds = offspr_frac >= offspr_size and \
                            offspr_frac <= (1 - offspr_size)
                if inBounds:
                    perimeter_loc = offspr_size if offspr_frac>=0.5 else (1 - offspr_size)
                    
                    settings = {parNames[0]    : parValues[0],
                                parNames[1]    : parValues[1],
                                parNames[2]    : parValues[2],
                                'offspr_size'  : offspr_size,
                                'offspr_frac'  : offspr_frac,
                                'run_idx'      : run_idx,
                                'perimeter_loc': perimeter_loc
                                }

                    curPar = set_model_par(model_par, settings)
                    modelParList.append(curPar)
    return modelParList

# run model code
def run_model(mainName, model_par, numCore):
    #get model parameters to scan
    modelParList = create_model_par_list(model_par)
        
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.single_run_finalstate)(par) for par in modelParList)

    # process and store output
    output, distFCoop, distGrSize = zip(*results)
    
    #store output to disk 
    dataFileName = create_data_name(mainName, model_par)
    dataFilePath = dataFileName + '_temp' + '.npz'
    np.savez(dataFilePath, 
             results   = results)

    #store output to disk 
    dataFileName = create_data_name(mainName, model_par)
    dataFilePath = dataFileName + '.npz'
    np.savez(dataFilePath, 
             results    = results,
             offsprSize = offspr_size_Vec, 
             offsprFrac = offspr_frac_Vec,
             parNames   = parNames,
             par1       = par0_vec,
             par2       = par1_vec,
             par3       = par2_vec,
             parList    = modelParList)

    return None

#run parscan
if __name__ == "__main__":
    statData = run_model(mainName, model_par, numCore)    

                            
