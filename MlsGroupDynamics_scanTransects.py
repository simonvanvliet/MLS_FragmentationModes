#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

Last Update Oct 22 2019

Scans model parameters along transect following outer edge of parameter space
Starts upper left corner (0), goes to right corner (0.5), ends in lower left corner (1) 
Output stored on disk
Plot with plotParScanTransect

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import itertools
import MlsGroupDynamics_main as mls
import datetime
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from pathlib import Path
 

"""============================================================================
Define parameters
============================================================================"""

numCore = 45 #number of cores to run code on
numThread = 1 #number of threads per core

#set location and name of output
data_folder = Path(".")
mainName = 'transect_March1'


#setup variables to scan
#set 1D perimeter grid
perimeter_loc_vec = np.linspace(0,1,51)

#set other parameters to scan
parNames = ['indv_mutR', 'indv_NType', 'indv_migrR'] #parameter keys
par0_vec = np.array([1e-3, 1e-2, 1e-1]) #parameter values
par1_vec = np.array([1, 2, 3, 4]) #parameter values
par2_vec = np.array([1e-3, 1e-2, 1e-1, 1, 10]) #parameter values

#set constant model settings
K_tot_def = 30000
K_tot_multiplier = 6 #if SFis=0 increase K_tot by this factor  

model_par = {
        #time and run settings
        "maxT":             10000,  # total run time
        "maxPopSize":       30000,  #stop simulation if population exceeds this number
        "minT":             200,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       400,    # average over this time window
        "rms_window":       400,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,   # when to stop calculations
        "rms_err_trNGr":    5E-1,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    10,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   10,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,  # cost of cooperation
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,   # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          1/100,
        'gr_SFis':          0,
        'alpha_Fis':        1,
        # extinction rate
        'delta_grp':        0,      # exponent of denisty dependence on group #
        'K_grp':            0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            K_tot_def,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.8,  # offspr_size < offspr_frac < 1-offspr_size'
        # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
    }


"""============================================================================
Define functions
============================================================================"""

#set abbreviated parameter names
parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_CFis'       : 'fisC',
                'gr_SFis'       : 'fisS',
                'alpha_Fis'     : 'fisA',
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
                'perimeter_loc' : 'pLoc',
                'run_idx'       : 'idxR'}

#create data name from parameter values
def create_data_name(mainName, model_par):
    #set parameters and order to include in file name
    parListName = ['indv_cost', 'indv_migrR',
                   'indv_K', 'K_grp', 'K_tot',
                   'indv_asymmetry',
                   'delta_indv','delta_grp','delta_tot','delta_size',
                   'gr_CFis']

    #create name string
    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    dataFileName = mainName + parName 
    
    return dataFileName


#set model parameters for fission mode
def set_fission_mode(perimeter_loc, par0, par1, par2, parNames):
    #copy model par (needed because otherwise it is changed in place)
    model_par_local = model_par.copy()

    #set perimeter location
    if perimeter_loc <= 0.5:
        offspr_size = perimeter_loc
        offspr_frac = 1 - offspr_size
    else:
        offspr_size = 1 - perimeter_loc
        offspr_frac = offspr_size
        
    #set model parameters
    model_par_local['perimeter_loc'] = perimeter_loc
    model_par_local['offspr_size'] = offspr_size
    model_par_local['offspr_frac'] = offspr_frac
    model_par_local[parNames[0]] = par0
    model_par_local[parNames[1]] = par1
    model_par_local[parNames[2]] = par2
    
    #adjust K_tot if needed
    if model_par_local['gr_SFis'] == 0:
       model_par_local['K_tot']  = K_tot_def * K_tot_multiplier
    else:
        model_par_local['K_tot'] = K_tot_def
        
    return model_par_local

# run model code
def run_model():
    #create model parameter list for all valid parameter range
    # *x unpacks variables stored in tuple x e.g. if x = (a1,a2,a3) than f(*x) = f(a1,a2,a3)
    # itertools.product creates all possible combination of parameters
    modelParList = [set_fission_mode(*x, parNames)
                    for x in itertools.product(*(perimeter_loc_vec, par0_vec, par1_vec, par2_vec))]

    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))

    with parallel_backend("loky", inner_max_num_threads=numThread):
        results = Parallel(n_jobs=nJobs, verbose=10, timeout=1.E8)(
            delayed(mls.single_run_finalstate)(par) for par in modelParList)
        
    #save data    
    dataFileName = create_data_name(mainName, model_par)
    dataFilePath = data_folder / (dataFileName + '.npz')    
    
    np.savez(dataFilePath, results=results,
             perimeter_loc_vec = perimeter_loc_vec,
             par0_vec = par0_vec,
             par1_vec = par1_vec,
             par2_vec = par2_vec,
             parNames = parNames,
             modelParList = modelParList, 
             date = datetime.datetime.now())
    
    return None

#run parscan 
if __name__ == "__main__":
    run_model()

