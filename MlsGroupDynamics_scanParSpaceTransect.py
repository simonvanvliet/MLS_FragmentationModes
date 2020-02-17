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

import itertools
import MlsGroupDynamics_main as mls
import datetime
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from pathlib import Path
 

"""============================================================================
Define parameters
============================================================================"""

override_data = True #set to true to force re-calculation
numCore = 40 #number of cores to run code on
numThread = 1 #number o threads per core
#where to store output?

data_folder = Path(".")
mainName = 'transact_Feb10'


#setup variables to scan
offspr_sizeVec = np.arange(0.01, 0.5001, 0.01)
mu_vec = np.array([1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
type_vec = np.arange(1,5)
slope_vec = np.linspace(0,0.5,11)

K_tot_def = 20000

#set other parameters
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
        'gr_Cfission':      1/100,
        'gr_Sfission':      1/50,
        # extinction rate
        'delta_grp':        0,      # exponent of denisty dependence on group #
        'K_grp':            0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            K_tot_def,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.8  # offspr_size < offspr_frac < 1-offspr_size'
    }


"""============================================================================
Define functions
============================================================================"""


parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_Cfission'   : 'fisC',
                'gr_Sfission'   : 'fisS',
                'indv_NType'    : 'nTyp', 
                'indv_asymmetry': 'asym',
                'indv_cost'     : 'cost', 
                'indv_mutR'     : 'mutR', 
                'indv_migrR'    : 'migR', 
                'indv_K'        : 'kInd', 
                'K_grp'         : 'kGrp', 
                'K_tot'         : 'kTot',
                'model_mode'    : 'mode',
                'slope_coef'    : 'sCof'}


def create_data_name(mainName, model_par):
    parListName = ['indv_cost', 'indv_migrR',
                   'indv_K', 'K_grp', 'K_tot',
                   'indv_asymmetry',
                   'delta_indv','delta_grp','delta_tot','delta_size',
                   'gr_Cfission']

    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    dataFileName = mainName + parName 
        
    
    return dataFileName


#set model parameters for fission mode
def set_fission_mode(offspr_size, indv_NType, indv_mutationR, Sfission):
    #copy model par (needed because otherwise it is changed in place)
    offspr_frac = 1 - offspr_size
    if Sfission == 0:
        K_tot = K_tot_def * 5
    else:
        K_tot = K_tot_def

    model_par_local = model_par.copy()
    model_par_local['offspr_size'] = offspr_size
    model_par_local['K_tot'] = K_tot
    model_par_local['offspr_frac'] = offspr_frac
    model_par_local['indv_NType'] = indv_NType
    model_par_local['indv_mutationR'] = indv_mutationR
    model_par_local['gr_Sfission'] = Sfission

    return model_par_local

# run model
def run_model():
    #create model paremeter list for all valid parameter range
    # *x unpacks variables stored in tuple x e.g. if x = (a1,a2,a3) than f(*x) = f(a1,a2,a3)
    # itertools.product creates all possible combination of parameters
    modelParList = [set_fission_mode(*x)
                    for x in itertools.product(*(offspr_sizeVec, type_vec, mu_vec, slope_vec))]

    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))

    with parallel_backend("loky", inner_max_num_threads=numThread):
        results = Parallel(n_jobs=nJobs, verbose=10, timeout=1.E8)(
            delayed(mls.single_run_finalstate)(par) for par in modelParList)
        
        
    dataFileName = create_data_name(mainName, model_par)
    dataFilePath = data_folder / (dataFileName + '.npz')    
    
    np.savez(dataFilePath, results=results,
             offspr_sizeVec = offspr_sizeVec,
             mu_vec = mu_vec,
             type_vec = type_vec,
             slope_vec=slope_vec,
             modelParList = modelParList, date=datetime.datetime.now())
    
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_model()

