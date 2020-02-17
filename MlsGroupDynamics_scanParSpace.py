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
from pathlib import Path
import datetime
from joblib import Parallel, delayed
import numpy as np
#import plotParScan

"""============================================================================
Define parameters
============================================================================"""

override_data = False #set to true to force re-calculation
numCoreDef = 40 #number of cores to run code on

#where to store output?


data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
data_folder = Path(".")

fig_Folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Figures/")
mainName = 'scan2D_Jan23'

#setup variables to scan
offspr_sizeVec = np.arange(0.01, 0.5, 0.034)
offspr_fracVec = np.arange(0.01, 1, 0.07) 


#set other parameters
model_par = {
        #time and run settings
        "maxT":             10000,  # total run time
        "maxPopSize":       20000,  #stop simulation if population exceeds this number
        "minT":             250,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       400,    # average over this time window
        "rms_window":       400,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,   # when to stop calculations
        "rms_err_trNGr":    5E-1,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    50,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   20,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.05,  # cost of cooperation
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,   # mutation rate to cheaters
        # group size control
        "indv_K":           50,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_Sfission':      0,
        'gr_Cfission':      1/100,
        # extinction rate
        'delta_group':      1,      # exponent of denisty dependence on group #
        'K_group':          1000,    # carrying capacity of groups
        'delta_tot':        0,      # exponent of denisty dependence on total #indvidual
        'K_tot':            4000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
    }






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




"""============================================================================
Define functions
============================================================================"""


def create_data_name(mainName, model_par):
    parListName = ['indv_cost', 'indv_mutR', 'indv_migrR',
                   'indv_K', 'K_grp', 'K_tot',
                   'indv_NType', 'indv_asymmetry',
                   'delta_indv','delta_grp','delta_tot','delta_size',
                   'gr_Cfission','gr_Sfission']

    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    dataFileName = mainName + parName 
        
    
    return dataFileName

#set model parameters for fission mode
def set_fission_mode(model_par, offspr_size, offspr_frac):
    #copy model par (needed because otherwise it is changed in place)
    model_par_local = model_par.copy()
    model_par_local['offspr_size'] = offspr_size
    model_par_local['offspr_frac'] = offspr_frac
    return model_par_local


# run model
def run_model(mainName, model_par, numCore):
   #create model paremeter list for all valid parameter range
    modelParList = []
    for offspr_size in offspr_sizeVec:
        for offspr_frac in offspr_fracVec:
            if offspr_frac >= offspr_size and offspr_frac <= (1 - offspr_size):
                modelParList.append(set_fission_mode(model_par, offspr_size, offspr_frac))

    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))
    results = Parallel(n_jobs=nJobs, verbose=9, timeout=1.E9)(
        delayed(mls.single_run_finalstate)(par) for par in modelParList)

    # process and store output
    Output, endDistFCoop, endDistGrSize = zip(*results)
    statData = np.vstack(Output)
    distFCoop = np.vstack(endDistFCoop)
    distGrSize = np.vstack(endDistGrSize)

    #store output to disk
    
    dataFileName = create_data_name(mainName, model_par)
    dataFilePath = data_folder / (dataFileName + '.npz')
    np.savez(dataFilePath, statData=statData, distFCoop=distFCoop, distGrSize=distGrSize,
             offspr_sizeVec=offspr_sizeVec, offspr_fracVec=offspr_fracVec,
             modelParList=modelParList, date=datetime.datetime.now())

    return (statData, distGrSize)


# checks if model parmaters have changed compared to file saved on disk
def check_model_par(model_par, model_par_load, parToIgnore):
    rerun = False
    for key in model_par_load:
        if not (key in parToIgnore):
            if model_par_load[key] != model_par[key]:
                print('Parameter "%s" has changed, rerunning model!' % key)
                rerun = True
    return rerun


# Load model is datafile found, run model if not found or if settings have changed
def load_or_run_model(mainName, model_par, numCore=numCoreDef):
    # need not check these parameters
    parToIgnore = ('offspr_size', 'offspr_frac')
    dataFileName = create_data_name(mainName, model_par)
    loadName = data_folder / (dataFileName + '.npz')
    if loadName.is_file():
        # open file and load data
        data_file = np.load(loadName, allow_pickle=True)
        statData = data_file['statData']
        rerun = check_model_par(model_par, data_file['modelParList'][0], parToIgnore)
        data_file.close()
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        statData = run_model(mainName, model_par, numCore)
    return statData


#run parscan and make figure
if __name__ == "__main__":
    statData = load_or_run_model(mainName, model_par)
    #plotParScan.make_fig(dataFileName)




