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
import plotParScan

"""============================================================================
Define parameters
============================================================================"""

override_data = False #set to true to force re-calculation
numCore = 4 #number of cores to run code on

#where to store output?


data_folder = Path(str(Path.home())+"/ownCloud/MLS_GroupDynamics_shared/Data/")
fig_Folder = Path(str(Path.home())+"/ownCloud/MLS_GroupDynamics_shared/Figures/")
mainName = 'scanFissionModes_Zoom'

#setup variables to scan
offspr_sizeVec = np.arange(0.005, 0.151, 0.005) 
offspr_fracVec = np.arange(0.5, 1.01, 0.05) 


#set other parameters
model_par = {
    # solver settings
    "maxT":             3000,  # total run time
    "minT":             400,   # min run time
    "sampleInt":        1,     # sampling interval
    "mav_window":       100,   # average over this time window
    "rms_window":       100,   # calc rms change over this time window
    "rms_err_trNCoop":  2E-2,  # when to stop calculations
    "rms_err_trNGr":    0.1,  # when to stop calculations
    # settings for initial condition
    "init_groupNum":    100,  # initial # groups
    # initial composition of groups (fractions)
    "init_groupComp":   [0.5, 0, 0.5, 0],
    "init_groupDens":   100,  # initial total cell number in group
    # settings for individual level dynamics
    "indv_cost":        0.05,  # cost of cooperation
    "indv_deathR":      0.001, # death rate individuals
    "indv_mutationR":   1E-2,  # mutation rate to cheaters
    "indv_interact":    1,      #0 1 to turn off/on crossfeeding
    # setting for group rates
    'gr_Sfission':      0.,    # fission rate = (1 + gr_Sfission * N)/gr_tau
    'gr_Sextinct':      0.,    # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
    'gr_K':             2E3,   # total carrying capacity of cells
    'gr_tau':           100,   # relative rate individual and group events
    # settings for fissioning
    'offspr_size':      0.5,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'

}

#setup name to save files
parName = '_cost%.0e_mu%.0e_tau%i_interact%i_dr%.0e_grK%.0e_sFis%.0e_sExt%.0e' % (
    model_par['indv_cost'], model_par['indv_mutationR'], 
    model_par['gr_tau'], model_par['indv_interact'],
    model_par['indv_deathR'], model_par['gr_K'],
    model_par['gr_Sfission'], model_par['gr_Sextinct'])
dataFileName = mainName + parName 
dataFilePath = data_folder / (dataFileName + '.npz')


"""============================================================================
Define functions
============================================================================"""


#set model parameters for fission mode
def set_fission_mode(offspr_size, offspr_frac):
    #copy model par (needed because otherwise it is changed in place)
    model_par_local = model_par.copy()
    model_par_local['offspr_size'] = offspr_size
    model_par_local['offspr_frac'] = offspr_frac
    return model_par_local


# run model
def run_model():
   #create model paremeter list for all valid parameter range
    modelParList = []
    for offspr_size in offspr_sizeVec:
        for offspr_frac in offspr_fracVec:
            if offspr_frac >= offspr_size and offspr_frac <= (1 - offspr_size):
                modelParList.append(set_fission_mode(offspr_size, offspr_frac))

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
    np.savez(dataFilePath, statData=statData, distFCoop=distFCoop, distGrSize=distGrSize,
             offspr_sizeVec=offspr_sizeVec, offspr_fracVec=offspr_fracVec,
             modelParList=modelParList, date=datetime.datetime.now())

    return (statData, distGrSize)


# checks if model parmaters have changed compared to file saved on disk
def check_model_par(model_par_load, parToIgnore):
    rerun = False
    for key in model_par_load:
        if not (key in parToIgnore):
            if model_par_load[key] != model_par[key]:
                print('Parameter "%s" has changed, rerunning model!' % 'load')
                rerun = True
    return rerun


# Load model is datafile found, run model if not found or if settings have changed
def load_or_run_model():
    # need not check these parameters
    parToIgnore = ('offspr_size', 'offspr_frac')
    loadName = dataFilePath
    if loadName.is_file():
        # open file and load data
        data_file = np.load(loadName, allow_pickle=True)
        statData = data_file['statData']
        rerun = check_model_par(data_file['modelParList'][0], parToIgnore)
        data_file.close()
    else:
        # cannot load, need to rerun model
        rerun = True
        print('Model data not found, running model')
    if rerun or override_data:
        # rerun model
        statData = run_model()
    return statData


#run parscan and make figure
if __name__ == "__main__":
    statData = load_or_run_model()
    plotParScan.make_fig(dataFileName)

