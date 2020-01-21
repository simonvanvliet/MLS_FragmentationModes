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
numCore = 40 #number of cores to run code on

#where to store output?


data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
data_folder = Path(".")

fig_Folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Figures/")
mainName = 'scan2D_Jan15'

#setup variables to scan
offspr_sizeVec = np.arange(0.01, 0.5, 0.034)
offspr_fracVec = np.arange(0.01, 1, 0.07) 


#set other parameters
model_par = {
        "maxT":             10000,  # total run time
        "minT":             250,   # min run time
        "sampleInt":        1,     # sampling interval
        "mav_window":       400,   # average over this time window
        "rms_window":       400,   # calc rms change over this time window
        "rms_err_trNCoop":  1E-1,  # when to stop calculations
        "rms_err_trNGr":    5E-1,  # when to stop calculations
        # settings for initial condition
        "init_groupNum":    20,  # initial # groups
        # initial composition of groups (fractions)
        "init_fCoop":       1,
        "init_groupDens":   10,  # initial total cell number in group
        # settings for individual level dynamics
        "indv_NType":       2,
        "indv_cost":        0.0005,  # cost of cooperation
        "indv_K":           50,  # total group size at EQ if f_coop=1
        "indv_mutationR":   1E-3,  # mutation rate to cheaters
        "delta_indv":       1, # zero if death rate is simply 1/k, one if death rate decreases with group size
        # difference in growth rate b(j+1) = b(j) / asymmetry
        "indv_asymmetry":   1,
        # setting for group rates
        # fission rate
        'gr_Sfission':      0,
        'Nmin':             0., # fission rate is zero below Nmin
        'group_offset':     1., # fission rate is linear with slope gr_Sfission and intercept "offset" above Nmin
        # extinction rate
        'gr_Sextinct':      0.,
        'gr_K':             2000,   # carrying capacity of groups
        'gr_tau':           100,   # relative rate individual and group events
        'delta_group':      0., 
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
    }

#setup name to save files
parName = '_NSpecies%i_Assym%.0g_cost%.0g_mu%.0g_tau%i_indvK%.0e_grK%.0g_sFis%.0g_sExt%.0g_Nmin%.0g_offset%.0g_deltaind%.0g_deltagr%.0g' % (
    model_par['indv_NType'],model_par['indv_asymmetry'],
    model_par['indv_cost'], model_par['indv_mutationR'], 
    model_par['gr_tau'], 
    model_par['indv_K'], model_par['gr_K'],
    model_par['gr_Sfission'], model_par['gr_Sextinct'],
    model_par['Nmin'], model_par['group_offset'], 
    model_par['delta_indv'], model_par['delta_group'])
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
    #plotParScan.make_fig(dataFileName)

