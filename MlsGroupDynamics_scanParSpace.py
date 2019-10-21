#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 2019

Last Update Oct 21 2019

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
import MlsGroupDynamics_utilities as util
from pathlib import Path
import datetime
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import math

"""============================================================================
Define parameters
============================================================================"""

override_data = False #set to true to force re-calculation

#where to store output?
data_folder = Path("Data/")
fig_Folder = Path("Figures/")
mainName = 'scanFissionModes'

#setup variables to scan
numX = 16
numY = 16
offspr_sizeVec = np.linspace(0, 0.5, numX) 
offspr_fracVec = np.linspace(0, 1, numX)

#set other parameters
model_par = {
    # solver settings
    "maxT":             1500,  # total run time
    "minT":             250,   # min run time
    "sampleInt":        1,     # sampling interval
    "mav_window":       100,   # average over this time window
    "rms_window":       100,   # calc rms change over this time window
    "rms_err_treshold": 2E-2,  # when to stop calculations
    # settings for initial condition
    "init_groupNum":    100,  # initial # groups
    # initial composition of groups (fractions)
    "init_groupComp":   [0.5, 0, 0.5, 0],
    "init_groupDens":   100,  # initial total cell number in group
    # settings for individual level dynamics
    "indv_cost":        0.05,  # cost of cooperation
    "indv_deathR":      0.001, # death rate individuals
    "indv_mutationR":   1E-2,  # mutation rate to cheaters
    # setting for group rates
    'gr_Sfission':      0.,    # fission rate = (1 + gr_Sfission * N)/gr_tau
    'gr_Sextinct':      0.,    # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
    'gr_K':             5E3,   # total carrying capacity of cells
    'gr_tau':           100,   # relative rate individual and group events
    # settings for fissioning
    'offspr_size':      0.5,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'

}

#setup name to save files
parName = '_cost%.0e_mu%.0e_tau%i' % (
    model_par['indv_cost'], model_par['indv_mutationR'], model_par['gr_tau'])
dataFileName = mainName + parName + '.npz'
dataFilePath = data_folder / dataFileName
figureName = mainName + parName + '.pdf'

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
    nJobs = min(len(modelParList), 4)
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

    return statData


# checks of model parmaters have changed compared to file saved on disk
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


#convert list of results to 2D matrix of offspring frac. size vs fraction of parent to offspring
def create_2d_matrix(offspr_sizeVec, offspr_fracVec, statData, fieldName):

    #get size of matrix
    numX = offspr_sizeVec.size
    numY = offspr_fracVec.size
    #init matrix to NaN
    dataMatrix = np.full((numY, numX), np.nan)

    #fill matrix
    for xx in range(numX):
        for yy in range(numY):
            #find items in list that have correct fissioning parameters for current location in matrix
            currXId = statData['offspr_size'] == offspr_sizeVec[xx]
            currYId = statData['offspr_frac'] == offspr_fracVec[yy]
            currId = np.logical_and(currXId, currYId)
            #extract output value and assign to matrix
            if currId.sum() == 1:
                dataMatrix[yy, xx] = np.asscalar(statData[fieldName][currId])
    return dataMatrix


#make heatmap of 2D matrix
def plot_heatmap(fig, ax, statData, dataName, rounTo):
    #convert 1D list to 2D matrix
    data2D = create_2d_matrix(
        offspr_sizeVec, offspr_fracVec, statData, dataName)
    
    #find max value 
    maxData = math.ceil(np.nanmax(data2D) / rounTo) * rounTo
    
    #plot heatmap
    im = ax.pcolormesh(offspr_sizeVec, offspr_fracVec, data2D,
                       cmap='plasma', vmin=0, vmax=maxData)
    #add colorbar
    fig.colorbar(im, ax=ax, orientation='horizontal',
                 label=dataName,
                 ticks=[0, maxData/2, maxData], 
                 fraction=0.5, pad=0.1)

    #make axis nice
    xRange = (offspr_sizeVec.min(), offspr_sizeVec.max())
    yRange = (offspr_fracVec.min(), offspr_fracVec.max())
    steps = (3, 3)
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    ax.set_xticks(np.linspace(*xRange, steps[0]))
    ax.set_yticks(np.linspace(*yRange, steps[1]))

    # set labels
    ax.set_xlabel('offspring frac. size')
    ax.set_ylabel('frac. parrent to offspring')

    return None


#plots result of parameter scan
def make_figure(statData):
    #open figure and set size
    fig = plt.figure()
    util.set_fig_size_cm(fig, 10, 10)

    #plot variables
    nR = 1
    nC = 2
    
    #plot average Cooperator density
    ax = plt.subplot(nR, nC, 1)
    plot_heatmap(fig, ax, statData, 'NCoop_mav', 500)

    #plot number of groups
    ax = plt.subplot(nR, nC, 2)
    plot_heatmap(fig, ax, statData, 'NGroup', 5)
    
    #clean up figure
    plt.tight_layout() 
    
    #save figure
    fig.savefig(fig_Folder / figureName,
                format="pdf", transparent=True)
        
    return None


#run parscan and make figure
if __name__ == "__main__":
    statData = load_or_run_model()
    make_figure(statData)

