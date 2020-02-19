#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Last Update Oct 23 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import matplotlib as mpl
import MlsGroupDynamics_plotUtilities as pltutl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#set name of file to load
fileName = 'transact_Feb10_cost0.01_migR0_kInd1e+02_kGrp0_kTot2e+04_asym1_dInd1_dGrp0_dTot1_dSiz0_fisC0.01.npz'

#set Folder
data_folder = Path(".")
fig_Folder = Path(
    "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Transect")

"""============================================================================
Set figure options 
============================================================================"""

font = {'family': 'Arial',
        'weight': 'light',
        'size': 6}

axes = {'linewidth': 0.5,
        'titlesize': 7,
        'labelsize': 6,
        'labelpad': 2,
        'spines.top': False,
        'spines.right': False,
        }

ticks = {'major.width': 0.5,
         'direction': 'in',
         'major.size': 2,
         'labelsize': 6,
         'major.pad': 2}

legend = {'fontsize': 6,
          'handlelength': 1.5,
          'handletextpad': 0.5,
          'labelspacing': 0.2}

figure = {'dpi': 300}
savefigure = {'dpi': 300,
              'transparent': True}

mpl.style.use('seaborn-ticks')
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('xtick', **ticks)
mpl.rc('ytick', **ticks)
#mpl.rc('ztick', **ticks)
mpl.rc('legend', **legend)
mpl.rc('figure', **figure)
mpl.rc('savefig', **savefigure)

"""============================================================================
Define functions
============================================================================"""


def make_fig(fileName, pathSave=fig_Folder, pathLoad=data_folder):
    #set folders of data

    #remove extension if needed
    if fileName[-4:] == '.npz':
        fileName = fileName[:-4]
     
    #load data
    loadName = pathLoad / (fileName + '.npz')
    data_file = np.load(loadName, allow_pickle=True)
    results = data_file['results']
    type_vec = data_file['type_vec']
    mu_vec = data_file['mu_vec']
    slope_vec = data_file['slope_vec']
    data_file.close()
    
    Output, _, _ = zip(*results)
    
    stateVar = ['NTot', 'NCoop', 'fCoop',
            'NGroup', 'groupSizeAv', 'groupSizeMed']
    
    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutR', 'indv_migrR', 'indv_asymmetry', 'delta_indv',
               'gr_Sfission', 'gr_Cfission', 'K_grp', 'K_tot', 'delta_grp', 'delta_tot', 'delta_size',
               'offspr_size', 'offspr_frac', 'run_time']

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVar]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVar]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3
    dType = np.dtype(dTypeList)
    statData = np.zeros(len(Output), dType)

    # store final state
    i = 0
    for data in Output:
        for var in stateVar:
            statData[var][i] = data[var]
            var_mav = var + '_mav'
            statData[var_mav][i] = data[var_mav]
        for par in parList:
            statData[par][i] = data[par]
        i += 1

    """============================================================================
    Make plot
    ============================================================================"""

    #things that can be plorred:
    #    'NA', 'NAprime', 'NB', 'NBprime',
    #    'NTot', 'NCoop', 'fCoop',
    #    'NGroup', 'groupSizeAv', 'groupSizeMed'
    #add _mav to get moving average value    

    parToPlot = ['NTot_mav', 'fCoop_mav', 'NGroup_mav']

    for curPar in parToPlot:
    
        fig = plt.figure()
        pltutl.set_fig_size_cm(fig, 60, 40)
        
        #plot variables
        nC = slope_vec.size
        nR = type_vec.size

        #loop over all variable parameters
        for rr in range(type_vec.size):
            for cc in range(slope_vec.size):
                index1 = rr * nC + cc + 1

                #create subplot for each combination of assymetry, # type, and tau
                ax1 = plt.subplot(nR, nC, index1)

                titleName = 'NType=%i, Sfis=%.0g' % (
                    type_vec[rr], slope_vec[cc])
            
                for mm in range(1): #mu_vec.size):
                    #plot all different values of mu in same subplot
                    #set parameters for current curve to extract
                    keyDict = {
                        'indv_NType': type_vec[rr],
                        'gr_Sfission': slope_vec[cc],
                        'indv_mutR': mu_vec[mm],
                    }
                    dataName = 'mu=%.0g' % mu_vec[mm]
                    #plot data
                    pltutl.plot_transect(
                        fig, ax1, statData, 'perimeter_loc', curPar, keyDict, dataName)
                    ax1.set_title(titleName)
                    ax1.legend()

        #clean up figure
        plt.tight_layout() 
        
        #save figure
        figureName = pathSave / (fileName + '_' + curPar + '.pdf')
        fig.savefig(figureName,
                    format="pdf", transparent=True)
    return None


#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)
