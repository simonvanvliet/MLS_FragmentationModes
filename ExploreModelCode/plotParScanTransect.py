#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Plots output of MlsGroupDynamics_scanTransects


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
fileName = 'transect_March1_cost0.01_migR0_kInd1e+02_kGrp0_kTot3e+04_asym1_dInd1_dGrp0_dTot1_dSiz0_fisC0.01.npz'

#set relative (rel) or absolute (abs)
plotStyle = 'rel' 
#set Folder
data_folder = Path(".")
fig_Folder = Path(
    "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Transect")

fileNameMod = '_' + plotStyle

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
    par0_vec = data_file['par0_vec']
    par1_vec = data_file['par1_vec']
    par2_vec = data_file['par2_vec']
    parNames = data_file['parNames']
    modelParList = data_file['modelParList']
    perimeter_loc_vec = data_file['perimeter_loc_vec']
    data_file.close()
    
    Output, _, _ = zip(*results)
    
    stateVar = ['NTot', 'NCoop', 'fCoop',
            'NGrp', 'groupSizeAv', 'groupSizeMed']
    
    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutR', 'indv_migrR', 'indv_asymmetry', 'delta_indv',
               'gr_SFis', 'gr_CFis', 'K_grp', 'K_tot', 'delta_grp', 'delta_tot', 'delta_size',
               'offspr_size', 'offspr_frac', 'run_time']

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVar]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVar]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3 + [('perimeter_loc', 'f8')]
    dType = np.dtype(dTypeList)
    statData = np.zeros(len(Output), dType)

    # store final state
    i = 0
    for data in Output:
        OffsprFrac = data['offspr_frac']
        OffsprSize = data['offspr_size']
        statData['perimeter_loc'][i] = OffsprSize if OffsprFrac >=0.5 else (1-OffsprSize)
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

    parToPlot = ['NTot_mav', 'fCoop_mav', 'NGrp_mav','groupSizeAv']

    for curPar in parToPlot:
    
        fig = plt.figure()
        pltutl.set_fig_size_cm(fig, 60, 40)
        
        #plot variables
        nC = par2_vec.size
        nR = par1_vec.size

        #loop over all variable parameters
        for rr in range(par1_vec.size):
            for cc in range(par2_vec.size):
                index1 = rr * nC + cc + 1

                #create subplot for each combination of assymetry, # type, and tau
                ax1 = plt.subplot(nR, nC, index1)

                titleName = '%s=%i, %s=%.g' % (
                    parNames[1], par1_vec[rr], 
                    parNames[2], par2_vec[cc])
            
                keyDictBaseLine = {
                        parNames[0]: par0_vec[0],
                        parNames[1]: par1_vec[rr],
                        parNames[2]: par2_vec[cc]
                    }
                        
                for mm in range(par0_vec.size):
                    #plot all different values of mu in same subplot
                    #set parameters for current curve to extract
                    keyDict = {
                        parNames[0]: par0_vec[mm],
                        parNames[1]: par1_vec[rr],
                        parNames[2]: par2_vec[cc]
                    }
                    dataName = '%s=%.0g' % (parNames[0], par0_vec[mm])
                    #plot data
                    if plotStyle == 'abs':
                        pltutl.plot_transect(
                            fig, ax1, statData, 'perimeter_loc', curPar, keyDict, dataName)
                    elif plotStyle == 'rel':
                        pltutl.plot_transect_relative(
                            fig, ax1, statData, 'perimeter_loc', curPar, keyDict, keyDictBaseLine, dataName)    
                        #ax1.set_ylim(0, 1)
                    ax1.set_title(titleName)
                    ax1.legend()

        #clean up figure
        plt.tight_layout() 
        
        #save figure
        figureName = pathSave / (fileName + '_' + curPar + fileNameMod + '.pdf')
        fig.savefig(figureName,
                    format="pdf", transparent=True)
    return None


#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)
