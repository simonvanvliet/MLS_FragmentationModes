#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Last Update Oct 23 2019

Plots output of MlsGroupDynamics_scanStates


@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import sys
sys.path.insert(0, '..')

import matplotlib as mpl
from mainCode import MlsGroupDynamics_plotUtilities as pltutl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#set name of file to load (no extension)
fileName = 'March28_kInd1e+02_fisC0_kTot1e+04_nTyp1_asym1.npz'

#data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
pathLoad = Path(".")
pathSave = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/scanStates")

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

def make_fig(fileName, pathSave=pathSave, pathLoad=pathLoad):
    
    #remove extension if needed
    if fileName[-4:] == '.npz':
        fileName = fileName[:-4]
            
    #load data
    loadName   = pathLoad / (fileName + '.npz')
    data_file  = np.load(loadName, allow_pickle=True)
    results    = data_file['results']
    offsprSize = data_file['offsprSize'] 
    offsprFrac = data_file['offsprFrac']
    par1       = data_file['par1']
    par2       = data_file['par2']
    par3       = data_file['par3']
    parNames   = data_file['parNames']
    data_file.close()
    
    
    # process output
    statData, _, _ = zip(*results)
    statData = np.vstack(statData)
    
    
    """============================================================================
    Make plot
    ============================================================================"""
    
    
    
    parToPlot = ['NTot_mav', 'fCoop_mav', 'NGrp_mav','groupSizeAv_mav','GrpBirths_mav','GrpDeaths_mav','GrpNetProd_mav']
    roundTo = np.array([1000,1,1,1,1,1,1])

    for i in range(len(parToPlot)):
        fig = plt.figure()
        pltutl.set_fig_size_cm(fig, 45, 20)
    
        curPar = parToPlot[i]
    
        plotData = statData[curPar]
        
        plotSettings = {
        'vmin'    :   0,
        'cstep'   :   3,
        'dataName':   curPar,
        'xlabel'  :   'offspring size (frac.)',
        'ylabel'  :   'offspring frac. to parent',
        'roundTo' :   roundTo[i],
        'title'   :   '',
        'NRepeat' :   1
        } 
        
        #plot variables
        nC = par2.size * par3.size
        nCsub = par3.size
        nR = par1.size

        #loop over all variable parameters
        for rr in range(par1.size):
            for cc in range(par2.size):
                for ccsub in range(par3.size):
                    index1 = rr * nC + cc * nCsub + ccsub + 1
#                    index1 = rr + 1
                    ax1 = plt.subplot(nR, nC, index1)
                    
                    titleName = '%s=%.1g, %s=%.2g, %s=%.2g' % (
                        parNames[0], par1[rr],
                        parNames[1], par2[cc],
                        parNames[2], par3[ccsub])
                    
                    plotSettings['title'] = titleName
                    
                    keyDict = {
                            parNames[0]: par1[rr],
                            parNames[1]: par2[cc],
                            parNames[2]: par3[ccsub],
                        }
                    
                    pltutl.plot_mutational_meltdown(fig, ax1, 
                                                    offsprSize, offsprFrac, 
                                                    statData, plotData,
                                                    keyDict, plotSettings)
        #clean up figure
        plt.tight_layout() 
        #save figure
        figureName = pathSave / (fileName + '_' + curPar + '.pdf')
        fig.savefig(figureName,
                    format="pdf", 
                    transparent=True)
        
    return None

#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)