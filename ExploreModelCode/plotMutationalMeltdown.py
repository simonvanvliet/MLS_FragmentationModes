#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Plots output of MlsGroupDynamics_scanMutationalMeltdown

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

import sys
sys.path.insert(0, '..')

import matplotlib as mpl
from mainCode import MlsGroupDynamics_plotUtilities as pltutl
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

#set name of file to load (no extension)
fileName = 'MutationMeltdown_March28_kInd1e+02_fisC0.01_kTot3e+04_asym1.npz'

#data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
data_folder = Path(".")
fig_folder = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/MutationalMeltdown/")

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

def make_fig(fileName, pathSave=fig_folder, pathLoad=data_folder):
    
    #remove extension if needed
    if fileName[-4:] == '.npz':
        fileName = fileName[:-4]

    #set folders of data
    figureName = pathSave / (fileName + '.pdf')
    
    #load data
    loadName   = pathLoad / (fileName + '.npz')
    data_file   = np.load(loadName, allow_pickle=True)
    statData   = data_file['statData']
    maxMu      = data_file['maxMu']
    NTot       = data_file['NTot']
    NCoop      = data_file['NCoop']
    NGrp       = data_file['NGrp']
    numRepeat  = data_file['numRepeat']
    offsprSize = data_file['offsprSize'] 
    offsprFrac = data_file['offsprFrac']
    mutR       = data_file['mutR']
    mode_vec   = data_file['mode_vec']
    par0_vec   = data_file['par0_vec']
    par1_vec   = data_file['par1_vec']
    mode_set   = data_file['mode_set']
    modeNames  = data_file['modeNames']
    parNames   = data_file['parNames']
    data_file.close()
    
    
    
    """============================================================================
    Make plot
    ============================================================================"""
    
    fig = plt.figure()
    pltutl.set_fig_size_cm(fig, 60, 30)
    
    plotData = np.log10(np.reshape(np.nanmean(maxMu,axis=1),(-1,1)))
    
    plotSettings = {
      'vmin'    :   np.min(np.log10(mutR)),
      'vmax'    :   np.max(np.log10(mutR)),
      'cstep'   :   6,
      'dataName':   'max mutation rate',
      'xlabel'  :   'offspring size (frac.)',
      'ylabel'  :   'offspring frac. to parent',
      'NRepeat' :   maxMu.shape[1]
    }
    
   #plot variables
    nC = mode_vec.size
    nRsub = par1_vec.size
    nR = par0_vec.size * par1_vec.size

    #loop over all variable parameters
    for rr in range(par0_vec.size):
        for rrsub in range(par1_vec.size):
            for cc in range(mode_vec.size):
                index1 = (rr * nRsub + rrsub) * nC + cc + 1
                ax1 = plt.subplot(nR, nC, index1)
                
                titleName = '%s=%.2g, %s=%.2g, %s=%.2g, %s=%.2g' % (
                    modeNames[0], mode_set[0, cc], 
                    modeNames[1], mode_set[1, cc],
                    parNames[0],  par0_vec[rr],
                    parNames[1],  par1_vec[rrsub])
                
                keyDict = { modeNames[0]: mode_set[0, cc], 
                            modeNames[1]: mode_set[1, cc],
                            parNames[0]:  par0_vec[rr],
                            parNames[1]:  par1_vec[rrsub]}
                
                pltutl.plot_mutational_meltdown(fig, ax1, 
                                                offsprSize, offsprFrac, 
                                                statData, plotData, 
                                                keyDict, plotSettings)
                ax1.set_title(titleName)

    #clean up figure
    plt.tight_layout() 
    
    #save figure
    fig.savefig(figureName,
                format="pdf", 
                transparent=True)
    
    return None

#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)