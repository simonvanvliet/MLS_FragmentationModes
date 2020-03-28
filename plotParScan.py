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

#SET name of file to load 
fileName = 'scan2D_Jan20_NSpecies2_Assym1_cost0.01_mu0.01_tau100_indvK5e+01_grK1e+02_sFis1_sExt0_Nmin0_offset1_deltaind1_deltagr0.npz'

#data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
data_folder = Path(".")
fig_Folder = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Jan22_2020")



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
    
    #remove extension if needed
    if fileName[-4:] == '.npz':
        fileName = fileName[:-4]

    #set folders of data
    figureName = pathSave / (fileName + '.pdf')
    
    #load data
    loadName = pathLoad / (fileName + '.npz')
    data_file = np.load(loadName, allow_pickle=True)
    statData = data_file['statData']
    offspr_fracVec = data_file['offspr_fracVec']
    offspr_sizeVec = data_file['offspr_sizeVec']
    data_file.close()
    
    """============================================================================
    Make plot
    ============================================================================"""
    
    
    #things that can be plotted:
    #    'NA', 'NAprime', 'NB', 'NBprime',
    #    'NTot', 'NCoop', 'fCoop',
    #    'NGroup', 'groupSizeAv', 'groupSizeMed'
    #add _mav to gte moving average value    
    
    fig = plt.figure()
    pltutl.set_fig_size_cm(fig, 15, 12)
    
    #plot variables
    nR = 2
    nC = 3
    
    #plot total cell density
    ax = plt.subplot(nR, nC, 1)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'NTot_mav', 100)
    
    #plot Cooperator density
    ax = plt.subplot(nR, nC, 2)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'NCoop_mav', 100)
    
    #plot cooperator fraction
    ax = plt.subplot(nR, nC, 3)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'fCoop_mav', 1)
    
    #plot number of groups
    ax = plt.subplot(nR, nC, 4)
    try:
        pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                            offspr_fracVec, statData, 'NGroup_mav', 5)
    except:
        pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                            offspr_fracVec, statData, 'NGrp_mav', 5)
    
    #plot mean group size
    ax = plt.subplot(nR, nC, 5)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'groupSizeAv_mav', 5)
    
    #plot median group size
    ax = plt.subplot(nR, nC, 6)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'groupSizeMed_mav', 5)
    
    #clean up figure
    plt.tight_layout() 
    
    #save figure
    fig.savefig(figureName,
                format="pdf", transparent=True)
    
    return None


#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)