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

#set name of file to load (no extension)
fileName = 'scanFissionModes_cost5e-02_mu1e-02_tau100_interact0_dr1e-03_grK5e+03_sFis0e+00_sExt0e+00'

"""============================================================================
Set figure options 
============================================================================"""

font = {'family': 'Helvetica',
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


def make_fig(fileName):

    #set folders of data
    data_folder = Path(str(Path.home())+"/ownCloud/MLS_GroupDynamics_shared/Data/")
    fig_Folder = Path(str(Path.home()) +
                      "/ownCloud/MLS_GroupDynamics_shared/Figures/")
    figureName = fig_Folder / (fileName + '.pdf')
    
    #load data
    loadName = data_folder / (fileName + '.npz')
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
                        offspr_fracVec, statData, 'NTot_mav', 5E4)
    
    #plot Cooperator density
    ax = plt.subplot(nR, nC, 2)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'NCoop_mav', 5E4)
    
    #plot cooperator fraction
    ax = plt.subplot(nR, nC, 3)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'fCoop_mav', 1)
    
    #plot number of groups
    ax = plt.subplot(nR, nC, 4)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'NGroup_mav', 100)
    
    #plot mean group size
    ax = plt.subplot(nR, nC, 5)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'groupSizeAv_mav', 50)
    
    #plot median group size
    ax = plt.subplot(nR, nC, 6)
    pltutl.plot_heatmap(fig, ax, offspr_sizeVec,
                        offspr_fracVec, statData, 'groupSizeMed_mav', 50)
    
    #clean up figure
    plt.tight_layout() 
    
    #save figure
    fig.savefig(figureName,
                format="pdf", transparent=True)
    
    return None


#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)