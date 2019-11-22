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

#set name of file to load (withou extension)
fileName = 'vanVliet_scanFrac_off_size0.1_cost0.05_indvK1e+02_grK1e+02_sFis0_sExt0_mk2'

#set Folder
data_folder = Path("./Data/")
fig_Folder = Path(str(Path.home()) +
                      "/ownCloud/MLS_GroupDynamics_shared/Figures/")

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

def make_fig(fileName):
    #set folders of data
    
    figureName = fig_Folder / (fileName + '.pdf')
    
    #load data
    loadName = data_folder / (fileName + '.npz')
    data_file = np.load(loadName, allow_pickle=True)
    results = data_file['results']
    gr_SfissionVec = data_file['gr_SfissionVec']
    gr_SextinctVec = data_file['gr_SextinctVec']
          
    data_file.close()
    
    
    Output, endDistFCoop, endDistGrSize = zip(*results)
    
    stateVar = ['NTot', 'NCoop', 'fCoop',
            'NGroup', 'groupSizeAv', 'groupSizeMed']
    
    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutationR', 'indv_asymmetry',
               'gr_Sfission', 'gr_Sextinct', 'gr_K', 'gr_tau',
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

    #things that can be plotted:
    #    'NA', 'NAprime', 'NB', 'NBprime',
    #    'NTot', 'NCoop', 'fCoop',
    #    'NGroup', 'groupSizeAv', 'groupSizeMed'
    #add _mav to get moving average value    
    
    fig = plt.figure()
    pltutl.set_fig_size_cm(fig, 60, 20)
    
    #plot variables
    gr_SfissionVec
    gr_SextinctVec
    
    
    nC = gr_SextinctVec.size
    nR = 2

    #loop over all variable parameters
    for cc in range(gr_SextinctVec.size):
        index1 = cc + 1
        index2 = nC + cc + 1

        #create subplot for each combination of assymetry, # type, and tau
        ax1 = plt.subplot(nR, nC, index1)
        ax2 = plt.subplot(nR, nC, index2)

        titleName = 'Slope Extinction=%.2g' % (gr_SextinctVec[cc])
    
        for mm in range(gr_SfissionVec.size):
            #plot all different values of mu in same subplot
            #set parameters for current curve to extract
            keyDict = {
                'gr_Sfission': gr_SfissionVec[mm],
                'gr_Sextinct': gr_SextinctVec[cc],
            }
            dataName = 'Sl. Fission=%.0g' % gr_SfissionVec[mm]
            #plot data
            pltutl.plot_transect(
                fig, ax1, statData, 'offspr_frac', 'NTot_mav', keyDict, dataName)
            ax1.set_title(titleName)
            ax1.legend()

            pltutl.plot_transect(
                fig, ax2, statData, 'offspr_frac', 'fCoop_mav', keyDict, dataName)

            ax2.set_title(titleName)
            ax2.legend()
            

#    #create subplot for each combination of assymetry, # type, and tau
#    ax1 = plt.subplot(1, 2, 1)
#    ax2 = plt.subplot(1, 2, 2)
#
#
#    #plot all different values of mu in same subplot
#    #set parameters for current curve to extract
#    keyDict = {}
#    #plot data
#    pltutl.plot_transect(
#        fig, ax1, statData, 'offspr_frac', 'NTot_mav', keyDict, "")
#    ax1.legend()
#
#    pltutl.plot_transect(
#        fig, ax2, statData, 'offspr_frac', 'fCoop_mav', keyDict, "")
#    ax2.legend()
    
    #clean up figure
    plt.tight_layout() 
    
    #save figure
    fig.savefig(figureName,
                format="pdf", transparent=True)
    
    return None


#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)
