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
fileName = 'rawData_cost0.1_indvK1e+02_grK1e+02_sFis0_sExt0'

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
    results = data_file['results']
    offspr_sizeVec = data_file['offspr_sizeVec']
    type_vec = data_file['type_vec']
    mu_vec = data_file['mu_vec']
    assymetry_vec = data_file['assymetry_vec']
    tau_vec = data_file['tau_vec']
    data_file.close()
    
    
    Output, endDistFCoop, endDistGrSize = zip(*results)
    
    stateVar = ['NTot', 'NCoop', 'fCoop',
            'NGroup', 'groupSizeAv', 'groupSizeMed']
    
    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutationR', 'indv_asymmetry',
               'gr_Sfission', 'gr_Sextinct', 'gr_K', 'gr_tau',
               'offspr_size', 'offspr_frac']

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
    pltutl.set_fig_size_cm(fig, 30, 12)
    
    #plot variables
    nC = type_vec.size * assymetry_vec.size
    nR = tau_vec.size * 2

    #loop over all variable parameters
    for tt in range(type_vec.size):
        for aa in range(assymetry_vec.size):
            for cc in range(tau_vec.size):
                index1 = cc * nC + tt * assymetry_vec.size + aa + 1
                index2 = (cc+tau_vec.size) * nC + tt * assymetry_vec.size + aa + 1
    
                #create subplot for each combination of assymetry, # type, and tau
                ax1 = plt.subplot(nR, nC, index1)
                ax2 = plt.subplot(nR, nC, index2)

                titleName = 'NType=%i, Assymetry=%.0g, tau=%.0g' % (type_vec[tt], assymetry_vec[aa], tau_vec[cc])
            
                for mm in range(mu_vec.size):
                    #plot all different values of mu in same subplot
                    #set parameters for current curve to extract
                    keyDict = {
                        'indv_NType': type_vec[tt],
                        'indv_asymmetry': assymetry_vec[aa],
                        'gr_tau': tau_vec[cc],
                        'indv_mutationR': mu_vec[mm],
                    }
                    dataName = 'mu=%.0g' % mu_vec[mm]
                    #plot data
                    pltutl.plot_transect(
                        fig, ax1, statData, 'offspr_size', 'NTot_mav', keyDict, dataName)
                    ax1.set_title(titleName)
                    ax1.legend()

                    pltutl.plot_transect(
                        fig, ax2, statData, 'offspr_size', 'fCoop_mav', keyDict, dataName)

                    ax2.set_title(titleName)
                    ax2.legend()
    
    #clean up figure
    plt.tight_layout() 
    
    #save figure
    fig.savefig(figureName,
                format="pdf", transparent=True)
    
    return None


#run parscan and make figure
if __name__ == "__main__":
    make_fig(fileName)
