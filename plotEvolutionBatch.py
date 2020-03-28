#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:54:31 2020

Plots a batch of evolution runs created with MlsEvoBatch

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
#import plotEvolutionMovie as evomo
import MlsGroupDynamics_plotUtilities as pltutl
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import glob

# set file and folder names
fig_Folder = "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/GroupEvolution"
fig_FolderPath = Path(fig_Folder)
baseName2D = 'group_evolution2D_March6'
baseNameEv = 'group_evolution_March6'
saveNameMod = ''

# set variables to scan
gr_SFis_vec = np.array([0.2,4,8])
par0Name = 'indv_tau'
par0NameAbrv = 'tInd'
par0Vec = np.array([1])

indv_K_vec = np.array([50, 200])


# gr_Sfission_Vec = np.array([0, 0.1, 2, 8])
# indv_KVec = np.array([50, 200])

# setup figure formatting
font = {'family': 'arial',
        'weight': 'normal',
        'size': 6}
matplotlib.rc('font', **font)

def plot_heatmap(fig, ax, offsprSize, offsprFrac, statData, dataName, roundTo):
    
    offsprFrac = np.sort(np.unique(statData['offspr_frac']))
    offsprSize = np.sort(np.unique(statData['offspr_size']))
    
    data2D = pltutl.create_2d_matrix(
        offsprSize, offsprFrac, statData, dataName)

    plotSettings = {
        'roundTo':   roundTo,
        'dataName':   dataName,
        'vmin':   0,
        'xlabel':   'offspring size (frac.)',
        'ylabel':   'offspring frac. to parent',
        'cmap':   'cividis',
        'cmap_bad':   'white',
        'xmin':    0.,
        'ymin':    0.,
        'xmax':    0.5,
        'ymax':    1,
        'alpha':    0.7
    }

    pltutl.plot_heatmap_sub(fig, ax,
                            offsprSize,
                            offsprFrac,
                            data2D, plotSettings)
    return None


def add_evo_traject(ax, outputEvo):
    timeColor = cm.cool(np.linspace(0, 1, outputEvo.size))
    ax.scatter(outputEvo['offspr_size_mav'],
               outputEvo['offspr_frac_mav'],
               s=3,
               edgecolor='none',
               color=timeColor)
    return None

def plot_evo_heatmap_time(axs, traitDistr):
    numT = traitDistr.shape[0]
    timeColor = cm.cool(np.linspace(0, 1, numT))
    alpha = np.linspace(0.4, 0.8, numT)
    
    xVec = np.linspace(0,0.5,traitDistr.shape[2])
    yVec = np.linspace(0,1,traitDistr.shape[1])

    for tt in range(numT):
        currData = traitDistr[tt, :, :]
        loc = np.nonzero(currData)
        xLoc = xVec[loc[1]]
        yLoc = yVec[loc[0]]
        size = currData[loc]
        
        axs.scatter(xLoc,
               yLoc,
               s=50*size,
               edgecolor='none',
               alpha = alpha[tt],
               color=timeColor[tt])
    axs.set_xlim(0,0.5)
    axs.set_ylim(0,1)
        
    return None


def plot_evo_heatmap(ax, traitDistr):
    currData = traitDistr[-1, :, :]
    image = ax.imshow(currData, cmap='hot',
                interpolation='nearest',
                extent=[0, 1, 0, 1],
                origin='lower',
                aspect='auto')
    return None


# set search for name
for gr_SFis in gr_SFis_vec:
    searchName2D = baseName2D + '*fisS%.0g_*' % (gr_SFis) 
    searchNameEv = baseNameEv + '*fisS%.0g_*' % (gr_SFis)
    
    # find 2D scans
    files2DAll = glob.glob(searchName2D)
    filesEvAll = glob.glob(searchNameEv)
        
    for par0 in par0Vec:
        subs = par0NameAbrv + '%.0g' % par0
        files2D = [i for i in files2DAll if subs in i] 
        filesEv = [i for i in filesEvAll if subs in i] 
        
        if len(files2D) == 1 and len(filesEv) > 0:

            fileName2D = files2D[0]
            figureName = fileName2D[:-4] + saveNameMod + '.pdf'
            figureDir = fig_FolderPath / figureName

            # Load 2D scan data
            data_file = np.load(fileName2D, allow_pickle=True)
            statData = data_file['statData']
            offsprFrac = data_file['offspr_sizeVec']
            offsprSize = data_file['offspr_fracVec']
            data_file.close()
            
            

            # open figure
            fig, axs = plt.subplots(3, 5)

            # plotheatmaps
            plot_heatmap(fig, axs[1,0], offsprSize, offsprFrac, 
                        statData, 'NGrp_mav', 5)
            plot_heatmap(fig, axs[1,1], offsprSize, offsprFrac,
                        statData, 'NTot_mav', 100)
            plot_heatmap(fig, axs[1,2], offsprSize, offsprFrac,
                        statData, 'NCoop_mav', 100)
            plot_heatmap(fig, axs[1,3], offsprSize, offsprFrac,
                        statData, 'groupSizeAv_mav', 5)
            plot_heatmap(fig, axs[1,4], offsprSize, offsprFrac,
                        statData, 'groupSizeMed_mav', 5)

            idx = 0 
            # plot evolution trajectories
            for fileNameEv in filesEv:
                # Load evolution data
                data_file = np.load(fileNameEv, allow_pickle=True)
                outputEvo = data_file['output']
                traitDistr = data_file['traitDistr']
                data_file.close()
                
#                traitDistr = traitDistr[::20,:,:]
#                outputEvo = outputEvo[::20]

                pltutl.plot_time_data(axs[0,0], outputEvo, "NGroup_mav")
                axs[0,0].set_ylabel("# group")
                pltutl.plot_time_data(axs[0,1], outputEvo, "NTot_mav")
                axs[0,1].set_ylabel("# cell tot")
                pltutl.plot_time_data(axs[0,2], outputEvo, "NCoop_mav")
                axs[0,2].set_ylabel("# cell coop")
                pltutl.plot_time_data(axs[0,3], outputEvo, "offspr_size_mav")
                axs[0,3].set_ylabel("offspring size frac.")
                axs[0,3].set_ylim(0, 0.5)
                pltutl.plot_time_data(axs[0,4], outputEvo, "offspr_frac_mav")
                axs[0,4].set_ylabel("offspring frac")
                axs[0,4].set_ylim(0, 1)

                # plot number of groups
                add_evo_traject(axs[1,0], outputEvo)
                add_evo_traject(axs[1,1], outputEvo)
                add_evo_traject(axs[1,2], outputEvo)
                add_evo_traject(axs[1,3], outputEvo)
                add_evo_traject(axs[1,4], outputEvo)
                
                plot_evo_heatmap_time(axs[2,idx], traitDistr)
                idx += 1

            # set figure size
            pltutl.set_fig_size_cm(fig, 30, 22)
            plt.tight_layout()  # cleans up figure and aligns things nicely
            
            # save figure
            fig.savefig(figureDir,
                        format="pdf",
                        transparent=True)


#evomo = evomo.create_movie(traitDistr[0::10, :, :], movieDir, fps=25, size=800)

