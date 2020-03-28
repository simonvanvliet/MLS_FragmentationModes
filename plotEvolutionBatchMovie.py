#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:54:31 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
import makeEvolutionMovie as evomo
import MlsGroupDynamics_plotUtilities as pltutl
from pathlib import Path
import glob

# set file and folder names
fig_Folder = "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Evolution"
fig_FolderPath = Path(fig_Folder)
baseName2D = 'evol2D_Feb28'
baseNameEv = 'evolution_Feb28'

# set variables to scan
gr_SFis_vec = np.array([0, 0.1, 2, 8])
indv_K_vec = np.array([50, 200])


# set search for name
for gr_SFis in gr_SFis_vec:
    for indv_K in indv_K_vec:
        searchName2D = baseName2D + '*kInd%.0g*fisS%.0g*.npz' % (indv_K, gr_SFis) 
        searchNameEv = baseNameEv + '*fisS%.0g*kInd%.0g*.npz' % (gr_SFis, indv_K)

        # find 2D scans
        files2D = glob.glob(searchName2D)
        filesEv = glob.glob(searchNameEv)

        if len(files2D) == 1 and len(filesEv) > 0:

            fileName2D = files2D[0]
            figureName = fileName2D[:-4] + '.pdf'
            figureDir = fig_FolderPath / figureName

            # Load 2D scan data
            data_file = np.load(fileName2D, allow_pickle=True)
            statData = data_file['statData']
            offsprFrac = data_file['offspr_sizeVec']
            offsprSize = data_file['offspr_fracVec']
            data_file.close()
            
            
            offsprFrac = np.sort(np.unique(statData['offspr_frac']))
            offsprSize = np.sort(np.unique(statData['offspr_size']))
            
            data2D = pltutl.create_2d_matrix(
                offsprSize, offsprFrac, statData, 'NTot_mav')
        
            
            # plot evolution trajectories
            for fileNameEv in filesEv:
                # Load evolution data
                data_file = np.load(fileNameEv, allow_pickle=True)
                traitDistr = data_file['traitDistr']
                data_file.close()
                
                traitDistr = traitDistr[:,:,:]
                
                figureName = fileNameEv[:-4] + '.mp4'
                movieDir = fig_FolderPath / figureName
            
                evomo.create_movie(traitDistr[0::10, :, :], movieDir, 
                                   data_bg=data2D, fps=25, size=800)

