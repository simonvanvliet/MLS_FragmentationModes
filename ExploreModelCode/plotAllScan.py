#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:34:56 2020

Scans folder for data files of 2D scans and plots all results in separate pdf files

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import plotParScan as plot
import glob 
from pathlib import Path

#find data files
data_folder = Path(".")
search_names = data_folder / 'evol2D_*'
scanFiles = glob.glob(search_names)

#Set where figures are stores
fig_Folder = "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Evolution"
fig_FolderPath = Path(fig_Folder)

#loop files
for file in scanFiles:    
    figName = fig_Folder + '/' + file[:-4] + ".pdf"
    plot.make_fig(file, pathSave=fig_FolderPath, pathLoad=data_folder)