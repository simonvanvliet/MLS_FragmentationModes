#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:34:56 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import plotParScan as plot
import glob 
from pathlib import Path


scanFiles = glob.glob('scan2D_Feb15*')


data_folder = Path(".")

fig_Folder = "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Feb15"
fig_FolderPath = Path(fig_Folder)


scanFigures = glob.glob(fig_Folder + "/scan2D*")
recreate = True


for file in scanFiles:    
    figName = fig_Folder + '/' + file[:-4] + ".pdf"
    #if (not figName in scanFigures) | recreate:
    plot.make_fig(file, pathSave=fig_FolderPath, pathLoad=data_folder)