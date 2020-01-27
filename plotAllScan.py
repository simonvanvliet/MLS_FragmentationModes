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


scanFiles = glob.glob('scan2D*')


data_folder = Path(".")
fig_Folder = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Jan23_2020_part4")




for file in scanFiles:
    plot.make_fig(file, pathSave=fig_Folder, pathLoad=data_folder)