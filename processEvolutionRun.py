#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:54:31 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
import plotSingleRun_evolution as pltRun
import plotEvolutionMovie as evomo
from pathlib import Path
import matplotlib.pyplot as plt


fig_Folder = "/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/"
fig_FolderPath = Path(fig_Folder)


filename = 'evolution_Feb16_fisS2_cost0.01_muTy0.001_muSi0.1_muFr0.1_siIn0.05_frIn0.06_kInd2e+02_migR0.npz' 
movieName = filename[:-4]+ '.mp4'
figureName = filename[:-4]+ '.pdf'


data_file = np.load(filename, allow_pickle=True)
output = data_file['output']
traitDistr = data_file['traitDistr']
model_par = data_file['model_par']
data_file.close()

model_par = model_par[0]


movieDir = fig_FolderPath / movieName
figureDir = fig_FolderPath / figureName


fig = pltRun.plot_single_run(model_par, output, traitDistr)
fig.set_size_inches(4, 4)
plt.tight_layout()  # cleans up figure and aligns things nicely

#save figure
fig.savefig(figureDir,
            format="pdf", 
            transparent=True)


evomo = evomo.create_movie(traitDistr[0::10, :, :], movieDir, fps=25, size=800)