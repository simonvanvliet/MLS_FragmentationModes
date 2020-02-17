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

filename = 'evolution_200215_1756.npz' 
movieName = 'evolution_200215_1756.mp4' 


data_file = np.load(filename, allow_pickle=True)
output = data_file['output']
traitDistr = data_file['traitDistr']
model_par = data_file['model_par']
data_file.close()

model_par = model_par[0]


pltRun.plot_single_run(model_par, output, traitDistr)

#evomo = evomo.create_movie(traitDistr, movieName, fps=25, size=800)