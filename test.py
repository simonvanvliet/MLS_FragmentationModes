#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:54:31 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


nBinOffsprSize = 100
nBinOffsprFrac = 100    

binsOffsprSize = np.linspace(0, 0.5, nBinOffsprSize+1)
binsOffsprFrac = np.linspace(0, 1, nBinOffsprFrac+1)

binCenterOffsprSize = (binsOffsprSize[1::]+binsOffsprSize[0:-1])/2
binCenterOffsprFrac = (binsOffsprFrac[1::]+binsOffsprFrac[0:-1])/2


def process_frame(data):
    processedData = np.copy(data)
    for ff in range(nBinOffsprFrac):
        for ss in range(nBinOffsprSize):
            toLow = binsOffsprFrac[ff] < binsOffsprSize[ss]
            toHigh = binsOffsprFrac[ff] > (1-binsOffsprSize[ss+1])
            isEmpty = data[ff,ss] == 0
            if (toLow or toHigh) and isEmpty:
                processedData[ff, ss] = np.nan
    return processedData


filename = 'evolution_Feb16_fisS2_cost0.01_muTy0.001_muSi0.1_muFr0.1_siIn0.5_frIn0.5_kInd2e+02_migR0.npz' 
movieName = filename[:-4]+ '.mp4'
figureName = filename[:-4]+ '.pdf'





data_file = np.load(filename, allow_pickle=True)
output = data_file['output']
traitDistr = data_file['traitDistr']
model_par = data_file['model_par']
data_file.close()


data = traitDistr[1, :, :]
data = process_frame(data)

maxValue = np.nanmax(data)


cmap = matplotlib.cm.get_cmap(name='viridis')
cmap.set_bad(color='black')

fig = plt.figure()
axs = plt.subplot(1, 1, 1)
im = axs.imshow(data, cmap=cmap,
                interpolation='nearest',
                extent=[0, 0.5, 0, 1],
                origin='lower',
                vmin = 0,
                vmax = maxValue,
                aspect='auto')

fig.subplots_adjust(0, 0, 1, 1)
axs.axis("off")
fig.set_size_inches(1, 1)
plt.tight_layout()  # cleans up figure and aligns things nicely


#init matrix to keep mutations inbounds
offsprFracMatrix = np.zeros((nBinOffsprFrac, nBinOffsprSize),dtype=int)
for ff in range(nBinOffsprFrac):
    for ss in range(nBinOffsprSize):
        offsprFracUp = binsOffsprFrac[1:]
        offsprFracLow = binsOffsprFrac[:-1]
        
        toLow = offsprFracUp[ff] < binsOffsprSize[ss]
        toHigh = offsprFracLow[ff] > (1-binsOffsprSize[ss])
        #decrease or increase offsprFracIdx till within bounds
        if toHigh:
            idx = np.arange(nBinOffsprFrac)
            withinBounds = offsprFracLow < (1 - binsOffsprSize[ss])
            offsprFracIdx = int(np.max(idx[withinBounds]))
        elif toLow:
            idx = np.arange(nBinOffsprFrac)
            withinBounds = offsprFracUp > binsOffsprSize[ss]
            offsprFracIdx = int(np.min(idx[withinBounds]))
        else:
            offsprFracIdx = ff
        offsprFracMatrix[ff, ss] = int(offsprFracIdx)
        
        
fig = plt.figure()
axs = plt.subplot(1, 1, 1)
im = axs.imshow(offsprFracMatrix, cmap=cmap,
                interpolation='nearest',
                extent=[0, 0.5, 0, 1],
                origin='lower',
                aspect='auto')

fig.subplots_adjust(0, 0, 1, 1)
axs.axis("off")
fig.set_size_inches(1, 1)
plt.tight_layout()  # cleans up figure and aligns things nicely        
        