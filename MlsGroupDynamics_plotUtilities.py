#!/usr/bin/env python3
# -*- CoDing: utf-8 -*-
"""
CreateD on Oct 23 2019
Last Update Oct 23 2019

@author: simonvanvliet
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca

Contains varies functions used in MLS model code and in figure code

"""
import numpy as np
import math

# %% Set figure size in cm
def set_fig_size_cm(fig, w, h):
    cmToInch = 0.393701
    wInch = w * cmToInch
    hInch = h * cmToInch
    fig.set_size_inches(wInch, hInch)
    return None


#convert list of results to 2D matrix of offspring frac. size vs fraction of parent to offspring
def create_2d_matrix(offspr_sizeVec, offspr_fracVec, statData, fieldName):
    #get size of matrix
    numX = offspr_sizeVec.size
    numY = offspr_fracVec.size
    #init matrix to NaN
    dataMatrix = np.full((numY, numX), np.nan)

    #fill matrix
    for xx in range(numX):
        for yy in range(numY):
            #find items in list that have correct fissioning parameters for current location in matrix
            currXId = statData['offspr_size'] == offspr_sizeVec[xx]
            currYId = statData['offspr_frac'] == offspr_fracVec[yy]
            currId = np.logical_and(currXId, currYId)
            #extract output value and assign to matrix
            if currId.sum() == 1:
                dataMatrix[yy, xx] = np.asscalar(statData[fieldName][currId])
    return dataMatrix


#make heatmap of 2D matrix
def plot_heatmap(fig, ax, offspr_sizeVec, offspr_fracVec, statData, dataName, rounTo):
    #convert 1D list to 2D matrix
    data2D = create_2d_matrix(
        offspr_sizeVec, offspr_fracVec, statData, dataName)
    
    #find max value 
    maxData = math.ceil(np.nanmax(data2D) / rounTo) * rounTo
    
    #plot heatmap
    im = ax.pcolormesh(offspr_sizeVec, offspr_fracVec, data2D,
                       cmap='plasma', vmin=0, vmax=maxData)
    #add colorbar
    fig.colorbar(im, ax=ax, orientation='horizontal',
                 label=dataName,
                 ticks=[0, maxData/2, maxData], 
                 fraction=0.5, pad=0.1)

    #make axis nice
    xRange = (offspr_sizeVec.min(), offspr_sizeVec.max())
    yRange = (offspr_fracVec.min(), offspr_fracVec.max())
    steps = (3, 3)
    ax.set_xlim(xRange)
    ax.set_ylim(yRange)
    ax.set_xticks(np.linspace(*xRange, steps[0]))
    ax.set_yticks(np.linspace(*yRange, steps[1]))

    # set labels
    ax.set_xlabel('offspring frac. size')
    ax.set_ylabel('frac. parrent to offspring')

    return None