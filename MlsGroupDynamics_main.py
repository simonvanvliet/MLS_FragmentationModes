#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:29:20 2019

Last Update Oct 16 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import numpy as np
import MlsGroupDynamics_utilities as util 
import math
import matplotlib.pyplot as plt
import matplotlib
import time
from numba import jit, void, f8, i8
#from numba.types import UniTuple, Tuple




"""============================================================================
Init functions 
============================================================================"""

# initialize output matrix
def init_output_matrix(Num_t_sample):
    # specify output fields
    dType = np.dtype([('NGroup', 'f8'),
                      ('NA', 'f8'),
                      ('NAprime', 'f8'),
                      ('NB', 'f8'),
                      ('NBprime', 'f8'),
                      ('time', 'f8')])

    #initialize outputs to NaN
    Output = np.full(Num_t_sample, np.nan, dType)
    Output['time'][0] = 0
    return Output


# initialize group matrix
# each column is a group and lists number of [A,A',B,B'] cells
def init_groupMat(model_par):
    numGroup = int(model_par["init_groupNum"])
    
    #convert list of initComp to numpy column vector
    init_groupComp = np.c_[model_par["init_groupComp"]]
    
    #multiply with density and round to get cell numbers
    init_groupComp = np.round(init_groupComp * model_par["init_groupDens"])
    
    #init all groups with same composition
    groupMat = init_groupComp @ np.ones((1, numGroup))
    
    # store in C-byte order
    groupMat = np.copy(groupMat, order='C')
    
    return groupMat


"""============================================================================
Sample model code 
============================================================================"""

# sample model
def sample_model(groupMatrix, output, sample_idx, currT):
    # store time
    output['time'][sample_idx] = currT
    
    # calc number of groups
    NGroup = groupMatrix.shape[1]
    output['NGroup'][sample_idx] = NGroup
    
    # calc total population sizes
    NTotalCell = groupMatrix.sum(1)
    output['NA'][sample_idx]        = NTotalCell[0]
    output['NAprime'][sample_idx]   = NTotalCell[1]
    output['NB'][sample_idx]        = NTotalCell[2]
    output['NBprime'][sample_idx]   = NTotalCell[3]

    sample_idx += 1
    return sample_idx


"""============================================================================
Sub functions of model code
============================================================================"""

#calculate birth and death rate for all groups and types
#@jit provides speedup by compling this function at start of execution
#To use @jit provide the data type of output and input, nopython=true makes compilation faster
@jit(f8[::1](f8[:, :], f8, f8, f8[::1]), nopython=True)
def calc_indv_rates(groupMat, cost, deathR, oneVec4):
    #calc total number of individuals per group, use matrix product for speed
    Ntot = oneVec4 @ groupMat
    
    #calc birth rates per type and group
    birthA  = (1-cost) * groupMat[2,:] * groupMat[0,:] / Ntot
    birthAp =            groupMat[2,:] * groupMat[1,:] / Ntot
    birthB  = (1-cost) * groupMat[0,:] * groupMat[2,:] / Ntot
    birthBp =            groupMat[0,:] * groupMat[3,:] / Ntot
    
    #calc death rates per type and group
    deathA  = deathR * groupMat[0,:] * Ntot
    deathAp = deathR * groupMat[1,:] * Ntot
    deathB  = deathR * groupMat[2,:] * Ntot
    deathBp = deathR * groupMat[3,:] * Ntot
    
    #combine all rates in single vector
    rates = np.concatenate((birthA, birthAp, birthB, birthBp,
                            deathA, deathAp, deathB,deathBp))
    
    return rates 


#process individual level events
@jit(i8(f8[:, :], f8[::1], f8, f8[::1]), nopython=True)
def process_indv_event(groupMat, rateVector, mutationR, rand):
    #Note: groupMat is updated in place, it does not need to be returned
    #get number of groups
    numGroup = groupMat.shape[1]
    
    #select random event based on propensity
    eventID = util.select_random_event(rateVector, rand[0])
    
    #get event type
    eventType = math.floor(eventID/numGroup)
    #get event group
    eventGroup = eventID % numGroup # % is modulo operator
    
    #track if any groups die in process
    groupDeathID = -1 #-1 is no death 

    if eventType < 4: #birth event 
        #add cell to group, check for mutations first
        cellType = eventType
        if (cellType % 2)==0: #Wild type cell, can mutate
            if rand[1] < mutationR: #birth with mutation
                groupMat[cellType+1, eventGroup] += 1
            else: #birth without mutation
                groupMat[cellType, eventGroup] += 1
        else: #cheater cell, cannot mutate 
            groupMat[cellType, eventGroup] += 1
    else: #death event
        #remove cell from group
        cellType = eventType - 4
        groupMat[cellType, eventGroup] -= 1
        
        #kill group if last cell died
        #use two stage check for increased speed
        if groupMat[cellType, eventGroup] == 0: #killed last of type
            #NInGroup = oneVec4 @ groupMat[:, eventGroup]
            NInGroup = groupMat[:, eventGroup].sum()
            if NInGroup == 0: #all other types are zero too
                groupDeathID = int(eventGroup)
            
    return groupDeathID


#remove empty group from group matrix
def remove_empty_group(groupMat, groupDeathID):
    #Note: groupMat is re-created, it has to be returned
    #create helper vector
    numGroup = groupMat.shape[1]
    hasDied = np.zeros(numGroup)
    hasDied[groupDeathID] = 1
    
    #copy remaining groups to new matrix
    groupMat = groupMat[:,hasDied==0]
    
    return groupMat


"""============================================================================
Main model code
============================================================================"""

#main model code
def run_model(model_par):
    #helper vector to calc sum over all cell types
    oneVec4 = np.array([1.,1.,1.,1.])

    ## Initialize model, get rates and init matrices
    #number of steps to run
    sampleInt = model_par['sampleInt']
    numTStep = int(model_par['numTimeStep'])
    numTSample = int(np.ceil(numTStep / sampleInt)+1)
    
    #initialize group matrix
    groupMat = init_groupMat(model_par)
    
    #get matrix with random numbers
    randMat = util.create_randMat(numTStep, 2)
    
    #initialize output matrix
    output = init_output_matrix(numTSample)
    
    # get individual rates
    indv_deathR, indv_cost, indv_mutationR = [float(model_par[x])
                            for x in ('indv_deathR', 'indv_cost', 'indv_mutationR')]
    
    # first sample
    currT = 0
    sampleIdx = 0
    sampleIdx = sample_model(groupMat, output, sampleIdx, currT)
    
    ## loop time steps
    for tt in range(numTStep):
        #calc rates of individual level events
        indvRates = calc_indv_rates(groupMat, indv_cost, indv_deathR, oneVec4)
        
        #calc time step
        dt = 1
        
        #select and process individual level event
        groupDeathID = process_indv_event(groupMat, indvRates, indv_mutationR, randMat[tt,:])
        if groupDeathID > -1: #group has died, remove it
            groupMat = remove_empty_group(groupMat, groupDeathID)
            numGroup = groupMat.shape[1]
            if numGroup==0: #all groups have died, end simulation
                break
            
        # update time
        currT += dt
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = sample_model(groupMat, output, sampleIdx, currT)         

    print(groupMat)
    
    return output 


"""============================================================================
Code that calls model and plots results
============================================================================"""

#code to plot data
# set type to "lin" or "log" to swicth between lin or log plot
def plot_data(dataStruc, FieldName, type='lin'):
    #linear plot
    if type == 'lin':
        plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    #log plot    
    elif type == 'log':
        plt.semilogy(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    
    #set x-label
    plt.xlabel("time")
    maxTData = np.nanmax(dataStruc['time'])
    plt.xlim((0, maxTData))
   
        
    return None


# run model, plot dynamics
def single_run_with_plot(model_par):
    # run code
    start = time.time()
    output = run_model(model_par)
    end = time.time()
    
    #print timing
    print("Elapsed time run 1 = %s" % (end - start))

    #setp figure formatting
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 6}
    matplotlib.rc('font', **font)

    #open figure
    fig = plt.figure()
    nR = 1
    nC = 2

    # plot number of groups
    plt.subplot(nR, nC, 1)
    plot_data(output, "NGroup")
    plt.ylabel("# group")
    plt.legend()

    # plot number of cells
    plt.subplot(nR, nC, 2)
    plot_data(output, "NA")
    plot_data(output, "NAprime")
    plot_data(output, "NB")
    plot_data(output, "NBprime")
    plt.ylabel("# cell")
    plt.legend()

    #set figure size
    fig.set_size_inches(4, 2)
    plt.tight_layout() #cleans up figure and alligns things nicely

    return None


# run model with default parameters
def run_w_def_parameter():
    model_par = {
        # solver settings
        "numTimeStep":      1E6,
        "sampleInt":        10,
        # settings for intial condition
        "init_groupNum":    10,
        "init_groupComp":   [0.5, 0, 0.5, 0],
        "init_groupDens":   50,
        # settings for individual level dynamics
        "indv_cost":        0.1,
        "indv_deathR":      0.001,
        "indv_mutationR":   1E-1
    }

    single_run_with_plot(model_par)

    return None


##this piece of code is run only when this script is executed as the main 
if __name__ == "__main__":
    print("running with default parameter")
    run_w_def_parameter()
