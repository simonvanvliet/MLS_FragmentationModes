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
from numba.types import UniTuple, Tuple




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
Sub functions individual dynamics 
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
    eventtRype = math.floor(eventID/numGroup)
    #get event group
    eventGroup = eventID % numGroup # % is modulo operator
    
    #track if any groups die in process
    groupDeathID = -1 #-1 is no death 

    if eventtRype < 4: #birth event 
        #add cell to group, check for mutations first
        cellType = eventtRype
        if (cellType % 2)==0: #Wild type cell, can mutate
            if rand[1] < mutationR: #birth with mutation
                groupMat[cellType+1, eventGroup] += 1
            else: #birth without mutation
                groupMat[cellType, eventGroup] += 1
        else: #cheater cell, cannot mutate 
            groupMat[cellType, eventGroup] += 1
    else: #death event
        #remove cell from group
        cellType = eventtRype - 4
        groupMat[cellType, eventGroup] -= 1
        
        #kill group if last cell died
        #use two stage check for increased speed
        if groupMat[cellType, eventGroup] == 0: #killed last of type
            #NInGroup = oneVec4 @ groupMat[:, eventGroup]
            NInGroup = groupMat[:, eventGroup].sum()
            if NInGroup == 0: #all other types are zero too
                groupDeathID = int(eventGroup)
            
    return groupDeathID


"""============================================================================
Sub functions group dynamics 
============================================================================"""

#remove group from group matrix
@jit(Tuple((f8[:, :], i8))(f8[:, :], i8), nopython=True)
def remove_group(groupMat, groupDeathID):
    #Note: groupMat is re-created, it has to be returned
    #create helper vector
    numGroup = groupMat.shape[1]
    hasDied = np.zeros(numGroup)
    hasDied[groupDeathID] = 1
    
    #copy remaining groups to new matrix
    groupMat = groupMat[:,hasDied==0]
    
    numGroup -= 1 
    
    return (groupMat, numGroup)


#calculate fission and extinction ratea  of all groups
@jit(f8[::1](f8[:, :], f8, f8, f8, f8[::1], f8[::1]), nopython=True)
def calc_group_rates(groupMat, group_Sfission, group_Sextinct, group_K, oneVec4, oneVecGroup):
    #calc total number of individuals per group, use matrix product for speed
    Ntot_group = oneVec4 @ groupMat
    
    #calc total number of individuals
    Ntot = oneVecGroup @ Ntot_group
    
    #calc fission rate
    if group_Sfission>0:
        fissionR = oneVecGroup + group_Sfission * Ntot_group
    else:
        fissionR = oneVecGroup
    
    #calc extinction rate
    groupDeathRate = Ntot / group_K
    if group_Sextinct>0:
        extinctR = (oneVecGroup + group_Sextinct * Ntot_group) * groupDeathRate 
    else:
        extinctR = oneVecGroup * groupDeathRate 
    
    #combine all rates in single vector
    rates = np.concatenate((fissionR, extinctR))

    return rates 


#process individual level events
@jit(Tuple((f8[:, :], i8))(f8[:, :], f8[::1], f8[::1]), nopython=True)
def process_group_event(groupMat, groupRates, rand):
    #get number of groups
    numGroup = groupMat.shape[1]
    
    #select random event based on propensity
    eventID = util.select_random_event(groupRates, rand[0])
    
    #get event type
    eventtRype = math.floor(eventID/numGroup)
    #get event group
    eventGroup = eventID % numGroup # % is modulo operator
    

    if eventtRype < 1: 
        #fission event - add new group and split cells
        #get parent composition
        parOld = groupMat[:,eventGroup]
        
        #calc daughter composition
        daughter = np.floor(parOld / 2)
        parNew = parOld - daughter
        
        #only add daughter if not empty
        if daughter.sum() > 0:
            #update parrent
            groupMat[:,eventGroup] = parNew
            #add new daughter group
            groupMat = np.column_stack((groupMat, daughter))
        
        numGroup = groupMat.shape[1]
        
    else:
        #extinction event - remove group
        groupMat, numGroup = remove_group(groupMat, eventGroup)
            
    return (groupMat, numGroup)



#create helper vectors for dot products
def create_helper_vector(NGroup):
    oneVecGroup = np.ones(NGroup)
    oneVecIndvR = np.ones(NGroup * 8)
    oneVecGrR = np.ones(NGroup * 2)
    return(oneVecGroup, oneVecIndvR, oneVecGrR)

"""============================================================================
Main model code
============================================================================"""

#main model 
def run_model(model_par):
    
    ## Initialize model, get rates and init matrices
    #number of steps to run
    sampleInt = model_par['sampleInt']
    maxT = model_par['maxT']
    numTSample = int(np.ceil(maxT / sampleInt)+1)
    
    #initialize group matrix
    groupMat = init_groupMat(model_par)
    
    #get matrix with random numbers
    #creates matrix with maxRandMatSize entries, it is recreated if needed
    maxRandMatSize = int(1E6)
    randMat = util.create_randMat(maxRandMatSize, 4)
    
    #initialize output matrix
    output = init_output_matrix(numTSample)
    
    #helper vector to calc sum over all cell types
    oneVec4 = np.array([1.,1.,1.,1.])
    
    #helper vector to calc sum over all groups
    numGroup = groupMat.shape[1]
    oneVecGroup, oneVecIndvR, oneVecGrR = create_helper_vector(numGroup)
    
    # get model rates
    indv_deathR, indv_cost, indv_mutationR = [float(model_par[x])
                            for x in ('indv_deathR', 'indv_cost', 'indv_mutationR')]
    
    gr_Sfission, gr_Sextinct, gr_K, gr_tau = [float(model_par[x])
                            for x in ('gr_Sfission', 'gr_Sextinct', 'gr_K', 'gr_tau')]
    
    # init counters 
    currT = 0
    ttR = 0
    sampleIdx = 0
    
    #get first sample of init state
    sampleIdx = sample_model(groupMat, output, sampleIdx, currT)
        
    ## loop time steps
    while currT <= maxT:
        
        # reset rand matrix when used up
        if ttR >= maxRandMatSize:
            randMat = util.create_randMat(maxRandMatSize, 4)
            ttR = 0            
            
        #calc rates of individual level events
        indvRates = calc_indv_rates(groupMat, 
                                    indv_cost, indv_deathR, oneVec4)
        
        #calc rates of group events
        groupRates = calc_group_rates(groupMat, 
                                      gr_Sfission, gr_Sextinct, gr_K, oneVec4, oneVecGroup)
        
        #calculate total propensities
        indvProp = oneVecIndvR @ indvRates
        groupProp = (oneVecGrR @ groupRates) / gr_tau
        totProp = indvProp + groupProp
        
        #calc time step
        dt = -1 * math.log(randMat[ttR,1]) / totProp
        
        #select group or individual event
        rescaledRand = randMat[ttR,0] * totProp
        if  rescaledRand < indvProp: 
            #individual level event - select and process individual level event
            groupDeathID = process_indv_event(groupMat, indvRates, indv_mutationR, randMat[ttR,2:3])
            
            #check if group has died
            if groupDeathID > -1: 
                #remove empty group
                groupMat, numGroup = remove_group(groupMat, groupDeathID)
                
                #if all groups have died, end simulation
                if numGroup == 0: break

                # recreate helper vector
                oneVecGroup, oneVecIndvR, oneVecGrR = create_helper_vector(numGroup) 
   
        else: 
            #group level event - select and process group level event
            groupMat, numGroup = process_group_event(groupMat, groupRates, randMat[ttR,2:3])
            
            #if all groups have died, end simulation
            if numGroup == 0: break
        
            # recreate helper vector
            oneVecGroup, oneVecIndvR, oneVecGrR = create_helper_vector(numGroup) 
            
        # update time
        currT += dt
        ttR += 1
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = sample_model(groupMat, output, sampleIdx, currT)         
    
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
    #maxTData = np.nanmax(dataStruc['time'])
    #plt.xlim((0, maxTData))
   
        
    return None


# run model, plot dynamics
def single_run_with_plot(model_par):
    # run code
    start = time.time()
    output = run_model(model_par)
    end = time.time()
    
    #print timing
    print("Elapsed time run 1 = %s" % (end - start))

    #setp figure formattRing
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
        # solver settRings
        "maxT":             500,                #total run time
        "sampleInt":        1,                  #sampling interval
        # settRings for intial condition
        "init_groupNum":    10,                 #initial # groups
        "init_groupComp":   [0.5, 0, 0.5, 0],   #initial composition of groups (fractions)
        "init_groupDens":   50,                 #initial total cell number in group
        # settRings for individual level dynamics
        "indv_cost":        0.1,                #cost of cooperation
        "indv_deathR":      0.01,               #death rate individuals
        "indv_mutationR":   1E-2,               #mutation rate to cheaters
        # settRing for group rates
        'gr_Sfission':      0.,                 #fission rate = (1 + gr_Sfission * N)/gr_tau
        'gr_Sextinct':      0.,                 #extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
        'gr_K':             2E3,                #total carrying capacity of cells
        'gr_tau':           10                  #relative rate individual and group events
    }

    single_run_with_plot(model_par)

    return None


##this piece of code is run only when this script is executed as the main 
if __name__ == "__main__":
    print("running with default parameter")
    run_w_def_parameter()
