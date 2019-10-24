#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 15 2019

Last Update Oct 22 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

from numba.types import UniTuple, Tuple
from numba import jit, void, f8, i8
import time
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import MlsGroupDynamics_utilities as util

#output variables to store
stateVar = ['NA', 'NAprime', 'NB', 'NBprime',
            'NTot', 'NCoop', 'fCoop',
            'NGroup', 'groupSizeAv', 'groupSizeMed']

"""============================================================================
Init functions 
============================================================================"""


# initialize output matrix
def init_output_matrix(model_par):
    sampleInt = model_par['sampleInt']
    maxT = model_par['maxT']
    numTSample = int(np.ceil(maxT / sampleInt) + 1)
    
    addVar = ['rms_err_NCoop', 'rms_err_NGroup', 'time']
    
    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVar]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVar]
    dTypeList3 = [(x, 'f8') for x in addVar]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3
    dType = np.dtype(dTypeList)

    # initialize outputs to NaN
    output = np.full(numTSample, np.nan, dType)
    output['time'][0] = 0

    # init matrix to track distribution fraction of cooperators 
    nBinFCoop = 20
    binFCoop = np.linspace(0, 1, nBinFCoop)
    distFCoop = np.full((numTSample, nBinFCoop-1), np.nan)

    # init matrix to track distribution fraction of cooperators
    nMax = 1 / (2 * model_par['indv_deathR']) #expected EQ. group size if all cooperator
    binGrSize = np.arange(0, nMax+1)
    distGrSize = np.full((numTSample, int(nMax)), np.nan)

    return (output, distFCoop, binFCoop, distGrSize, binGrSize)


# initialize group matrix
# each column is a group and lists number of [A,A',B,B'] cells
def init_groupMat(model_par):
    numGroup = int(model_par["init_groupNum"])
    # convert list of initComp to numpy column vector
    init_groupComp = np.c_[model_par["init_groupComp"]]
    
    if model_par['indv_interact']==0:
        init_groupComp[2:3] = 0
    
    # multiply with density and round to get cell numbers
    init_groupComp = np.round(init_groupComp * model_par["init_groupDens"])
    # init all groups with same composition
    groupMat = init_groupComp @ np.ones((1, numGroup))
    # store in C-byte order
    groupMat = np.copy(groupMat, order='C')

    return groupMat


"""============================================================================
Sample model code 
============================================================================"""

#create distribution
@jit(f8[:](f8[:], f8[:]), nopython=True)
def calc_distri(dataVec, binEdges):
    numGroup = dataVec.size
    # get distribution of average cooperator fraction per host
    binCount, _ = np.histogram(dataVec, bins=binEdges)
    distribution = binCount / numGroup
    return distribution

# calculate average cooperator fraction in total population
@jit(Tuple((f8, f8, f8, f8[:], f8[:], f8[:]))(f8[:, :]), nopython=True)
def calc_cell_stat(groupMat):
    # calc total number of individuals per group, use matrix product for speed
    Ntot_group = groupMat.sum(0)
    # calc total number of cooperators per group
    Ncoop_group = groupMat[0, :] + groupMat[2, :]
    # calc fraction cooperators per group
    fCoop_group = Ncoop_group / Ntot_group

    # calc total number cells per type
    Ntot_type = groupMat.sum(1)
    # calc total fraction cooperators
    fCoop = (Ntot_type[0] + Ntot_type[2]) / Ntot_type.sum()

    #calc group statistics
    groupSizeAv = Ntot_group.mean()
    groupSizeMed = np.median(Ntot_group)

    return (fCoop, groupSizeAv, groupSizeMed, Ntot_type, fCoop_group, Ntot_group)



# sample model
def sample_model(groupMatrix, output, distFCoop, binFCoop,
    distGrSize, binGrSize, sample_idx, currT, mavInt, rmsInt):
    # store time
    output['time'][sample_idx] = currT

    # calc number of groups
    NGroup = groupMatrix.shape[1]

    # get group statistics
    fCoop, groupSizeAv, groupSizeMed, Ntot_type, fCoop_group, Ntot_group = calc_cell_stat(
        groupMatrix)

    # calc total population sizes
    output['NA'][sample_idx] = Ntot_type[0]
    output['NAprime'][sample_idx] = Ntot_type[1]
    output['NB'][sample_idx] = Ntot_type[2]
    output['NBprime'][sample_idx] = Ntot_type[3]

    output['NTot'][sample_idx] = Ntot_type.sum()
    output['NCoop'][sample_idx] = Ntot_type[0] + Ntot_type[2]
    output['fCoop'][sample_idx] = fCoop
    
    output['NGroup'][sample_idx] = NGroup
    output['groupSizeAv'][sample_idx] = groupSizeAv
    output['groupSizeMed'][sample_idx] = groupSizeMed

    #calc moving average 
    if sample_idx >= 1:
        for varname in stateVar:
            outname = varname + '_mav'
            mav, _ = util.calc_moving_av(
                output[varname], sample_idx, mavInt)
            output[outname][sample_idx] = mav

    # calc rms error
    if sample_idx >= rmsInt:
        output['rms_err_NCoop'][sample_idx] = util.calc_rms_error(
            output['NCoop_mav'], sample_idx, rmsInt) / output['NCoop_mav'][sample_idx]
        output['rms_err_NGroup'][sample_idx] = util.calc_rms_error(
            output['NGroup_mav'], sample_idx, rmsInt) / output['NGroup_mav'][sample_idx]

    # calc distribution groupsizes
    distGrSize[sample_idx, :] = calc_distri(Ntot_group, binGrSize)

    # calc distribution fraction cooperator
    distFCoop[sample_idx, :] = calc_distri(fCoop_group, binFCoop)

    sample_idx += 1
    return sample_idx


# sample model
def sample_extinction(output, distFCoop, binFCoop,
                 distGrSize, sample_idx, currT):
    # store time
    output['time'][sample_idx] = currT


    # calc total population sizes
    for varname in stateVar:
        outname = varname + '_mav'
        output[varname][sample_idx] = 0
        output[outname][sample_idx] = 0
        
    output['rms_err_NCoop'][sample_idx] = 0
    output['rms_err_NGroup'][sample_idx] = 0


    # calc distribution groupsizes
    distGrSize[sample_idx, :] = 0

    # calc distribution fraction cooperator
    distFCoop[sample_idx,:] = 0
    
    sample_idx += 1

    return sample_idx

"""============================================================================
Sub functions individual dynamics 
============================================================================"""

# calculate birth and death rate for all groups and types
# @jit provides speedup by compling this function at start of execution
# To use @jit provide the data type of output and input, nopython=true makes compilation faster
@jit(f8[::1](f8[:, :], f8, f8, i8, f8[::1], f8[::1]), nopython=True)
def calc_indv_rates(groupMat, cost, deathR, indv_interact, oneVec4, oneVecGroup):
    # calc total number of individuals per group, use matrix product for speed
    Ntot = oneVec4 @ groupMat

    if indv_interact == 1:
        # calc birth rates per type and group
        birthA  = (1-cost) * groupMat[2, :] * groupMat[0, :] / Ntot
        birthAp = groupMat[2, :] * groupMat[1, :] / Ntot
        birthB  = (1-cost) * groupMat[0, :] * groupMat[2, :] / Ntot
        birthBp = groupMat[0, :] * groupMat[3, :] / Ntot
        
        # calc death rates per type and group
        deathA  = deathR * groupMat[0, :] * Ntot
        deathAp = deathR * groupMat[1, :] * Ntot
        deathB  = deathR * groupMat[2, :] * Ntot
        deathBp = deathR * groupMat[3, :] * Ntot
        
    elif indv_interact == 0:
        zeroVec = 0 * oneVecGroup

        # calc birth rates per type and group
        birthA  = (1-cost) * groupMat[0, :] * groupMat[0, :] / Ntot
        birthAp = groupMat[0, :] * groupMat[1, :] / Ntot
        birthB  = zeroVec
        birthBp = zeroVec  
        
        # calc death rates per type and group
        deathA  = deathR * groupMat[0, :] * Ntot
        deathAp = deathR * groupMat[1, :] * Ntot
        deathB  = zeroVec
        deathBp = zeroVec

    # combine all rates in single vector
    rates = np.concatenate((birthA, birthAp, birthB, birthBp,
                            deathA, deathAp, deathB, deathBp))

    return rates


# process individual level events
@jit(i8(f8[:, :], f8[::1], f8, f8[::1]), nopython=True)
def process_indv_event(groupMat, rateVector, mutationR, rand):
    # Note: groupMat is updated in place, it does not need to be returned
    # get number of groups
    numGroup = groupMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(rateVector, rand[0])

    # get event type
    eventtRype = math.floor(eventID/numGroup)
    # get event group
    eventGroup = eventID % numGroup  # % is modulo operator

    # track if any groups die in process
    groupDeathID = -1  # -1 is no death

    if eventtRype < 4:  # birth event
        # add cell to group, check for mutations first
        cellType = eventtRype
        if (cellType % 2) == 0:  # Wild type cell, can mutate
            if rand[1] < mutationR:  # birth with mutation
                groupMat[cellType+1, eventGroup] += 1
            else:  # birth without mutation
                groupMat[cellType, eventGroup] += 1
        else:  # cheater cell, cannot mutate
            groupMat[cellType, eventGroup] += 1
    else:  # death event
        # remove cell from group
        cellType = eventtRype - 4
        groupMat[cellType, eventGroup] -= 1

        # kill group if last cell died
        # use two stage check for increased speed
        if groupMat[cellType, eventGroup] == 0:  # killed last of type
            #NInGroup = oneVec4 @ groupMat[:, eventGroup]
            NInGroup = groupMat[:, eventGroup].sum()
            if NInGroup == 0:  # all other types are zero too
                groupDeathID = int(eventGroup)

    return groupDeathID


"""============================================================================
Sub functions group dynamics 
============================================================================"""

# remove group from group matrix
@jit(Tuple((f8[:, :], i8))(f8[:, :], i8), nopython=True)
def remove_group(groupMat, groupDeathID):
    # Note: groupMat is re-created, it has to be returned
    # create helper vector
    numGroup = groupMat.shape[1]
    hasDied = np.zeros(numGroup)
    hasDied[groupDeathID] = 1

    # copy remaining groups to new matrix
    groupMat = groupMat[:, hasDied == 0]

    numGroup -= 1

    return (groupMat, numGroup)


# calculate fission and extinction rate of all groups
@jit(f8[::1](f8[:, :], f8, f8, f8, f8[::1], f8[::1]), nopython=True)
def calc_group_rates(groupMat, group_Sfission, group_Sextinct, group_K, oneVec4, oneVecGroup):
    # calc total number of individuals per group, use matrix product for speed
    Ntot_group = oneVec4 @ groupMat

    # calc total number of individuals
    Ntot = oneVecGroup @ Ntot_group

    # calc fission rate
    if group_Sfission > 0:
        fissionR = oneVecGroup + group_Sfission * Ntot_group
    else:
        fissionR = oneVecGroup

    # calc extinction rate
    groupDeathRate = Ntot / group_K
    if group_Sextinct > 0:
        extinctR = (oneVecGroup + group_Sextinct * Ntot_group) * groupDeathRate
        extinctR[extinctR < 0] = 0
    else:
        extinctR = oneVecGroup * groupDeathRate

    # combine all rates in single vector
    rates = np.concatenate((fissionR, extinctR))

    return rates


@jit(Tuple((f8[:],f8[:,:]))(f8[:], f8, f8), nopython=True)
def fission_groups(parentGroup, offspr_size, offspr_frac):        
    #number of cells in parents
    cellNumPar = parentGroup.sum()
   
    #calc number of offspring, draw from Poisson distribution
    # <#offspring> = numCellsToOffspr / sizeOfOffspr
    # = offspr_frac * cellNumPar / offspr_size * cellNumPar
    # = offspr_frac / offspr_size
    expectedNumOffSpr = offspr_frac / offspr_size
    numOffSpr = int(np.random.poisson(expectedNumOffSpr))
    #calc total number of cells passed on to offspring, keep at least 1 cell in parent
    numCellsToOffspr = int(min(round(offspr_frac * cellNumPar), cellNumPar-1))

    #assign cells to offspring
    if numOffSpr > 0:
        parrentPool = parentGroup
        #init offspring array
        offspring = np.zeros((4, numOffSpr))

        #perform random sampling
        randMat = util.create_randMat(numCellsToOffspr, 1)

        for ii in range(numCellsToOffspr):
            #randomly pick cell from parent using weighted lottery
            typePicked = util.select_random_event(parrentPool, randMat[ii, 0])
            #deal round the table: select offsping to assign cell to
            offspringPicked = ii % numOffSpr 
            #assign cell to offspring
            offspring[typePicked, offspringPicked] += 1
            #remove cell from parent
            parrentPool[typePicked] -= 1
        
        #remove empty daughter groups
        numCellInOffspr = offspring.sum(0)
        offspring = offspring[:, numCellInOffspr > 0]

        #update parent to new state
        parrentNew = parrentPool
    else:
        #nothing happens
        parrentNew = parentGroup
        offspring = np.zeros((4,1))
            
    return (parrentNew, offspring)


# process individual level events
@jit(Tuple((f8[:, :], i8))(f8[:, :], f8[::1], f8[::1], f8, f8), nopython=True)
def process_group_event(groupMat, groupRates, rand, offspr_size, offspr_frac):
    # get number of groups
    numGroup = groupMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(groupRates, rand[0])

    # get event type
    eventtRype = math.floor(eventID/numGroup)
    # get event group
    eventGroup = eventID % numGroup  # % is modulo operator

    if eventtRype < 1:
        # fission event - add new group and split cells
        # get parent composition
        parentGroup = groupMat[:, eventGroup]

        #perform fission process
        if offspr_size > 0:
            parrentNew, offspring = fission_groups(parentGroup, offspr_size, offspr_frac)

            # only add daughter if not empty
            if offspring.sum() > 0:
                # update parrent
                groupMat[:, eventGroup] = parrentNew
                # add new daughter group
                groupMat = np.column_stack((groupMat, offspring))
    
            numGroup = groupMat.shape[1]

    else:
        # extinction event - remove group
        groupMat, numGroup = remove_group(groupMat, eventGroup)

    return (groupMat, numGroup)


# create helper vectors for dot products
def create_helper_vector(NGroup):
    oneVecGroup = np.ones(NGroup)
    oneVecIndvR = np.ones(NGroup * 8)
    oneVecGrR = np.ones(NGroup * 2)
    return(oneVecGroup, oneVecIndvR, oneVecGrR)


"""============================================================================
Main model code
============================================================================"""

# main model
def run_model(model_par):

    # Initialize model, get rates and init matrices
    # get time rates
    sampleInt = model_par['sampleInt']
    maxT = model_par['maxT']
    # calc time windows to average over
    mavInt = int(
        math.ceil(model_par['mav_window'] / model_par['sampleInt']))
    rmsInt = int(
        math.ceil(model_par['rms_window'] / model_par['sampleInt']))

    if 'minT' in model_par:
        minTRun = max(model_par['minT'], rmsInt+1)
    else:
        minTRun = rmsInt + 1
        
    # initialize group matrix
    groupMat = init_groupMat(model_par)

    # get matrix with random numbers
    # creates matrix with maxRandMatSize entries, it is recreated if needed
    maxRandMatSize = int(1E6)
    randMat = util.create_randMat(maxRandMatSize, 4)

    # initialize output matrix
    output, distFCoop, binFCoop, distGrSize, binGrSize = init_output_matrix(model_par)

    # helper vector to calc sum over all cell types
    oneVec4 = np.array([1., 1., 1., 1.])

    # helper vector to calc sum over all groups
    numGroup = groupMat.shape[1]
    oneVecGroup, oneVecIndvR, oneVecGrR = create_helper_vector(numGroup)

    # get model rates
    indv_deathR, indv_cost, indv_mutationR = [float(model_par[x])
                                              for x in ('indv_deathR', 'indv_cost', 'indv_mutationR')]

    indv_interact = int(model_par['indv_interact'])

    gr_Sfission, gr_Sextinct, gr_K, gr_tau = [float(model_par[x])
                                              for x in ('gr_Sfission', 'gr_Sextinct', 'gr_K', 'gr_tau')]

    offspr_size, offspr_frac = [float(model_par[x])
                                for x in ('offspr_size', 'offspr_frac')]
    
    #check rates
    if offspr_size > 0.5: 
        print('cannot do that: offspr_size < 0.5 and offspr_size < offspr_frac < 1')
        raise ValueError
    elif offspr_frac < offspr_size or offspr_frac > (1-offspr_size):
        print('cannot do that: offspr_frac should be offspr_size < offspr_frac < 1-offspr_size')
        raise ValueError
        

    # init counters
    currT = 0
    ttR = 0
    sampleIdx = 0

    # get first sample of init state
    sampleIdx = sample_model(groupMat, output, distFCoop, binFCoop,
                             distGrSize, binGrSize, sampleIdx, currT, mavInt, rmsInt)

    # loop time steps
    while currT <= maxT:

        # reset rand matrix when used up
        if ttR >= maxRandMatSize:
            randMat = util.create_randMat(maxRandMatSize, 4)
            ttR = 0

        # calc rates of individual level events
        indvRates = calc_indv_rates(groupMat,
                                    indv_cost, indv_deathR, indv_interact, oneVec4, oneVecGroup)

        # calc rates of group events
        groupRates = calc_group_rates(groupMat,
                                      gr_Sfission, gr_Sextinct, gr_K, oneVec4, oneVecGroup)

        # calculate total propensities
        indvProp = oneVecIndvR @ indvRates
        groupProp = (oneVecGrR @ groupRates) / gr_tau
        totProp = indvProp + groupProp

        # calc time step
        dt = -1 * math.log(randMat[ttR, 1]) / totProp

        # select group or individual event
        rescaledRand = randMat[ttR, 0] * totProp
        if rescaledRand < indvProp:
            # individual level event - select and process individual level event
            groupDeathID = process_indv_event(
                groupMat, indvRates, indv_mutationR, randMat[ttR, 2:3])

            # check if group has died
            if groupDeathID > -1:
                # remove empty group
                groupMat, numGroup = remove_group(groupMat, groupDeathID)

                # if all groups have died, end simulation
                if numGroup == 0:
                    sampleIdx = sample_extinction(output, distFCoop, binFCoop,
                                      distGrSize, sampleIdx, currT)
                    break

                # recreate helper vector
                oneVecGroup, oneVecIndvR, oneVecGrR = create_helper_vector(
                    numGroup)

        else:
            # group level event - select and process group level event
            groupMat, numGroup = process_group_event(
                groupMat, groupRates, randMat[ttR, 2:3], offspr_size, offspr_frac)

            # if all groups have died, end simulation
            if numGroup == 0:
                sampleIdx = sample_extinction(output, distFCoop, binFCoop,
                                  distGrSize, sampleIdx, currT)
                break

            # recreate helper vector
            oneVecGroup, oneVecIndvR, oneVecGrR = create_helper_vector(
                numGroup)

        # update time
        currT += dt
        ttR += 1
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = sample_model(groupMat, output, distFCoop, binFCoop,
                                     distGrSize, binGrSize, sampleIdx, currT, mavInt, rmsInt)
            # check if steady state has been reached
            if currT > minTRun:
                NCoopStable = output['rms_err_NCoop'][sampleIdx - 1] \
                    < model_par['rms_err_trNCoop']
                NGroupStable = output['rms_err_NGroup'][sampleIdx - 1] \
                    < model_par['rms_err_trNGr']

                if NCoopStable and NGroupStable:
                    break

    # cut off non existing time points at end
    output = output[0:sampleIdx]
    distFCoop = distFCoop[0:sampleIdx, :]
    distGrSize = distGrSize[0:sampleIdx, :]
    
    if output['NCoop'][-1] == 0:
        output['NCoop_mav'][-1] = 0
    
    return (output, distFCoop, distGrSize)


"""============================================================================
Code that calls model and plots results
============================================================================"""

# code to plot data
# set type to "lin" or "log" to switch between lin or log plot

def plot_data(dataStruc, FieldName, type='lin'):
    # linear plot
    if type == 'lin':
        plt.plot(dataStruc['time'], dataStruc[FieldName], label=FieldName)
    # log plot
    elif type == 'log':
        plt.semilogy(dataStruc['time'], dataStruc[FieldName], label=FieldName)

    # set x-label
    plt.xlabel("time")
    #maxTData = np.nanmax(dataStruc['time'])
    #plt.xlim((0, maxTData))

    return None


def plot_heatmap(fig, axs, data, yName, type='lin'):
    # linear plot
    if type == 'lin':
        currData = data.transpose()
        labelName = "density"
        cRange = [0, 0.1]
    # log plot
    elif type == 'log':
        currData = np.log10(
            data.transpose() + np.finfo(float).eps)
        labelName = "log10 density"
        cRange = [-2, -1]

    im = axs.imshow(currData, cmap="viridis",
                    interpolation='nearest',
                    extent=[0, 1, 0, 1],
                    origin='lower',
                    vmin = cRange[0],
                    vmax = cRange[1],
                    aspect='auto')
    axs.set_xticks([0, 1])
    axs.set_yticks([0, 1])
    axs.set_ylabel(yName)
    axs.set_xlabel('time')
    fig.colorbar(im, ax=axs, orientation='vertical',
                fraction=.1, label=labelName)
    axs.set_yticklabels([0, 1])

    return None

# run model, plot dynamics
def single_run_with_plot(model_par):
    # run code
    start = time.time()
    output, distFCoop, distGrSize = run_model(model_par)
    end = time.time()

    # print timing
    print("Elapsed time run 1 = %s" % (end - start))

    # setup figure formattRing
    font = {'family': 'arial',
            'weight': 'normal',
            'size': 6}
    matplotlib.rc('font', **font)

    # open figure
    fig = plt.figure()
    nR = 3
    nC = 2

    # plot number of groups
    plt.subplot(nR, nC, 1)
    plot_data(output, "NGroup")
    plot_data(output, "NGroup_mav")

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

    # plot fraction of coop
    plt.subplot(nR, nC, 3)
    plot_data(output, "NCoop")
    plot_data(output, "NCoop_mav")
    plt.ylabel("density cooperator")
    plt.legend()

    # plot rms error
    plt.subplot(nR, nC, 4)
    plot_data(output, "rms_err_NCoop",type='log')
    plot_data(output, "rms_err_NGroup",type='log')
    plt.legend()
    plt.ylabel("rms error")

    #plot distribution group size
    axs = plt.subplot(nR, nC, 5)
    plot_heatmap(fig, axs, distGrSize, 'group size', type='lin')

    #plot distribution fraction coop
    axs = plt.subplot(nR, nC, 6)
    plot_heatmap(fig, axs, distFCoop, 'coop. freq.', type='lin')

    # set figure size
    fig.set_size_inches(8, 6)
    plt.tight_layout()  # cleans up figure and aligns things nicely

    return None


# run model with default parameters
def run_w_def_parameter():
    model_par = {
        # solver settings
        "maxT":             200,  # total run time
        "minT":             100,   # min run time
        "sampleInt":        1,     # sampling interval
        "mav_window":       100,   # average over this time window
        "rms_window":       100,   # calc rms change over this time window
        "rms_err_treshold": 2E-2,  # when to stop calculations
        # settings for initial condition
        "init_groupNum":    100,  # initial # groups
        # initial composition of groups (fractions)
        "init_groupComp":   [0.5, 0, 0.5, 0],
        "init_groupDens":   100,  # initial total cell number in group
        # settings for individual level dynamics
        "indv_cost":        0.05,  # cost of cooperation
        "indv_deathR":      0.001, # death rate individuals
        "indv_mutationR":   1E-2,  # mutation rate to cheaters
        "indv_interact":    1,      #0 1 to turn off/on crossfeeding
        # setting for group rates
        'gr_Sfission':      0.,    # fission rate = (1 + gr_Sfission * N)/gr_tau
        'gr_Sextinct':      0.,    # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
        'gr_K':             2E3,   # total carrying capacity of cells
        'gr_tau':           100,   # relative rate individual and group events
        # settings for fissioning
        'offspr_size':      0.05,  # offspr_size <= 0.5 and
        'offspr_frac':      0.9    # offspr_size < offspr_frac < 1-offspr_size'

    }

    single_run_with_plot(model_par)

    return None

#run model store only final state 
def single_run_finalstate(model_par):
    # run model
    output, distFCoop, distGrSize = run_model(model_par)

    #input parameters to store
    parList = ['indv_cost', 'indv_deathR', 'indv_mutationR', 'indv_interact',
               'gr_Sfission', 'gr_Sextinct', 'gr_K', 'gr_tau',
               'offspr_size', 'offspr_frac']

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVar]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVar]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3
    dType = np.dtype(dTypeList)

    output_matrix = np.zeros(1, dType)

    # store final state
    for var in stateVar:
        output_matrix[var] = output[var][-1]
        var_mav = var + '_mav'
        output_matrix[var_mav] = output[var_mav][-1]

    for par in parList:
        output_matrix[par] = model_par[par]

    endDistFCoop = distFCoop[-1,:]
    endDistGrSize = distGrSize[-1, :]

    return (output_matrix, endDistFCoop, endDistGrSize)


# this piece of code is run only when this script is executed as the main
if __name__ == "__main__":
    print("running with default parameter")
    run_w_def_parameter()
