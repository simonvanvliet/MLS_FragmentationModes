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
import math
import numpy as np
import MlsGroupDynamics_utilities as util

#output variables to store
stateVar = ['NTot', 'NCoop', 'fCoop',
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
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
                ['N%imut' % x for x in range(model_par['indv_NType'])]

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVarPlus]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVarPlus]
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
    nMax = model_par['indv_K'] #expected EQ. group size if all cooperator
    binGrSize = np.arange(0., nMax+1.)
    distGrSize = np.full((numTSample, int(nMax)), np.nan)

    return (output, distFCoop, binFCoop, distGrSize, binGrSize)


# initialize group matrix
# each column is a group and lists number of [A,A',B,B'] cells
def init_groupMat(model_par):
    #get properties
    numGroup = int(model_par["init_groupNum"])
    numType = int(model_par['indv_NType'])

    #create group composition vector
    init_groupComp = np.zeros((numType*2, 1))
    nCoop = round(model_par["init_groupDens"] * model_par['init_fCoop'] / model_par['indv_NType'])
    nDef  = round(model_par["init_groupDens"] * (1 - model_par['init_fCoop']) / model_par['indv_NType'])
    init_groupComp[0::2] = nCoop
    init_groupComp[1::2] = nDef

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
@jit(Tuple((f8, f8, f8, f8, f8[:], f8[:], f8[:]))(f8[:, :]), nopython=True)
def calc_cell_stat(groupMat):
    # calc total number of individuals per group, use matrix product for speed
    Ntot_group = groupMat.sum(0)
    # calc total number of cooperators per group
    Ncoop_group = groupMat[0::2, :].sum(0)
    # calc fraction cooperators per group
    fCoop_group = Ncoop_group / Ntot_group

    # calc total number cells per type
    Ntot_type = groupMat.sum(1)
    # calc total fraction cooperators
    NTot = Ntot_type.sum()
    NCoop = Ntot_type[0::2].sum()

    #calc group statistics
    groupSizeAv = Ntot_group.mean()
    groupSizeMed = np.median(Ntot_group)

    return (NTot, NCoop, groupSizeAv, groupSizeMed, Ntot_type, fCoop_group, Ntot_group)


# sample model
def sample_model(groupMatrix, output, distFCoop, binFCoop,
                 distGrSize, binGrSize, sample_idx, currT, mavInt, rmsInt, stateVarPlus):
    # store time
    output['time'][sample_idx] = currT

    # calc number of groups
    shapeGroupMat= groupMatrix.shape
    NGroup = shapeGroupMat[1]
    NType = int(shapeGroupMat[0] / 2)

    # get group statistics
    NTot, NCoop, groupSizeAv, groupSizeMed, Ntot_type, fCoop_group, Ntot_group = calc_cell_stat(
        groupMatrix)

    # calc total population sizes
    for tt in range(NType):
        output['N%i' %tt][sample_idx] = Ntot_type[tt*2]
        output['N%imut' %tt][sample_idx] = Ntot_type[tt*2+1]
       
    output['NTot'][sample_idx] = NTot
    output['NCoop'][sample_idx] = NCoop
    output['fCoop'][sample_idx] = NCoop / NTot
    
    output['NGroup'][sample_idx] = NGroup
    output['groupSizeAv'][sample_idx] = groupSizeAv
    output['groupSizeMed'][sample_idx] = groupSizeMed

    #calc moving average 
    if sample_idx >= 1:
        for varname in stateVarPlus:
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
                      distGrSize, sample_idx, currT, stateVarPlus):
    # store time
    output['time'][sample_idx] = currT

    # calc total population sizes
    for varname in stateVarPlus:
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
@jit(void(f8[::1], f8[:, ::1], f8[::1], f8, f8[::1], f8[::1]), nopython=True)
def calc_indv_rates(rates, groupMat, bVec, deathR, oneVecType, oneVecGroup):
    # calc total number of individuals per group, use matrix product for speed
    Ntot = oneVecType @ groupMat
    
    #get number of types and groups
    nType = int(oneVecType.size / 2)
    nGroup = oneVecGroup.size
        
    #loop cell types
    for tt in range(nType):
        #setup indices
        cIdx = 2 * tt
        dIdx = 2 * tt + 1
        bIdxC1 = cIdx * nGroup 
        bIdxD1 = dIdx * nGroup
        dIdxC1 = bIdxC1 + 2 * nType * nGroup
        dIdxD1 = bIdxD1 + 2 * nType * nGroup
        
        #calc density of cooperating partners
        if nType == 1:
            #coopPart = n0/ntot
            coopPart = groupMat[0, :] / Ntot
        else:
            #coopPart for type 1 = n2/ntot * n3/ntot * ... * n(nType)/ntot
            #coopPart for type 2 = n1/ntot * n3/ntot * ... * n(nType)/ntot
            #etc
            #vector of ones with size nGroup
            coopPart = np.copy(oneVecGroup)
            for pp in range(nType):
                if pp != tt: #exlude self
                    coopPart *= groupMat[pp*2, :] / Ntot
                
        # calc rates
        rates[bIdxC1: bIdxC1 + nGroup] = bVec[cIdx] * coopPart * groupMat[cIdx, :]
        rates[bIdxD1: bIdxD1 + nGroup] = bVec[dIdx] * coopPart * groupMat[dIdx, :] 
        rates[dIdxC1: dIdxC1 + nGroup] = deathR * Ntot * groupMat[cIdx, :] 
        rates[dIdxD1: dIdxD1 + nGroup] = deathR * Ntot * groupMat[dIdx, :]

    return None

# process individual level events
@jit(i8(f8[:, ::1], f8[::1], f8, f8[::1]), nopython=True)
def process_indv_event(groupMat, rateVector, mutationR, rand):
    # Note: groupMat is updated in place, it does not need to be returned

    # calc number of groups
    shapeGroupMat = groupMat.shape
    NGroup = shapeGroupMat[1]
    NTypeWMut = shapeGroupMat[0]
    

    # select random event based on propensity
    eventID = util.select_random_event(rateVector, rand[0])

    # get event type
    eventType = math.floor(eventID/NGroup)
    # get event group
    eventGroup = eventID % NGroup  # % is modulo operator

    # track if any groups die in process
    groupDeathID = -1  # -1 is no death

    if eventType < NTypeWMut:  # birth event
        # add cell to group, check for mutations first
        cellType = eventType
        if (cellType % 2) == 0:  # Wild type cell, can mutate
            if rand[1] < mutationR:  # birth with mutation
                groupMat[cellType+1, eventGroup] += 1
            else:  # birth without mutation
                groupMat[cellType, eventGroup] += 1
        else:  # cheater cell, cannot mutate
            groupMat[cellType, eventGroup] += 1
    else:  # death event
        # remove cell from group
        cellType = eventType - NTypeWMut
        groupMat[cellType, eventGroup] -= 1

        # kill group if last cell died
        # use two stage check for increased speed
        if groupMat[cellType, eventGroup] == 0:  # killed last of type
            #NInGroup = oneVecType @ groupMat[:, eventGroup]
            NInGroup = groupMat[:, eventGroup].sum()
            if NInGroup == 0:  # all other types are zero too
                groupDeathID = int(eventGroup)

    return groupDeathID


"""============================================================================
Sub functions group dynamics 
============================================================================"""

# remove group from group matrix
@jit(Tuple((f8[:, ::1], i8))(f8[:, ::1], i8), nopython=True)
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
@jit(f8[::1](f8[:, ::1], f8, f8, f8, f8[::1], f8[::1]), nopython=True)
def calc_group_rates(groupMat, group_Sfission, group_Sextinct, group_K, oneVecType, oneVecGroup):
    # calc total number of individuals per group, use matrix product for speed
    Ntot_group = oneVecType @ groupMat

#    # calc total number of individuals
#    Ntot = oneVecGroup @ Ntot_group

    # calc fission rate
    if group_Sfission > 0:
        fissionR = oneVecGroup + group_Sfission * Ntot_group
    else:
        fissionR = oneVecGroup

    # calc extinction rate
    groupDeathRate = oneVecGroup.size / group_K
    if group_Sextinct > 0:
        extinctR = (oneVecGroup + group_Sextinct * Ntot_group) * groupDeathRate
        extinctR[extinctR < 0] = 0
    else:
        extinctR = oneVecGroup * groupDeathRate

    # combine all rates in single vector
    rates = np.concatenate((fissionR, extinctR))

    return rates


@jit(Tuple((f8[::1],f8[:, ::1]))(f8[::1], f8, f8), nopython=True)
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
        offspring = np.zeros((parentGroup.size, numOffSpr))

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
        offspring = np.zeros((parentGroup.size, 1))
            
    return (parrentNew, offspring)


# process individual level events
@jit(Tuple((f8[:, ::1], i8))(f8[:, ::1], f8[::1], f8[::1], f8, f8), nopython=True)
def process_group_event(groupMat, groupRates, rand, offspr_size, offspr_frac):
    # get number of groups
    numGroup = groupMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(groupRates, rand[0])

    # get event type
    eventype = math.floor(eventID/numGroup)
    # get event group
    eventGroup = eventID % numGroup  # % is modulo operator

    if eventype < 1:
        # fission event - add new group and split cells
        # get parent composition
        parentGroup = groupMat[:, eventGroup].copy()

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
def create_helper_vector(NGroup, NType):
    oneVecGroup = np.ones(NGroup)
    oneVecIndvR = np.ones(NGroup * NType * 4)
    oneVecGrR = np.ones(NGroup * 2)
    #init rates matrix
    rates = np.ones(4 * NType * NGroup)   
    
    return(oneVecGroup, oneVecIndvR, oneVecGrR, rates)


"""============================================================================
Main model code
============================================================================"""

# main model
def run_model(model_par):

    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]


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
  
    # helper vector to calc sum over all groups
    numGroup=groupMat.shape[1]
    numType = int(model_par['indv_NType'])
    oneVecGroup, oneVecIndvR, oneVecGrR, indvRates = create_helper_vector(
        numGroup, numType)
    
    oneVecType = np.ones(2 * numType)

    # get model rates
    indv_K, indv_cost, indv_mutationR, indv_asymmetry = [float(model_par[x])
                                                         for x in ('indv_K', 'indv_cost', 'indv_mutationR', 'indv_asymmetry')]

    #calc birth rates to keep constant EQ group size when all cells are cooperators
    bVecCoop = 1 / indv_asymmetry**(np.arange(numType))
    bVecCoop *= (bVecCoop.sum()**(numType - 1)) / np.prod(bVecCoop)
    #include costs and calc birth rates for cooperators and defectors respectively
    bVec=np.kron(bVecCoop, np.array([(1 - indv_cost), 1]))
    #convert caryinf capacity to death rate
    indv_deathR = 1 / indv_K

    #get group rates
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
                             distGrSize, binGrSize, sampleIdx, currT, mavInt, rmsInt, stateVarPlus)

    # loop time steps
    while currT <= maxT:

        # reset rand matrix when used up
        if ttR >= maxRandMatSize:
            randMat = util.create_randMat(maxRandMatSize, 4)
            ttR = 0

        # calc rates of individual level events
        calc_indv_rates(indvRates, groupMat, bVec,
                                    indv_deathR, oneVecType, oneVecGroup)

        # calc rates of group events
        groupRates = calc_group_rates(groupMat,
                                      gr_Sfission, gr_Sextinct, gr_K, oneVecType, oneVecGroup)

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
                groupMat, indvRates, indv_mutationR, randMat[ttR, 2:4])

            # check if group has died
            if groupDeathID > -1:
                # remove empty group
                groupMat, numGroup = remove_group(groupMat, groupDeathID)

                # if all groups have died, end simulation
                if numGroup == 0:
                    sampleIdx = sample_extinction(output, distFCoop, binFCoop,
                                                  distGrSize, sampleIdx, currT, stateVarPlus)
                    break

                # recreate helper vector
                oneVecGroup, oneVecIndvR, oneVecGrR, indvRates = create_helper_vector(
                    numGroup, numType)

        else:
            # group level event - select and process group level event
            groupMat, numGroup = process_group_event(
                groupMat, groupRates, randMat[ttR, 2:4], offspr_size, offspr_frac)

            # if all groups have died, end simulation
            if numGroup == 0:
                sampleIdx = sample_extinction(output, distFCoop, binFCoop,
                                              distGrSize, sampleIdx, currT, stateVarPlus)
                break

            # recreate helper vector
            oneVecGroup, oneVecIndvR, oneVecGrR, indvRates = create_helper_vector(
                numGroup, numType)

        # update time
        currT += dt
        ttR += 1
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = sample_model(groupMat, output, distFCoop, binFCoop,
                                     distGrSize, binGrSize, sampleIdx, currT, mavInt, rmsInt, stateVarPlus)
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

#run model store only final state 
def single_run_finalstate(model_par):
    # run model
    output, distFCoop, distGrSize = run_model(model_par)

    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutationR', 'indv_asymmetry',
               'gr_Sfission', 'gr_Sextinct', 'gr_K', 'gr_tau',
               'offspr_size', 'offspr_frac']

    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVarPlus]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVarPlus]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3
    dType = np.dtype(dTypeList)

    output_matrix = np.zeros(1, dType)

    # store final state
    for var in stateVarPlus:
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

    model_par = {
        # solver settings
        "maxT":             1000,  # total run time
        "minT":             200,   # min run time
        "sampleInt":        1,     # sampling interval
        "mav_window":       200,   # average over this time window
        "rms_window":       200,   # calc rms change over this time window
        "rms_err_trNCoop":    2E-2,  # when to stop calculations
        "rms_err_trNGr":    1E-1,  # when to stop calculations
        # settings for initial condition
        "init_groupNum":    10,  # initial # groups
        # initial composition of groups (fractions)
        "init_fCoop":       1,
        "init_groupDens":   50,  # initial total cell number in group
        # settings for individual level dynamics
        "indv_NType":       2,
        "indv_cost":        0.01,  # cost of cooperation
        "indv_K":           50,  # total group size at EQ if f_coop=1
        "indv_mutationR":   1E-3,  # mutation rate to cheaters
        # difference in growth rate b(j+1) = b(j) / asymmetry
        "indv_asymmetry":    5,
        # setting for group rates
        # fission rate = (1 + gr_Sfission * N)/gr_tau
        'gr_Sfission':      0.,
        # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
        'gr_Sextinct':      0.,
        'gr_K':             20,   # carrying capacity of groups
        'gr_tau':           100,   # relative rate individual and group events
        # settings for fissioning
        'offspr_size':      0.5,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'

    }

    output, distFCoop, distGrSize = run_model(model_par)
