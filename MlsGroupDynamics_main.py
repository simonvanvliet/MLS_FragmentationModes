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
import time


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
    NGroup = int(model_par["init_groupNum"])
    NType = int(model_par['indv_NType'])

    #create group composition vector
    init_groupComp = np.zeros((NType*2, 1))
    nCoop = round(model_par["init_groupDens"] * model_par['init_fCoop'] / model_par['indv_NType'])
    nDef  = round(model_par["init_groupDens"] * (1 - model_par['init_fCoop']) / model_par['indv_NType'])
    init_groupComp[0::2] = nCoop
    init_groupComp[1::2] = nDef

    # init all groups with same composition
    groupMat = init_groupComp @ np.ones((1, NGroup))
    # store in C-byte order
    groupMat = np.copy(groupMat, order='C')

    return groupMat


"""============================================================================
Sample model code 
============================================================================"""

#create distribution
@jit(f8[:](f8[:], f8[:]), nopython=True)
def calc_distri(dataVec, binEdges):
    NGroup = dataVec.size
    # get distribution of average cooperator fraction per host
    binCount, _ = np.histogram(dataVec, bins=binEdges)
    distribution = binCount / NGroup
    return distribution

# calculate average cooperator fraction in total population
@jit(Tuple((f8, f8, f8, f8, f8[:], f8[:], f8[:]))(f8[:, :]), nopython=True)
def calc_cell_stat(groupMat):
    # calc total number of individuals per group, use matrix product for speed
    grSizeVec = groupMat.sum(0)
    # calc total number of cooperators per group
    Ncoop_group = groupMat[0::2, :].sum(0)
    # calc fraction cooperators per group
    fCoop_group = Ncoop_group / grSizeVec

    # calc total number cells per type
    NTot_type = groupMat.sum(1)
    # calc total fraction cooperators
    NTot = NTot_type.sum()
    NCoop = NTot_type[0::2].sum()

    #calc group statistics
    groupSizeAv = grSizeVec.mean()
    groupSizeMed = np.median(grSizeVec)

    return (NTot, NCoop, groupSizeAv, groupSizeMed, NTot_type, fCoop_group, grSizeVec)


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
    NTot, NCoop, groupSizeAv, groupSizeMed, NTot_type, fCoop_group, grSizeVec = calc_cell_stat(
        groupMatrix)

    # calc total population sizes
    for tt in range(NType):
        output['N%i' %tt][sample_idx] = NTot_type[tt*2]
        output['N%imut' %tt][sample_idx] = NTot_type[tt*2+1]
       
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
    distGrSize[sample_idx, :] = calc_distri(grSizeVec, binGrSize)

    # calc distribution fraction cooperator
    distFCoop[sample_idx, :] = calc_distri(fCoop_group, binFCoop)

    sample_idx += 1
    return sample_idx

# sample model
def sample_nan(groupMatrix, output, distFCoop, binFCoop,
                 distGrSize, binGrSize, sample_idx, currT, mavInt, rmsInt, stateVarPlus):
    # store time
    output['time'][sample_idx] = currT

    # calc number of groups
    shapeGroupMat= groupMatrix.shape
    NType = int(shapeGroupMat[0] / 2)

    # calc total population sizes
    for tt in range(NType):
        output['N%i' %tt][sample_idx] = np.nan
        output['N%imut' %tt][sample_idx] = np.nan
       
    output['NTot'][sample_idx] = np.nan
    output['NCoop'][sample_idx] = np.nan
    output['fCoop'][sample_idx] = np.nan
    
    output['NGroup'][sample_idx] = np.nan
    output['groupSizeAv'][sample_idx] = np.nan
    output['groupSizeMed'][sample_idx] = np.nan

    #calc moving average 
    for varname in stateVarPlus:
        outname = varname + '_mav'
        output[outname][sample_idx] = np.nan

    return None

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
@jit(void(f8[::1], f8[:, ::1], f8[::1], f8[::1], f8, f8, i8, i8), nopython=True)
def calc_indv_rates(rates, groupMat, grSizeVec, indvBirthVec, deathR, delta_indv, NType, NGroup):
    
    #loop cell types
    for tt in range(NType):
        #setup indices
        cIdx = 2 * tt
        dIdx = 2 * tt + 1
        bIdxC1 = cIdx * NGroup 
        bIdxD1 = dIdx * NGroup
        dIdxC1 = bIdxC1 + 2 * NType * NGroup
        dIdxD1 = bIdxD1 + 2 * NType * NGroup
        
        #calc density of cooperating partners
        if NType == 1:
            #coopPart = n0/grSizeVec
            coopPart = groupMat[0, :] / grSizeVec
        else:
            #coopPart for type 1 = n2/grSizeVec * n3/grSizeVec * ... * n(NType)/grSizeVec
            #coopPart for type 2 = n1/grSizeVec * n3/N_gr * ... * n(NType)/N_gr
            #etc
            #vector of ones with size NGroup
            coopPart = np.ones(NGroup)
            for pp in range(NType):
                if pp != tt: #exlude self
                    coopPart *= groupMat[pp*2, :] / grSizeVec
                
        # calc rates
        rates[bIdxC1: bIdxC1 + NGroup] = indvBirthVec[cIdx] * coopPart * groupMat[cIdx, :]
        rates[bIdxD1: bIdxD1 + NGroup] = indvBirthVec[dIdx] * coopPart * groupMat[dIdx, :] 
        if delta_indv != 0:
            rates[dIdxC1: dIdxC1 + NGroup] = deathR * groupMat[cIdx, :] * (grSizeVec ** delta_indv)
            rates[dIdxD1: dIdxD1 + NGroup] = deathR * groupMat[dIdx, :] * (grSizeVec ** delta_indv)
        else:
            rates[dIdxC1: dIdxC1 + NGroup] = deathR * groupMat[cIdx, :]
            rates[dIdxD1: dIdxD1 + NGroup] = deathR * groupMat[dIdx, :] 

    return None

# process individual level events
@jit(i8(f8[:, ::1], f8[::1], f8, f8[::1], i8, i8), nopython=True)
def process_indv_event(groupMat, indvRates, mutR, rand, NType, NGroup):
    # Note: groupMat is updated in place, it does not need to be returned 
    NTypeWMut = NType*2
    
    # select random event based on propensity
    eventID = util.select_random_event(indvRates, rand[0])

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
            if rand[1] < mutR:  # birth with mutation
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
            #NINGroup = oneVecType @ groupMat[:, eventGroup]
            NINGroup = groupMat[:, eventGroup].sum()
            if NINGroup == 0:  # all other types are zero too
                groupDeathID = int(eventGroup)

    return groupDeathID


"""============================================================================
Sub functions migration dynamics 
============================================================================"""

# process migration event
@jit(i8(f8[:, ::1], f8[::1], i8, i8, f8[::1]), nopython=True)
def process_migration_event(groupMat, grSizeVec, NGroup, NType, rand):
    # Note: groupMat is updated in place, it does not need to be returned

    # select random group of origin based on size
    grpIDSource = util.select_random_event(grSizeVec, rand[0])

    # select random type of migrant based on population size
    cellType = util.select_random_event(groupMat[:, grpIDSource], rand[1])
    
    # select random target group
    grpIDTarget = int(np.floor(rand[2] * NGroup))

    #perform migration
    groupMat[cellType, grpIDSource] -= 1
    groupMat[cellType, grpIDTarget] += 1

    # track if any groups die in process
    groupDeathID = int(-1)  # -1 is no death

    # kill group if last cell died
    # use two stage check for increased speed
    if groupMat[cellType, grpIDSource] == 0:  # killed last of type
        #NINGroup = oneVecType @ groupMat[:, eventGroup]
        NINGroup = groupMat[:, grpIDSource].sum()
        if NINGroup == 0:  # all other types are zero too
            groupDeathID = int(grpIDSource)

    return groupDeathID


"""============================================================================
Sub functions group dynamics 
============================================================================"""

# remove group from group matrix
@jit(Tuple((f8[:, ::1], i8))(f8[:, ::1], i8), nopython=True)
def remove_group(groupMat, groupDeathID):
    # Note: groupMat is re-created, it has to be returned
    # create helper vector
    NGroup = groupMat.shape[1]
    hasDied = np.zeros(NGroup)
    hasDied[groupDeathID] = 1

    # copy remaining groups to new matrix
    groupMat = groupMat[:, hasDied == 0]

    NGroup -= 1

    return (groupMat, NGroup)


# calculate fission and extinction rate of all groups
@jit(void(f8[::1], f8[:, ::1], f8[::1], f8, i8, f8, f8, f8, f8, f8, f8, f8), nopython=True)
def calc_group_rates(groupRates, groupMat, grSizeVec, NTot, NGroup, 
    gr_CFis, gr_SFis, K_grp, K_tot,
    delta_grp, delta_tot, delta_size):

    # calc fission rate
    fissionR = grSizeVec * gr_SFis + gr_CFis
  
    # calc extinction rate
    if delta_grp != 0:
        groupDep = (NGroup / K_grp) ** delta_grp
    else:
        groupDep = 1
        
    if delta_tot != 0:
        popDep = (NTot / K_tot) ** delta_tot
    else:
        popDep = 1    

    if delta_size != 0:
        sizeEffect = (1/grSizeVec) ** delta_size
        #sizeEffect = sizeEffect /  (oneVecGrp @ sizeEffect)
        extinctR = groupDep * popDep * sizeEffect
    else:
        extinctR = groupDep * popDep * np.ones(NGroup)

    # combine all rates in single vector
    groupRates[0:NGroup] = fissionR
    groupRates[NGroup::] = extinctR

    return None


@jit(Tuple((f8[::1],f8[:, ::1]))(f8[::1], f8, f8), nopython=True)
def fission_group(parentGroup, offspr_size, offspr_frac):        
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
    NGroup = groupMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(groupRates, rand[0])

    # get event type
    eveNType = math.floor(eventID/NGroup)
    # get event group
    eventGroup = eventID % NGroup  # % is modulo operator

    if eveNType < 1:
        # fission event - add new group and split cells
        # get parent composition
        parentGroup = groupMat[:, eventGroup].copy()

        #perform fission process
        if offspr_size > 0:
            parrentNew, offspring = fission_group(parentGroup, offspr_size, offspr_frac)

            # only add daughter if not empty
            if offspring.sum() > 0:
                # update parrent
                groupMat[:, eventGroup] = parrentNew
                # add new daughter group
                groupMat = np.column_stack((groupMat, offspring))
    
            NGroup = groupMat.shape[1]

    else:
        # extinction event - remove group
        groupMat, NGroup = remove_group(groupMat, eventGroup)

    return (groupMat, NGroup)


# create helper vectors for dot products
def create_helper_vector(NGroup, NType):
    oneVecGrp = np.ones(NGroup)
    oneVecIndvR = np.ones(NGroup * NType * 4)
    oneVecGrR = np.ones(NGroup * 2)
    #init rates matrix
    indvRates = np.ones(4 * NType * NGroup)
    groupRates = np.ones(2 * NGroup)

    return(oneVecGrp, oneVecIndvR, oneVecGrR, indvRates, groupRates)

#calc group properties


# calc total number of individuals per group, use matrix product for speed
@jit(Tuple((f8[::1], f8))(f8[:, ::1], f8[::1], f8[::1]), nopython=True)
def calc_group_state(groupMat, oneVecType, oneVecGrp):
    #vector with size of each group
    grSizeVec = oneVecType @ groupMat
    #float total number of individuals
    NTot = oneVecGrp @ grSizeVec

    return(grSizeVec, NTot)


def calc_time_steps(model_par):
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

    return (maxT, minTRun, sampleInt, mavInt, rmsInt)


def adjust_indv_rates(model_par):

    NType = int(model_par['indv_NType'])
    indv_asymmetry = float(model_par['indv_asymmetry'])
    indv_cost = float(model_par['indv_cost'])
    indv_K = float(model_par['indv_K'])

    #calc birth rates to keep constant EQ group size when all cells are cooperators
    indvBirthVecCoop = 1 / (indv_asymmetry ** (np.arange(NType)))
    indvBirthVecCoop *= (indvBirthVecCoop.sum() ** (NType - 1)) / np.prod(indvBirthVecCoop)
    #include costs and calc birth rates for cooperators and defectors respectively
    indvBirthVec = np.kron(indvBirthVecCoop, np.array([(1 - indv_cost), 1]))
    #convert caryinf capacity to death rate
    indv_deathR = 1 / indv_K

    oneVecType = np.ones(2 * NType)

    return (oneVecType, indvBirthVec, indv_deathR)

"""============================================================================
Main model code
============================================================================"""

# main model
def run_model(model_par):

    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]


    # Initialize model, get rates and init matrices
    maxT, minTRun, sampleInt, mavInt, rmsInt = calc_time_steps(model_par)
                
    # initialize group matrix
    groupMat = init_groupMat(model_par)

    # get matrix with random numbers
    # creates matrix with maxRandMatSize entries, it is recreated if needed
    maxRandMatSize = int(1E6)
    randMat = util.create_randMat(maxRandMatSize, 5)

    # initialize output matrix
    output, distFCoop, binFCoop, distGrSize, binGrSize = init_output_matrix(model_par)
  
    # helper vector to calc sum over all groups
    NGroup = groupMat.shape[1]
    
    # get individual rates
    indv_mutR, delta_indv, inv_migrR = [float(model_par[x])
        for x in ('indv_mutR', 'delta_indv','indv_migrR')]
    NType = int(model_par['indv_NType'])
    
    indv_K     = float(model_par['indv_K'])
    
    #get group fission rates
    gr_CFis, gr_SFis = [float(model_par[x])
        for x in ('gr_Cfission','gr_Sfission')] 
    #get group death rates 
    K_grp, K_tot, delta_grp, delta_tot, delta_size = [float(model_par[x])
        for x in ('K_grp', 'K_tot', 'delta_grp', 'delta_tot', 'delta_size')]
    #get group reproduction traits
    offspr_size, offspr_frac = [float(model_par[x])
        for x in ('offspr_size', 'offspr_frac')]
    
    #adjust fission for K
    gr_SFis /= indv_K
    
    
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

    #init static helper vectors
    oneVecType, indvBirthVec, indv_deathR = adjust_indv_rates(model_par)

    #init dynamic helper vectors
    oneVecGrp, oneVecIndvR, oneVecGrR, indvRates, groupRates = \
        create_helper_vector(NGroup, NType)

    # get first sample of init state
    sampleIdx = sample_model(groupMat, output, distFCoop, binFCoop,
                             distGrSize, binGrSize, sampleIdx, currT, mavInt, rmsInt, stateVarPlus)

    # loop time steps
    while currT <= maxT:

        # reset rand matrix when used up
        if ttR >= maxRandMatSize:
            randMat = util.create_randMat(maxRandMatSize, 5)
            ttR = 0

        #calc group state
        grSizeVec, NTot = calc_group_state(groupMat, oneVecType, oneVecGrp)


        # calc rates of individual level events
        calc_indv_rates(indvRates, groupMat, grSizeVec, indvBirthVec,
                        indv_deathR, delta_indv,
                        NType, NGroup)
        
        
        # calc rates of group events
        calc_group_rates(groupRates, groupMat, grSizeVec, NTot, NGroup,
                        gr_CFis, gr_SFis, K_grp, K_tot,
                        delta_grp, delta_tot, delta_size)

        # calculate total propensities
        indvProp = oneVecIndvR @ indvRates
        groupProp = oneVecGrR @ groupRates
        migrProp = inv_migrR * NTot
        totProp = indvProp + groupProp + migrProp

        # calc time step
        dt = -1 * math.log(randMat[ttR, 1]) / totProp

        # select group or individual event
        rescaledRand = randMat[ttR, 0] * totProp
        groupsHaveChanged = False
        if rescaledRand < indvProp:
            # individual level event - select and process individual level event
            groupDeathID = process_indv_event(
                groupMat, indvRates, indv_mutR, randMat[ttR, 2:4], NType, NGroup)

            if groupDeathID > -1:  # remove empty group
                groupMat, NGroup = remove_group(groupMat, groupDeathID)
                groupsHaveChanged = True

        elif rescaledRand < (indvProp + migrProp):
            # migration event - select and process migration event
            groupDeathID = process_migration_event(
                groupMat, grSizeVec, NGroup, NType, randMat[ttR, 2:5])
            
            if groupDeathID > -1:  # remove empty group
                groupMat, NGroup = remove_group(groupMat, groupDeathID)
                groupsHaveChanged = True

        else:
            # group level event - select and process group level event
            groupMat, NGroup = process_group_event(
                groupMat, groupRates, randMat[ttR, 2:4], offspr_size, offspr_frac)
            groupsHaveChanged = True

        if groupsHaveChanged:
            if NGroup == 0:  # if all groups have died, end simulation
                sampleIdx = sample_extinction(output, distFCoop, binFCoop,
                                              distGrSize, sampleIdx, currT, stateVarPlus)
                break
            else: #otherwise, recreate helper vectors
                oneVecGrp, oneVecIndvR, oneVecGrR, indvRates, groupRates = \
                    create_helper_vector(NGroup, NType)

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
                
            # check if population size remains in bounds
            if output['NTot'][sampleIdx - 1] > model_par['maxPopSize']:
                sample_nan(groupMat, output, distFCoop, binFCoop,
                                     distGrSize, binGrSize, sampleIdx - 1, currT, mavInt, rmsInt, stateVarPlus)
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
    
    start = time.time()
    output, distFCoop, distGrSize = run_model(model_par)    
    end = time.time()

 
    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutR','indv_migrR', 'indv_asymmetry', 'delta_indv',
               'gr_Sfission', 'gr_Cfission', 'K_grp', 'K_tot', 'delta_grp', 'delta_tot', 'delta_size',
               'offspr_size','offspr_frac']
                                                        
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVarPlus]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVarPlus]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3 +[('run_time', 'f8')]
    dType = np.dtype(dTypeList)

    output_matrix = np.zeros(1, dType)

    # store final state
    for var in stateVarPlus:
        output_matrix[var] = output[var][-1]
        var_mav = var + '_mav'
        output_matrix[var_mav] = output[var_mav][-1]

    for par in parList:
        output_matrix[par] = model_par[par]
        
    output_matrix['run_time'] = end - start   

    endDistFCoop = distFCoop[-1,:]
    endDistGrSize = distGrSize[-1, :]

    return (output_matrix, endDistFCoop, endDistGrSize)


# this piece of code is run only when this script is executed as the main
if __name__ == "__main__":
    print("running with default parameter")

    model_par = {
        #        #time and run settings
#        "maxT":             400,  # total run time
#        "minT":             250,    # min run time
#        "sampleInt":        1,      # sampling interval
#        "mav_window":       400,    # average over this time window
#        "rms_window":       400,    # calc rms change over this time window
#        "rms_err_trNCoop":  1E-1,   # when to stop calculations
#        "rms_err_trNGr":    5E-1,   # when to stop calculations
         #time and run settings
        "maxT":             50,  # total run time
        "maxPopSize":       10000,  #stop simulation if population exceeds this number
        "minT":             1,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       10,    # average over this time window
        "rms_window":       10,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-51,   # when to stop calculations
        "rms_err_trNGr":    5E-51,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    10,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   10,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,  # cost of cooperation
        "indv_mutR":   1E-3,   # mutation rate to cheaters
        "indv_migrR":   0,   # mutation rate to cheaters
        # group size control
        "indv_K":           50,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_Cfission':      1/100,
        'gr_Sfission':      1/50,
        # extinction rate
        'delta_grp':      0,      # exponent of denisty dependence on group #
        'K_grp':          0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            30000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac': 0.8  # offspr_size < offspr_frac < 1-offspr_size'
    }
    output, distFCoop, distGrSize = run_model(model_par)
