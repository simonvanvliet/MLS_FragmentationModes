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

from numba.types import Tuple, UniTuple
from numba import jit, f8, i8
import math
import numpy as np
import MlsGroupDynamics_utilities as util
import MlsGroupDynamics_main as mls
import time


"""============================================================================
GLOBAL Constants
============================================================================"""
#outputMat variables to store
stateVar = ['NTot', 'NCoop', 'fCoop',
            'NGroup', 'groupSizeAv', 'groupSizeMed', 
            'offspr_size','offspr_frac']

#setup bins and vectors for group traits
nBinOffsprSize = 100
nBinOffsprFrac = 100    

binsOffsprSize = np.linspace(0, 0.5, nBinOffsprSize+1)
binsOffsprFrac = np.linspace(0, 1, nBinOffsprFrac+1)

binCenterOffsprSize = (binsOffsprSize[1::]+binsOffsprSize[0:-1])/2
binCenterOffsprFrac = (binsOffsprFrac[1::]+binsOffsprFrac[0:-1])/2

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
                    

"""============================================================================
Init functions 
============================================================================"""

# initialize outputMat matrix
def init_outputMat_matrix(model_par):
    sampleInt = model_par['sampleInt']
    maxT = model_par['maxT']
    numTSample = int(np.ceil(maxT / sampleInt) + 1)
    
    addVar = ['rms_err_NCoop', 'rms_err_NGroup', 'time']
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
                ['N%imut' % x for x in range(model_par['indv_NType'])]

    # init outputMat matrix
    dTypeList1 = [(x, 'f8') for x in stateVarPlus]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVarPlus]
    dTypeList3 = [(x, 'f8') for x in addVar]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3
    dType = np.dtype(dTypeList)

    # initialize outputMats to NaN
    outputMat = np.full(numTSample, np.nan, dType)
    outputMat['time'][0] = 0

    # init matrix to track distribution replication strategies 
    traitDistr = np.full((numTSample, nBinOffsprFrac, nBinOffsprSize), np.nan)

    return (outputMat, traitDistr)


# initialize group matrix
# each column is a group and lists number of [A,A',B,B'] cells
def init_grpMat(model_par):
    #get properties
    NGroup = int(model_par["init_groupNum"])
    NType = int(model_par['indv_NType'])

    #get group reproduction traits
    offspr_size, offspr_frac = [float(model_par[x])
                                for x in ('offspr_sizeInit', 'offspr_fracInit')]

    #check rates
    if offspr_size > 0.5:
        print('cannot do that: offspr_size < 0.5 and offspr_size < offspr_frac < 1')
        raise ValueError
    elif offspr_frac < offspr_size or offspr_frac > (1-offspr_size):
        print('cannot do that: offspr_frac should be offspr_size < offspr_frac < 1-offspr_size')
        raise ValueError

    #discretize traits
    offspr_size_idx = min(nBinOffsprSize, round(nBinOffsprSize * offspr_size / 0.5))
    offspr_frac_idx = min(nBinOffsprFrac, round(nBinOffsprFrac * offspr_frac / 1))

    #create group composition vector
    nCoop = round(model_par["init_groupDens"] * model_par['init_fCoop'] / model_par['indv_NType'])
    nDef  = round(model_par["init_groupDens"] * (1 - model_par['init_fCoop']) / model_par['indv_NType'])
    
    # init all groups with zero
    grpMat = np.zeros((NType * 2, NGroup, nBinOffsprFrac, nBinOffsprSize), order='C')
    grpMat2D = np.zeros((NType * 2, NGroup), order='C')

    # set group prop
    grpMat[0::2, :, offspr_frac_idx, offspr_size_idx] = nCoop
    grpMat[1::2, :, offspr_frac_idx, offspr_size_idx] = nDef
    
    grpMat2D[0::2, :] = nCoop
    grpMat2D[1::2, :] = nDef

    return (grpMat, grpMat2D)


"""============================================================================
Sample model code 
============================================================================"""
@jit(Tuple((f8[:, ::1], f8[:, ::1], f8, f8))(f8[:, :, :, ::1]), nopython=True)
def summarize_grpMat(grpMat):
    #matrix with number per type
    grpMat2D = grpMat.sum(axis=3).sum(axis=2) 
    traitDistr = grpMat.sum(axis=0).sum(axis=0) 

    cellNumTot = traitDistr.sum()

    marginalSize = traitDistr.sum(axis=0) / cellNumTot
    marginalFrac = traitDistr.sum(axis=1) / cellNumTot
    av_size = np.sum(binCenterOffsprSize * marginalSize)
    av_frac = np.sum(binCenterOffsprFrac * marginalFrac)

    return (grpMat2D, traitDistr, av_size, av_frac)

# sample model
def sample_model(grpMatrix, outputMat, traitDistr, 
    sample_idx, currT, mavInt, rmsInt, stateVarPlus):
    # store time
    outputMat['time'][sample_idx] = currT

    # calc number of groups
    shapegrpMat = grpMatrix.shape
    NGroup = shapegrpMat[1]
    NType = int(shapegrpMat[0] / 2)

    # summarize groups
    grpMat2D, traitDistrCurr, av_size, av_frac = summarize_grpMat(grpMatrix)

    # get group statistics
    NTot, NCoop, groupSizeAv, groupSizeMed, NTot_type, fCoop_group, grSizeVec = mls.calc_cell_stat(
        grpMat2D)

    # calc total population sizes
    for tt in range(NType):
        outputMat['N%i' %tt][sample_idx] = NTot_type[tt*2]
        outputMat['N%imut' %tt][sample_idx] = NTot_type[tt*2+1]
       
    outputMat['NTot'][sample_idx] = NTot
    outputMat['NCoop'][sample_idx] = NCoop
    outputMat['fCoop'][sample_idx] = NCoop / NTot
    
    outputMat['NGroup'][sample_idx] = NGroup
    outputMat['groupSizeAv'][sample_idx] = groupSizeAv
    outputMat['groupSizeMed'][sample_idx] = groupSizeMed

    outputMat['offspr_size'][sample_idx] = av_size
    outputMat['offspr_frac'][sample_idx] = av_frac

    #calc moving average 
    if sample_idx >= 1:
        for varname in stateVarPlus:
            outname = varname + '_mav'
            mav, _ = util.calc_moving_av(
                outputMat[varname], sample_idx, mavInt)
            outputMat[outname][sample_idx] = mav

    # calc rms error
    if sample_idx >= rmsInt:
        outputMat['rms_err_NCoop'][sample_idx] = util.calc_rms_error(
            outputMat['NCoop_mav'], sample_idx, rmsInt) / outputMat['NCoop_mav'][sample_idx]
        outputMat['rms_err_NGroup'][sample_idx] = util.calc_rms_error(
            outputMat['NGroup_mav'], sample_idx, rmsInt) / outputMat['NGroup_mav'][sample_idx]

    # store distribution of traits
    traitDistr[sample_idx, :, :] = traitDistrCurr / NTot

    sample_idx += 1
    return sample_idx

# sample model
def sample_nan(grpMatrix, outputMat, traitDistr, 
    sample_idx, currT, mavInt, rmsInt, stateVarPlus):
    # store time
    outputMat['time'][sample_idx] = currT

    # calc total population sizes
    for varname in stateVarPlus:
        outname = varname + '_mav'
        outputMat[varname][sample_idx] = np.nan
        outputMat[outname][sample_idx] = np.nan
        
    outputMat['rms_err_NCoop'][sample_idx] = np.nan
    outputMat['rms_err_NGroup'][sample_idx] = np.nan

    traitDistr[sample_idx, :, :] = np.nan

    return None

# sample model
def sample_extinction(outputMat, traitDistr, sample_idx, currT, stateVarPlus):
    # store time
    outputMat['time'][sample_idx] = currT

    # calc total population sizes
    for varname in stateVarPlus:
        outname = varname + '_mav'
        outputMat[varname][sample_idx] = 0
        outputMat[outname][sample_idx] = 0
        
    outputMat['rms_err_NCoop'][sample_idx] = 0
    outputMat['rms_err_NGroup'][sample_idx] = 0

    # calc distribution groupsizes
    traitDistr[sample_idx, :, :] = 0
    sample_idx += 1

    return sample_idx

"""============================================================================
Sub functions individual dynamics 
============================================================================"""

@jit(UniTuple(i8,2)(i8, i8), nopython=True)
def keep_traits_in_bounds(offsprSizeIdx, offsprFracIdx):
    #make sure offspring fraction remains within bounds
    #evaluate at left boundary, as always at max there
    offsprSize = binsOffsprSize[offsprSizeIdx]
    offsprFracUp = binsOffsprFrac[offsprFracIdx+1]
    offsprFracLow = binsOffsprFrac[offsprFracIdx]
    #decrease or increase offsprFracIdx till within bounds
    if offsprFracLow > (1 - offsprSize):
        idx = np.arange(nBinOffsprFrac)
        withinBounds = binsOffsprFrac[:-1] < (1 - offsprSize)
        offsprFracIdx = np.max(idx[withinBounds])
        
    elif offsprFracUp < offsprSize:
        idx = np.arange(nBinOffsprFrac)
        withinBounds = binsOffsprFrac[1:] > offsprSize
        offsprFracIdx = np.min(idx[withinBounds])
    
    return (offsprSizeIdx, offsprFracIdx)

# process individual level events
@jit(i8(f8[:, :, :, ::1], f8[:, ::1], f8[::1], f8[::1], i8, i8, f8, f8, f8), nopython=True)
def process_indv_event(grpMat, grpMat2D, indvRate, rand, 
                       NType, NGroup, mutR_type, mutR_size, mutR_frac):
    # Note: grpMat is updated in place, it does not need to be returned 
    NTypeWMut = NType*2
    
    # select random event based on propensity
    eventID = util.select_random_event(indvRate, rand[0])

    # get event type
    eventType = math.floor(eventID / NGroup)
    # get event group
    grpIdx = eventID % NGroup  # % is modulo operator
    typeIdx = eventType % NTypeWMut

    #find trait of affected cell
    cellID = util.select_random_event_2D(grpMat[typeIdx, grpIdx, :, :], rand[1])
    sizeIdx = math.floor(cellID / nBinOffsprFrac)
    fracIdx = cellID % nBinOffsprFrac

    # track if any groups die in process
    groupDeathID = -1  # -1 is no death

    if eventType < NTypeWMut:  # birth event
        # check for mutation in cell type
        isWT = (typeIdx % 2) == 0  # Wild type cell
        typeMutates = rand[2] < mutR_type  # mutates to other type
        offsprTypeIdx = typeIdx + 1 if (isWT and typeMutates) else typeIdx

        #check for mutation in offspring size
        if rand[3] < mutR_size / 2:  # offspring size mutates to lower value
            offsprSizeIdx = max(0, sizeIdx - 1)
        elif rand[3] < mutR_size:  # offspring size mutates to lower value
            offsprSizeIdx = min(nBinOffsprSize - 1, sizeIdx + 1)
        else:
            offsprSizeIdx = sizeIdx

        #check for mutation in offspring fraction
        if rand[4] < mutR_frac / 2:  # offspring size mutates to lower value
            offsprFracIdx = max(0, fracIdx - 1)
        elif rand[4] < mutR_frac:  # offspring size mutates to lower value
            offsprFracIdx = min(nBinOffsprFrac - 1, fracIdx + 1)
        else:
            offsprFracIdx = fracIdx

        #make sure we stay inside allowed trait space
        #offsprSizeIdx, offsprFracIdx = keep_traits_in_bounds(offsprSizeIdx, offsprFracIdx)
        offsprFracIdx = offsprFracMatrix[offsprFracIdx, offsprSizeIdx]
        
        # place new offspring
        grpMat[offsprTypeIdx, grpIdx, offsprSizeIdx, offsprFracIdx] += 1
        grpMat2D[offsprTypeIdx, grpIdx] += 1

    else:  # death event
        # remove cell from group
        grpMat[typeIdx, grpIdx, sizeIdx, fracIdx] -= 1
        grpMat2D[typeIdx, grpIdx] -= 1

        # kill group if last cell died
        # use two stage check for increased speed
        if grpMat2D[typeIdx, grpIdx] == 0:  # killed last of type
            #NINGroup = onesNType @ grpMat[:, eventGroup]
            NINGroup = grpMat2D[:, grpIdx].sum()
            if NINGroup == 0:  # all other types are zero too
                groupDeathID = int(grpIdx)

    return groupDeathID


"""============================================================================
Sub functions migration dynamics 
============================================================================"""

# process migration event
@jit(i8(f8[:, :, :, ::1], f8[:, ::1], f8[::1], i8, i8, f8[::1]), nopython=True)
def process_migration_event(grpMat, grpMat2D, grSizeVec, NGroup, NType, rand):
    # Note: grpMat is updated in place, it does not need to be returned

    # select random group of origin based on size
    grpIDSource = util.select_random_event(grSizeVec, rand[0])

    # select random type of migrant based on population size
    typeIdx = util.select_random_event(grpMat2D[:, grpIDSource], rand[1])
        
    # find trait of affected cell
    cellID = util.select_random_event_2D(grpMat[typeIdx, grpIDSource, :, :], rand[2])
    sizeIdx = math.floor(cellID / nBinOffsprFrac)
    fracIdx = cellID % nBinOffsprFrac

    # select random target group
    grpIDTarget = int(np.floor(rand[3] * NGroup))

    #perform migration
    grpMat[typeIdx, grpIDSource, sizeIdx, fracIdx] -= 1
    grpMat[typeIdx, grpIDTarget, sizeIdx, fracIdx] += 1
    
    grpMat2D[typeIdx, grpIDSource] -= 1
    grpMat2D[typeIdx, grpIDTarget] += 1

    # track if any groups die in process
    groupDeathID = int(-1)  # -1 is no death

    # kill group if last cell died
    # use two stage check for increased speed
    if grpMat2D[typeIdx, grpIDSource] == 0:  # killed last of type
        #NINGroup = onesNType @ grpMat[:, eventGroup]
        NINGroup = grpMat2D[:, grpIDSource].sum()
        if NINGroup == 0:  # all other types are zero too
            groupDeathID = int(grpIDSource)

    return groupDeathID


"""============================================================================
Sub functions group dynamics 
============================================================================"""

# remove group from group matrix
@jit(Tuple((f8[:, :, :, ::1], f8[:, ::1]))(f8[:, :, :, ::1], f8[:, ::1], i8), nopython=True)
def remove_group(grpMat, grpMat2D, groupDeathID):
    # Note: grpMat is re-created, it has to be returned
    # create helper vector
    grShape = grpMat.shape    
    # copy remaining groups to new matrix
    newShape = (grShape[0], grShape[1]-1, grShape[2], grShape[3])
    grpMatNew = np.zeros(newShape)
    grpMatNew[:,:groupDeathID,:,:] = grpMat[:,:groupDeathID,:,:]
    grpMatNew[:,groupDeathID::,:,:] = grpMat[:,groupDeathID+1::,:,:]

    #create new 2D matrix
    grpMat2DNew = grpMatNew.sum(axis=3).sum(axis=2)
    
    return (grpMatNew, grpMat2DNew)

@jit(Tuple((f8, f8, f8))(f8[:, :, ::1]), nopython=True)
def calc_mean_group_prop(parentGroup):
    #parent group: type / frac  / size     
    #number of cells in parents
    parTraitMatrix = parentGroup.sum(axis=0)
    cellNumPar = parTraitMatrix.sum()

    marginalSize = parTraitMatrix.sum(axis=0) / cellNumPar
    marginalFrac = parTraitMatrix.sum(axis=1) / cellNumPar
    offspr_size = np.sum(binCenterOffsprSize * marginalSize)
    offspr_frac = np.sum(binCenterOffsprFrac * marginalFrac)
    
    return(offspr_size, offspr_frac, cellNumPar)

@jit(Tuple((f8[:, :, ::1],f8[:, :, :, ::1]))(f8[:, :, ::1]), nopython=True)
def fission_group(parentGroup):   
    #get group properties
    offspr_size, offspr_frac, cellNumPar = calc_mean_group_prop(parentGroup)
   
    #calc number of offspring, draw from Poisson distribution
    # <#offspring> = numCellsToOffspr / sizeOfOffspr
    # = offspr_frac * cellNumPar / offspr_size * cellNumPar
    # = offspr_frac / offspr_size
    expectedNumOffSpr = offspr_frac / offspr_size
    numOffSpr = int(np.random.poisson(expectedNumOffSpr))
    #calc total number of cells passed on to offspring, keep at least 1 cell in parent
    #numCellPerOffspr = round(offspr_size * cellNumPar)
    #numCellsToOffspr = int(min(numCellPerOffspr*numOffSpr, cellNumPar-1))
    numCellsToOffspr = int(min(round(offspr_frac * cellNumPar), cellNumPar-1))

    #assign cells to offspring
    if numOffSpr > 0:
        parrentPool = parentGroup
        matShape = parentGroup.shape

        #init offspring array
        offspring = np.zeros((matShape[0], numOffSpr, matShape[1], matShape[2]))

        #perform random sampling
        randMat = util.create_randMat(numCellsToOffspr, 1)

        for ii in range(numCellsToOffspr):
            #randomly pick cell from parent using weighted lottery
            typePicked = util.select_random_event_3D(parrentPool, randMat[ii, 0])
            idxPick = util.flat_to_3d_index(typePicked, matShape)
            #idxPick = np.unravel_index(typePicked, matShape)

            #deal round the table: select offsping to assign cell to
            offspringPicked = ii % numOffSpr 
            #assign cell to offspring
            offspring[idxPick[0], offspringPicked, idxPick[1], idxPick[2]] += 1
            #remove cell from parent
            parrentPool[idxPick] -= 1
        
        #remove empty daughter groups
        numCellInOffspr = offspring.sum(axis=3).sum(axis=2).sum(axis=0)
        numNonEmptyOffspring = int(np.sum(numCellInOffspr>0))
        
        if numNonEmptyOffspring < numOffSpr:
            offspringNew = np.zeros((matShape[0], numNonEmptyOffspring, matShape[1], matShape[2]))
            idx = 0
            for i in range(numOffSpr):
                if numCellInOffspr[i] > 0:
                    offspringNew[:, idx, :, :] = offspring[:, i, :, :]
                    idx += 1
            offspring = offspringNew        
        
        #update parent to new state
        parrentNew = parrentPool
    else:
        #nothing happens
        parrentNew = parentGroup
        offspring = np.zeros((0, 0, 0, 0))
            
    return (parrentNew, offspring)


# process individual level events
@jit(Tuple((f8[:, :, :, ::1], f8[:, ::1]))(f8[:, :, :, ::1], f8[:, ::1], f8[::1], f8[::1]), nopython=True)
def process_group_event(grpMat, grpMat2D, grpRate, rand):
    # get number of groups
    NGroup = grpMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(grpRate, rand[0])

    # get event type
    eveNType = math.floor(eventID/NGroup)
    # get event group
    eventGroup = eventID % NGroup  # % is modulo operator

    if eveNType < 1:
        # fission event - add new group and split cells
        # get parent composition
        parentGroup = grpMat[:, eventGroup, :, :].copy()

        #perform fission process
        parrentNew, offspring = fission_group(parentGroup)

        # only add daughter if not empty
        if offspring.sum() > 0:
            numOffspring = offspring.shape[1]
            # update parrent
            grpMat[:, eventGroup, :, :] = parrentNew
            # add new daughter group
            newShape = (grpMat.shape[0], grpMat.shape[1]+numOffspring, grpMat.shape[2], grpMat.shape[3])
            grpMatNew = np.zeros(newShape)
            grpMatNew[:,:-numOffspring,:,:] = grpMat
            grpMatNew[:,-numOffspring::,:,:] = offspring
                        
            grpMat2DNew = grpMatNew.sum(axis=3).sum(axis=2)
        else:
            grpMatNew = grpMat
            grpMat2DNew = grpMat2D

    else:
        # extinction event - remove group
        grpMatNew, grpMat2DNew = remove_group(grpMat, grpMat2D, eventGroup)
    
    return (grpMatNew, grpMat2DNew)


#calc group properties
# calc total number of individuals per group, use matrix product for speed
@jit(Tuple((f8[:, ::1], f8[::1], f8))(f8[:, :, :, ::1], f8[::1], f8[::1]), nopython=True)
def calc_group_state(grpMat, onesNType, onesNGrp):
    #matrix with number per type
    grpMat2D = grpMat.sum(axis=3).sum(axis=2)
    #vector with size of each group
    grSizeVec = onesNType @ grpMat2D
    #float total number of individuals
    NTot = onesNGrp @ grSizeVec

    return(grpMat2D, grSizeVec, NTot)



"""============================================================================
Main model code
============================================================================"""

# main model
def run_model(model_par):
    
    #create state variables
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]
                
    # get individual rates
    delta_indv = float(model_par['delta_indv'])
    indv_K     = float(model_par['indv_K'])
    inv_migrR  = float(model_par['indv_migrR'])
    NType      = int(model_par['indv_NType'])
    mutR_type  = float(model_par['mutR_type'])
    mutR_size  = float(model_par['mutR_size'])
    mutR_frac  = float(model_par['mutR_frac'])
    # get group rates
    gr_CFis    = float(model_par['gr_Cfission'])
    gr_SFis    = float(model_par['gr_Sfission']) / indv_K
    K_grp      = float(model_par['K_grp'])
    K_tot      = float(model_par['K_tot'])
    delta_grp  = float(model_par['delta_grp'])
    delta_tot  = float(model_par['delta_tot'])
    delta_size = float(model_par['delta_size'])
    
    # Initialize model, get rates and init matrices
    maxT, minT, sampleInt, mavInt, rmsInt = mls.calc_time_steps(model_par)
        
    # init counters
    currT = 0
    ttR = 0
    sampleIdx = 0
    
    # get matrix with random numbers
    rndSize1 = 7
    rndSize0 = int(1E6)
    randMat = util.create_randMat(rndSize0, rndSize1)
    
    # initialize outputMat matrix
    outputMat, traitDistr = init_outputMat_matrix(model_par)

    #init static helper vectors
    onesNType, birthRVec, deathR = mls.adjust_indv_rates(model_par)
    
    # initialize group matrix
    grpMat, grpMat2D = init_grpMat(model_par)
    NGroup = grpMat.shape[1]

    #init dynamic helper vectors
    onesNGrp, onesIndR, onesGrR, indvRate, grpRate = mls.create_helper_vector(
        NGroup, NType)

    # get first sample of init state
    sampleIdx = sample_model(grpMat, outputMat, traitDistr, 
        sampleIdx, currT, mavInt, rmsInt, stateVarPlus)

    # loop time steps
    while currT <= maxT:
        # reset rand matrix when used up
        if ttR >= rndSize0:
            randMat = util.create_randMat(rndSize0, rndSize1)
            ttR = 0 

        #calc group state
        grSizeVec, NTot = mls.calc_group_state(grpMat2D, 
                                               onesNType, onesNGrp)    

        # calc rates of individual level events
        mls.calc_indv_rates(indvRate, grpMat2D, grSizeVec, birthRVec,
                            deathR, delta_indv, NType, NGroup)
        
        
        # calc rates of group events
        mls.calc_group_rates(grpRate, grpMat2D, grSizeVec, NTot, NGroup,
                            gr_CFis, gr_SFis, K_grp, K_tot,
                            delta_grp, delta_tot, delta_size)

        # calculate total propensities
        indvProp = onesIndR @ indvRate
        grpProp = onesGrR @ grpRate
        migrProp = inv_migrR * NTot
        totProp = indvProp + grpProp + migrProp

        # calc time step
        dt = -1 * math.log(randMat[ttR, 1]) / totProp

        # select group or individual event
        rescaledRand = randMat[ttR, 0] * totProp
        groupsHaveChanged = False
        if rescaledRand < indvProp:
            # individual level event - select and process individual level event
            groupDeathID = process_indv_event(grpMat, grpMat2D, indvRate, 
                                              randMat[ttR, 2:7], NType, NGroup, 
                                              mutR_type, mutR_size, mutR_frac)
            if groupDeathID > -1:  # remove empty group
                grpMat, grpMat2D = remove_group(grpMat, grpMat2D, groupDeathID)
                groupsHaveChanged = True
        elif rescaledRand < (indvProp + migrProp):
            # migration event - select and process migration event
            groupDeathID = process_migration_event(grpMat, grpMat2D, grSizeVec, 
                                                   NGroup, NType, randMat[ttR, 2:6])
            if groupDeathID > -1:  # remove empty group
                grpMat, grpMat2D = remove_group(grpMat, grpMat2D, groupDeathID)
                groupsHaveChanged = True
        else:
            # group level event - select and process group level event
            grpMat, grpMat2D = process_group_event(grpMat, grpMat2D, grpRate, randMat[ttR, 2:4])
            groupsHaveChanged = True
         
        # update group matrices if needed    
        if groupsHaveChanged:
            NGroup = grpMat2D.shape[1]
            if NGroup > 0:  #update group matrices
                onesNGrp, onesIndR, onesGrR, indvRate, grpRate = mls.create_helper_vector(
                    NGroup, NType) 
            else: #otherwise, if all groups have died, end simulation
                sampleIdx = sample_extinction(outputMat, traitDistr, sampleIdx, currT, stateVarPlus)
                print('System has gone extinct')
                break

        # update time
        currT += dt
        ttR += 1
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = sample_model(grpMat, outputMat, traitDistr, sampleIdx, 
                currT, mavInt, rmsInt, stateVarPlus)
            if currT % 10 == 0:
                print('current time = %i' % currT)
            
    # cut off non existing time points at end
    outputMat = outputMat[0:sampleIdx]
    traitDistr = traitDistr[0:sampleIdx, :, :]
    
    if outputMat['NCoop'][-1] == 0:
        outputMat['NCoop_mav'][-1] = 0
    
    return (outputMat, traitDistr)


"""============================================================================
Code that calls model and plots results
============================================================================"""

# code to plot data
# set type to "lin" or "log" to switch between lin or log plot


def single_run_save(model_par, mainName):
    parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_Cfission'   : 'fisC',
                'gr_Sfission'   : 'fisS',
                'indv_NType'    : 'nTyp', 
                'indv_asymmetry': 'asym',
                'indv_cost'     : 'cost', 
                'mutR_type'     : 'muTy', 
                'mutR_size'     : 'muSi', 
                'mutR_frac'     : 'muFr', 
                'indv_migrR'    : 'migR', 
                'indv_K'        : 'kInd', 
                'K_grp'         : 'kGrp', 
                'K_tot'         : 'kTot',
                'offspr_sizeInit':'siIn',
                'offspr_fracInit':'frIn'}
    
    parListName = ['gr_Sfission', 'indv_cost', 'mutR_type',
                   'mutR_size', 'mutR_frac', 'offspr_sizeInit',
                   'offspr_fracInit', 'indv_K',
                   'indv_migrR']

    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    fileName = mainName + parName + '.npz'
    
    try: 
        outputMat, traitDistr = run_model(model_par)  
        np.savez(fileName, output=outputMat, traitDistr=traitDistr,
                 model_par=[model_par])
    except:
        print("error with run")
        traitDistr = np.full((1, nBinOffsprFrac, nBinOffsprSize), np.nan)
    
    return traitDistr
    
    

#run model store only final state 
def single_run_finalstate(model_par):
    # run model
    start = time.time()
    outputMat, traitDistr = run_model(model_par)    
    end = time.time()

    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_migrR', 'indv_asymmetry', 'delta_indv',
               'gr_Sfission', 'gr_Cfission', 'K_grp', 'K_tot', 'delta_grp', 'delta_tot', 'delta_size',
               'mutR_type', 'mutR_size','mutR_frac']
                                                        
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]

    # init outputMat matrix
    dTypeList1 = [(x, 'f8') for x in stateVarPlus]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVarPlus]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3 +[('run_time', 'f8')]
    dType = np.dtype(dTypeList)

    outputMat_matrix = np.zeros(1, dType)

    # store final state
    for var in stateVarPlus:
        outputMat_matrix[var] = outputMat[var][-1]
        var_mav = var + '_mav'
        outputMat_matrix[var_mav] = outputMat[var_mav][-1]

    for par in parList:
        outputMat_matrix[par] = model_par[par]
        
    outputMat_matrix['run_time'] = end - start  
     
    traitDistrEnd = traitDistr[-1, :, :]

    return (outputMat_matrix, traitDistrEnd)


# this piece of code is run only when this script is executed as the main
if __name__ == "__main__":
    print("running with default parameter")

    model_par = {
        #time and run settings
        "maxT":             10,  # total run time
        "maxPopSize":       0,  #stop simulation if population exceeds this number
        "minT":             10,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       5,    # average over this time window
        "rms_window":       5,    # calc rms change over this time window
        "rms_err_trNCoop":  0,   # when to stop calculations
        "rms_err_trNGr":    0,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    5,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   100,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,  # cost of cooperation
        "indv_migrR":       0,   # mutation rate to cheaters
        # set mutation rates
        'mutR_type':        1E-3,
        'mutR_size':        1E-1, 
        'mutR_frac':        1E-1, 
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_Cfission':      1/100,
        'gr_Sfission':      2,
        # extinction rate
        'delta_grp':        0,      # exponent of denisty dependence on group #
        'K_grp':            0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
        'K_tot':            30000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # initial settings for fissioning
        'offspr_sizeInit':  0.25,  # offspr_size <= 0.5 and
        'offspr_fracInit':  0.5  # offspr_size < offspr_frac < 1-offspr_size'
    }
    outputMat, traitDistr = run_model(model_par)
