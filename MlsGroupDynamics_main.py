#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 15 2019
Last Update March 28 2020

Implements main MLS model of group dynamics (no evolution)

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
            'NGrp', 'groupSizeAv', 'groupSizeMed','GrpDeaths','GrpBirths','GrpNetProd']


"""============================================================================
Init functions 
============================================================================"""


# initialize output matrix
def init_output_matrix(model_par):
    # get parameters
    sampleInt = model_par['sampleInt']
    maxT = model_par['maxT']
    numTSample = int(np.ceil(maxT / sampleInt) + 1)
    
    #create list of state variables to store
    addVar = ['rms_err_NCoop', 'rms_err_NGrp', 'time','GrpDeathsCumul','GrpBirthsCumul']
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
    NGrp = int(model_par["init_groupNum"])
    NType = int(model_par['indv_NType'])

    #create group composition vector
    init_groupComp = np.zeros((NType*2, 1))
    nCoop = round(model_par["init_groupDens"] * model_par['init_fCoop'] / model_par['indv_NType'])
    nDef  = round(model_par["init_groupDens"] * (1 - model_par['init_fCoop']) / model_par['indv_NType'])
    init_groupComp[0::2] = nCoop
    init_groupComp[1::2] = nDef

    # init all groups with same composition
    groupMat = init_groupComp @ np.ones((1, NGrp))
    # store in C-byte order
    groupMat = np.copy(groupMat, order='C')

    return groupMat


"""============================================================================
Sample model code 
============================================================================"""

#create distribution
@jit(f8[:](f8[:], f8[:]), nopython=True)
def calc_distri(dataVec, binEdges):
    NGrp = dataVec.size
    # get distribution of average cooperator fraction per host
    binCount, _ = np.histogram(dataVec, bins=binEdges)
    distribution = binCount / NGrp
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
                 distGrSize, binGrSize, sample_idx, currT, 
                 mavInt, rmsInt, 
                 stateVarPlus, NBGrp, NDGrp):
    # store time
    output['time'][sample_idx] = currT

    # calc number of groups
    shapeGroupMat= groupMatrix.shape
    NGrp = shapeGroupMat[1]
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
    
    output['NGrp'][sample_idx] = NGrp
    output['groupSizeAv'][sample_idx] = groupSizeAv
    output['groupSizeMed'][sample_idx] = groupSizeMed
    
    output['GrpDeathsCumul'][sample_idx] = NDGrp
    output['GrpBirthsCumul'][sample_idx] = NBGrp
    
    #calc change in group births and deaths
    if sample_idx>0:
        output['GrpDeaths'][sample_idx] = NDGrp - output['GrpDeathsCumul'][sample_idx-1] 
        output['GrpBirths'][sample_idx] = NBGrp - output['GrpBirthsCumul'][sample_idx-1] 
        output['GrpNetProd'][sample_idx] = output['GrpBirths'][sample_idx]  - \
                                           output['GrpDeaths'][sample_idx] 

    #calc moving average 
    if sample_idx >= 1:
        for varname in stateVarPlus:
            outname = varname + '_mav'
            mav, _ = util.calc_moving_av(
                output[varname], sample_idx, mavInt)
            output[outname][sample_idx] = mav

    # calc rms error
    if sample_idx >= rmsInt:
        if output['NCoop_mav'][sample_idx] > 0:
            output['rms_err_NCoop'][sample_idx] = util.calc_rms_error(
                    output['NCoop_mav'], sample_idx, rmsInt) / output['NCoop_mav'][sample_idx]
        else:
            output['rms_err_NCoop'][sample_idx] = np.nan    
        output['rms_err_NGrp'][sample_idx] = util.calc_rms_error(
            output['NGrp_mav'], sample_idx, rmsInt) / output['NGrp_mav'][sample_idx]

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
    
    output['NGrp'][sample_idx] = np.nan
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
    output['rms_err_NGrp'][sample_idx] = 0

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
@jit(void(f8[::1], f8[:, ::1], f8[::1], f8[::1], f8, f8, i8, i8, f8), nopython=True)
def calc_indv_rates(rates, groupMat, grSizeVec, birthRVec, deathR, delta_indv, NType, NGrp, alpha_b):
    
    #loop cell types
    for tt in range(NType):
        #setup indices
        cIdx = 2 * tt
        dIdx = 2 * tt + 1
        bIdxC1 = cIdx * NGrp 
        bIdxD1 = dIdx * NGrp
        dIdxC1 = bIdxC1 + 2 * NType * NGrp
        dIdxD1 = bIdxD1 + 2 * NType * NGrp
        
        #calc density of cooperating partners
        if NType == 1:
            #coopPart = n0/grSizeVec
            coopPart = groupMat[0, :] / grSizeVec
        else:
            #coopPart for type 1 = n2/grSizeVec * n3/grSizeVec * ... * n(NType)/grSizeVec
            #coopPart for type 2 = n1/grSizeVec * n3/N_gr * ... * n(NType)/N_gr
            #etc
            #vector of ones with size NGrp
            coopPart = np.ones(NGrp)
            for pp in range(NType):
                if pp != tt: #exclude self
                    coopPart *= groupMat[pp*2, :] / grSizeVec
                
        # calc rates
        if alpha_b != 0: 
            # to simulate results from Pichugin et al
            # implements group size dependent birth rate grpBEf = (Ni/Kind)^alpha_b
            # 1/Kind = deathR
            grpBEf = (deathR * grSizeVec) ** alpha_b
            rates[bIdxC1: bIdxC1 + NGrp] = grpBEf * birthRVec[cIdx] * coopPart * groupMat[cIdx, :]
            rates[bIdxD1: bIdxD1 + NGrp] = grpBEf * birthRVec[dIdx] * coopPart * groupMat[dIdx, :] 
        else:
            rates[bIdxC1: bIdxC1 + NGrp] = birthRVec[cIdx] * coopPart * groupMat[cIdx, :]
            rates[bIdxD1: bIdxD1 + NGrp] = birthRVec[dIdx] * coopPart * groupMat[dIdx, :] 

      
        if delta_indv != 0:
            rates[dIdxC1: dIdxC1 + NGrp] = deathR * groupMat[cIdx, :] * (grSizeVec ** delta_indv)
            rates[dIdxD1: dIdxD1 + NGrp] = deathR * groupMat[dIdx, :] * (grSizeVec ** delta_indv)
        else:
            rates[dIdxC1: dIdxC1 + NGrp] = deathR * groupMat[cIdx, :]
            rates[dIdxD1: dIdxD1 + NGrp] = deathR * groupMat[dIdx, :] 

    return None

# process individual level events
@jit(i8(f8[:, ::1], f8[::1], f8, f8[::1], i8, i8), nopython=True)
def process_indv_event(groupMat, indvRate, mutR, rand, NType, NGrp):
    # Note: groupMat is updated in place, it does not need to be returned 
    NTypeWMut = NType*2
    
    # select random event based on propensity
    eventID = util.select_random_event(indvRate, rand[0])

    # get event type
    eventType = math.floor(eventID/NGrp)
    # get event group
    eventGroup = eventID % NGrp  # % is modulo operator

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
            #NINGrp = oneVecType @ groupMat[:, eventGroup]
            NINGrp = groupMat[:, eventGroup].sum()
            if NINGrp == 0:  # all other types are zero too
                groupDeathID = int(eventGroup)

    return groupDeathID


"""============================================================================
Sub functions migration dynamics 
============================================================================"""

# process migration event
@jit(i8(f8[:, ::1], f8[::1], i8, i8, f8[::1]), nopython=True)
def process_migration_event(groupMat, grSizeVec, NGrp, NType, rand):
    # Note: groupMat is updated in place, it does not need to be returned

    # select random group of origin based on size
    grpIDSource = util.select_random_event(grSizeVec, rand[0])

    # select random type of migrant based on population size
    cellType = util.select_random_event(groupMat[:, grpIDSource], rand[1])
    
    # select random target group
    grpIDTarget = int(np.floor(rand[2] * NGrp))

    #perform migration
    groupMat[cellType, grpIDSource] -= 1
    groupMat[cellType, grpIDTarget] += 1

    # track if any groups die in process
    groupDeathID = int(-1)  # -1 is no death

    # kill group if last cell died
    # use two stage check for increased speed
    if groupMat[cellType, grpIDSource] == 0:  # killed last of type
        #NINGrp = oneVecType @ groupMat[:, eventGroup]
        NINGrp = groupMat[:, grpIDSource].sum()
        if NINGrp == 0:  # all other types are zero too
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
    NGrp = groupMat.shape[1]
    hasDied = np.zeros(NGrp)
    hasDied[groupDeathID] = 1

    # copy remaining groups to new matrix
    groupMat = groupMat[:, hasDied == 0]
    NGrp -= 1

    return (groupMat, NGrp)


# calculate fission and extinction rate of all groups
@jit(void(f8[::1], f8[:, ::1], f8[::1], f8, i8, f8, f8, f8, f8, f8, f8, f8), nopython=True)
def calc_group_rates(grpRate, groupMat, grSizeVec, NTot, NGrp, 
    gr_CFis, gr_SFis, K_grp, K_tot,
    delta_grp, delta_tot, delta_size):

    # calc fission rate
    fissionR = grSizeVec * gr_SFis + gr_CFis
  
    # calc extinction rate
    if delta_grp != 0:
        groupDep = (NGrp / K_grp) ** delta_grp
    else:
        groupDep = 1
        
    if delta_tot != 0:
        popDep = (NTot / K_tot) ** delta_tot
    else:
        popDep = 1    

    if delta_size != 0:
        sizeEffect = (1/grSizeVec) ** delta_size
        #sizeEffect = sizeEffect /  (onesGrp @ sizeEffect)
        extinctR = groupDep * popDep * sizeEffect
    else:
        extinctR = groupDep * popDep * np.ones(NGrp)

    # combine all rates in single vector
    grpRate[0:NGrp] = fissionR
    grpRate[NGrp::] = extinctR

    return None

    
@jit(Tuple((i8[:],i8))(f8, f8, i8), nopython=True)
def distribute_offspring(offspr_size, offspr_frac, NCellPar):
    #vector with destination index for all cells
    #initialize to -1: stay with parent
    destinationIdx = np.full(NCellPar, -1)
        
    # calc expected values
    nPerOff_expect = offspr_size * NCellPar
    nToOff_expect = offspr_frac * NCellPar
    
    #draw total number of cells passed to offspring from Poisson distribution
    nToOff = util.truncated_poisson(nToOff_expect, NCellPar)
    
    if nToOff > 0 and nPerOff_expect>0: 
        #draw size of offspring groups from truncated Poisson
        #min group size is 1, max is nToOff
        nPerOff = max(1, util.truncated_poisson(nPerOff_expect, nToOff))

        #first process full offspring
        nOffFull = int(np.floor(nToOff/nPerOff))
        nToOffFull = nOffFull*nPerOff
        destinationIdx[0:nToOffFull] = np.kron(np.arange(nOffFull), np.ones(nPerOff))
        #assign remaining cell to last offspring group
        destinationIdx[nToOffFull:nToOff] = nOffFull
        
        nOffspring = nOffFull        
        if nToOffFull < nToOff:
            nOffspring += 1
        
        #random shuffle matrix 
        destinationIdx = np.random.permutation(destinationIdx)
    else:
        nOffspring = 0
    
    return (destinationIdx, nOffspring)

@jit(Tuple((f8[::1],f8[:, ::1],i8))(f8[::1], f8, f8), nopython=True)
def fission_group(parentGroup, offspr_size, offspr_frac): 
    #get group properties
    NCellPar = int(parentGroup.sum())
    matShape = parentGroup.shape
    #distribute cells   
    destinationIdx, nOffspring = distribute_offspring(offspr_size, 
                                                      offspr_frac, 
                                                      NCellPar)
    if  nOffspring>0:   
        #init offspring and parremt array
        offspring = np.zeros((matShape[0], nOffspring))
        parrentNew = np.zeros(matShape[0])
        
        #find non zero elements
        ttIDxTuple = np.nonzero(parentGroup)
        ttIDxVec = ttIDxTuple[0]
        #loop all cells in parentgroup and assign to new group
        idx = 0
        for ttIDx in ttIDxVec:
            numCell = parentGroup[ttIDx]
            while numCell>0:
                currDest = destinationIdx[idx]
                if currDest == -1: #stays in parrent
                    parrentNew[ttIDx] += 1
                else:
                    offspring[ttIDx, currDest] += 1
                numCell -= 1
                idx += 1
    else:
        #nothing happens
        parrentNew = parentGroup
        offspring = np.zeros((0, 0)) 
        
    return (parrentNew, offspring, nOffspring)

# process individual level events
@jit(Tuple((f8[:, ::1], i8, i8, i8))(f8[:, ::1], f8[::1], f8[::1], f8, f8, i8, i8), nopython=True)
def process_group_event(groupMat, grpRate, rand, offspr_size, offspr_frac, NBGrp, NDGrp):
    # get number of groups
    NGrp = groupMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(grpRate, rand[0])

    # get event type
    eveNType = math.floor(eventID/NGrp)
    # get event group
    eventGroup = eventID % NGrp  # % is modulo operator

    if eveNType < 1:
        # fission event - add new group and split cells
        # get parent composition
        parentGroup = groupMat[:, eventGroup].copy()

        #perform fission process
        if offspr_size > 0:
            parrentNew, offspring, nOffspring = fission_group(parentGroup, offspr_size, offspr_frac)

            # only add daughter if not empty
            if offspring.size > 0:                
                if parrentNew.sum() > 0: # update parrent
                    groupMat[:, eventGroup] = parrentNew
                else: #remove parrent
                    groupMat, NGrp = remove_group(groupMat, eventGroup)
                    NDGrp += 1
                # add new daughter group
                groupMat = np.column_stack((groupMat, offspring))
                NBGrp += nOffspring
    
            NGrp = groupMat.shape[1]

    else:
        # extinction event - remove group
        groupMat, NGrp = remove_group(groupMat, eventGroup)
        NDGrp += 1

    return (groupMat, NGrp, NBGrp, NDGrp)


# create helper vectors for dot products
def create_helper_vector(NGrp, NType):
    onesGrp = np.ones(NGrp)
    onesIndR = np.ones(NGrp * NType * 4)
    onesGrpR = np.ones(NGrp * 2)
    #init rates matrix
    indvRate = np.ones(4 * NType * NGrp)
    grpRate = np.ones(2 * NGrp)

    return(onesGrp, onesIndR, onesGrpR, indvRate, grpRate)


# calc total number of individuals per group, use matrix product for speed
@jit(Tuple((f8[::1], f8))(f8[:, ::1], f8[::1], f8[::1]), nopython=True)
def calc_group_state(groupMat, oneVecType, onesGrp):
    #vector with size of each group
    grSizeVec = oneVecType @ groupMat
    #float total number of individuals
    NTot = onesGrp @ grSizeVec
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
    
    
    mavInt = max(mavInt, 1)
    rmsInt = max(rmsInt, 1)

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
    birthRVecCoop = 1 / (indv_asymmetry ** (np.arange(NType)))
    birthRVecCoop *= (birthRVecCoop.sum() ** (NType - 1)) / np.prod(birthRVecCoop)
    #include costs and calc birth rates for cooperators and defectors respectively
    birthRVec = np.kron(birthRVecCoop, np.array([(1 - indv_cost), 1]))
    #convert caryinf capacity to death rate
    indv_deathR = 1 / indv_K
    oneVecType = np.ones(2 * NType)

    return (oneVecType, birthRVec, indv_deathR)

"""============================================================================
Main model code
============================================================================"""

# main model
def run_model(model_par):
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]

    # get individual rates
    NType      = int(model_par['indv_NType'])
    indv_mutR  = float(model_par['indv_mutR'])
    delta_indv = float(model_par['delta_indv'])
    inv_migrR  = float(model_par['indv_migrR'])
    indv_K     = float(model_par['indv_K'])
    if 'indv_tau' in model_par:
        indv_tau   = float(model_par['indv_tau'])
    else:
        indv_tau = 1
        
    #get group rates
    gr_CFis    = float(model_par['gr_CFis'])
    gr_SFis    = float(model_par['gr_SFis']) / indv_K
    alpha_b    = float(model_par['alpha_b'])
    K_grp      = float(model_par['K_grp'])
    K_tot      = float(model_par['K_tot'])
    delta_grp  = float(model_par['delta_grp'])
    delta_tot  = float(model_par['delta_tot'])
    delta_size = float(model_par['delta_size'])

    #get group reproduction traits
    offspr_size = float(model_par['offspr_size'])
    offspr_frac = float(model_par['offspr_frac'])
    
    #check rates
    if offspr_size > 0.5: 
        print('cannot do that: offspr_size < 0.5 and offspr_size < offspr_frac < 1')
        raise ValueError
    elif offspr_frac < offspr_size or offspr_frac > (1-offspr_size):
        print('cannot do that: offspr_frac should be offspr_size < offspr_frac < 1-offspr_size')
        raise ValueError
    
    # Initialize model, get rates and init matrices
    maxT, minTRun, sampleInt, mavInt, rmsInt = calc_time_steps(model_par)
                        
    # init counters
    currT = 0
    ttR = 0
    sampleIdx = 0
    #counters to count group birth and death events
    NBGrp = 0
    NDGrp = 0

    #init static helper vectors
    oneVecType, birthRVec, indv_deathR = adjust_indv_rates(model_par)
    
    # initialize output matrix
    output, distFCoop, binFCoop, distGrSize, binGrSize = init_output_matrix(model_par)
    
    # initialize group matrix
    groupMat = init_groupMat(model_par)
    NGrp     = groupMat.shape[1]

    # creates matrix with rndSize0 entries, it is recreated if needed
    rndSize0 = int(1E6)
    rndSize1 = 5 
    randMat = util.create_randMat(rndSize0, rndSize1)

    #init dynamic helper vectors
    onesGrp, onesIndR, onesGrpR, indvRate, grpRate = \
        create_helper_vector(NGrp, NType)

    # get first sample of init state
    sampleIdx = sample_model(groupMat, output, distFCoop, binFCoop,
                             distGrSize, binGrSize, sampleIdx, currT, 
                             mavInt, rmsInt, stateVarPlus,
                             NBGrp, NDGrp)

    # loop time steps
    while currT <= maxT:
        # reset rand matrix when used up
        if ttR >= rndSize0:
            randMat = util.create_randMat(rndSize0, rndSize1)
            ttR = 0

        #calc group state
        grSizeVec, NTot = calc_group_state(groupMat, oneVecType, onesGrp)

        # calc rates of individual level events
        calc_indv_rates(indvRate, groupMat, grSizeVec, birthRVec,
                        indv_deathR, delta_indv,
                        NType, NGrp, alpha_b)
        
        # calc rates of group events
        calc_group_rates(grpRate, groupMat, grSizeVec, NTot, NGrp,
                        gr_CFis, gr_SFis, K_grp, K_tot,
                        delta_grp, delta_tot, delta_size)

        # calculate total propensities
        indvProp = indv_tau * (onesIndR @ indvRate)
        groupProp = onesGrpR @ grpRate
        migrProp = inv_migrR * NTot
        totProp = indvProp + groupProp + migrProp

        # calc time step
        dt = -1 * math.log(randMat[ttR, 1]) / totProp

        # select group or individual event
        rescaledRand = randMat[ttR, 0] * totProp
        groupsHaveChanged = False
        if rescaledRand < indvProp:
            # individual level event - select and process individual level event
            groupDeathID = process_indv_event(groupMat, indvRate, 
                                              indv_mutR, randMat[ttR, 2:4], NType, NGrp)
            if groupDeathID > -1:  # remove empty group
                groupMat, NGrp = remove_group(groupMat, groupDeathID)
                NDGrp += 1
                groupsHaveChanged = True
        elif rescaledRand < (indvProp + migrProp):
            # migration event - select and process migration event
            groupDeathID = process_migration_event(groupMat, grSizeVec, 
                                                   NGrp, NType, randMat[ttR, 2:5])
            if groupDeathID > -1:  # remove empty group
                groupMat, NGrp = remove_group(groupMat, groupDeathID)
                NDGrp += 1
                groupsHaveChanged = True
        else:
            # group level event - select and process group level event
            groupMat, NGrp, NBGrp, NDGrp = process_group_event(groupMat, grpRate, randMat[ttR, 2:4], 
                                                 offspr_size, offspr_frac, NBGrp, NDGrp)
            groupsHaveChanged = True

        if groupsHaveChanged:
            if NGrp == 0:  # if all groups have died, end simulation
                sampleIdx = sample_extinction(output, distFCoop, binFCoop,
                                              distGrSize, sampleIdx, currT, stateVarPlus)
                break
            else: #otherwise, recreate helper vectors
                onesGrp, onesIndR, onesGrpR, indvRate, grpRate = create_helper_vector(NGrp, 
                                                                                      NType)

        # update time
        currT += dt
        ttR += 1
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = sample_model(groupMat, output, distFCoop, binFCoop,
                                     distGrSize, binGrSize, sampleIdx, currT, 
                                     mavInt, rmsInt, stateVarPlus,
                                     NBGrp, NDGrp)
            # check if steady state has been reached
            if currT > minTRun:
                NCoopStable = output['rms_err_NCoop'][sampleIdx - 1] \
                    < model_par['rms_err_trNCoop']
                NGrpStable = output['rms_err_NGrp'][sampleIdx - 1] \
                    < model_par['rms_err_trNGr']

                if NCoopStable and NGrpStable:
                    break
                
            # check if population size remains in bounds
            if output['NTot'][sampleIdx - 1] > model_par['maxPopSize']:
                sample_nan(groupMat, output, distFCoop, binFCoop,
                                     distGrSize, binGrSize, sampleIdx - 1, 
                                     currT, mavInt, rmsInt, stateVarPlus)
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

#run model store only final state 
def single_run_finalstate(model_par):
    """[Runs MLS model and stores final state]
    
    Parameters
    ----------
    model_par : [Dictionary]
        [Stores model parameters]
    
    Returns
    -------
    output_matrix : [Numpy recarray]
        [Contains steady state values of system variables and parameters]
    endDistFCoop : [Numpy ndarray]
        [Contains steady state distribution of frequency of cooperators]
    endDistGrSize : [Numpy ndarray]
        [Contains steady state distribution of group sizes]
        
    """    
    # run model
    
    start = time.time()
    output, distFCoop, distGrSize = run_model(model_par)    
    end = time.time()
 
    #input parameters to store
    parList = ['indv_NType', 'indv_cost', 'indv_K', 
               'indv_mutR','indv_migrR', 'indv_asymmetry', 'delta_indv',
               'gr_SFis', 'gr_CFis', 'alpha_b', 'K_grp', 'K_tot',
               'delta_grp', 'delta_tot', 'delta_size',
               'offspr_size','offspr_frac','run_idx','perimeter_loc']
                                                        
    stateVarPlus = stateVar + \
        ['N%i' % x for x in range(model_par['indv_NType'])] + \
        ['N%imut' % x for x in range(model_par['indv_NType'])]

    # init output matrix
    dTypeList1 = [(x, 'f8') for x in stateVarPlus]
    dTypeList2 = [(x+'_mav', 'f8') for x in stateVarPlus]
    dTypeList3 = [(x, 'f8') for x in parList]
    dTypeList = dTypeList1 + dTypeList2 + dTypeList3 + [('run_time', 'f8')]
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
         #time and run settings
        "maxT":             100,  # total run time
        "maxPopSize":       10000,  #stop simulation if population exceeds this number
        "minT":             1,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       10,    # average over this time window
        "rms_window":       10,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-51,   # when to stop calculations
        "rms_err_trNGr":    5E-51,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    300,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   20,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       2,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,  # cost of cooperation
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,   # mutation rate to cheaters
        # group size control
        "indv_K":           50,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          1/100,
        'gr_SFis':          1/50,
        'alpha_b':          0,
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            30000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.125,  # offspr_size <= 0.5 and
        'offspr_frac':      0.8,  # offspr_size < offspr_frac < 1-offspr_size'
         # extra settings
        'run_idx':          1,
        'perimeter_loc':    0
    }
    output, distFCoop, distGrSize = run_model(model_par)
