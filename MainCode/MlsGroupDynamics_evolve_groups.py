#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 6 2020
Last Update March 28 2020

Implements main MLS model of group dynamics with evolution of group level traits

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
def init_traitMat(model_par):
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
    
    offspr_frac_idx = offsprFracMatrix[offspr_frac_idx, offspr_size_idx]

    #create group composition vector
    nCoop = round(model_par["init_groupDens"] * model_par['init_fCoop'] / model_par['indv_NType'])
    nDef  = round(model_par["init_groupDens"] * (1 - model_par['init_fCoop']) / model_par['indv_NType'])
    
    # init all groups with zero
    traitMat = np.zeros((2, NGroup), dtype='i8', order='C')
    grpMat = np.zeros((NType * 2, NGroup), order='C')

    # set group prop
    traitMat[0, :] = offspr_frac_idx 
    traitMat[1, :] = offspr_size_idx
        
    grpMat[0::2, :] = nCoop
    grpMat[1::2, :] = nDef

    return (traitMat, grpMat)


"""============================================================================
Sample model code 
============================================================================"""
@jit(Tuple((f8[:, ::1], f8, f8))(i8[:, ::1]), nopython=True)
def summarize_traitMat(traitMat):
    #matrix with number per type
    #find existing groups and sum over all groups and cell types
    traitDistr = np.zeros((nBinOffsprFrac, nBinOffsprSize))
    grpNum = traitMat.shape[1]
    
    for i in range(grpNum):
        traitDistr[traitMat[0,i], traitMat[1,i]] += 1
    
    #calculate mean trait values using marginal distributions
    av_size = np.sum(binCenterOffsprSize[traitMat[1, :]]) / grpNum
    av_frac = np.sum(binCenterOffsprFrac[traitMat[0, :]]) / grpNum

    return (traitDistr, av_size, av_frac)

# sample model
def sample_model(traitMat, grpMat, outputMat, traitDistr, 
    sample_idx, currT, mavInt, rmsInt, stateVarPlus):
    # store time
    outputMat['time'][sample_idx] = currT

    # calc number of groups
    shapetraitMat = grpMat.shape
    NGroup = shapetraitMat[1]
    NType = int(shapetraitMat[0] / 2)

    # summarize groups
    traitDistrCurr, av_size, av_frac = summarize_traitMat(traitMat)

    # get group statistics
    NTot, NCoop, groupSizeAv, groupSizeMed, NTot_type, fCoop_group, grSizeVec = mls.calc_cell_stat(
        grpMat)

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

    # store distribution of traits
    traitDistr[sample_idx, :, :] = traitDistrCurr / NTot

    sample_idx += 1
    return sample_idx

# sample model
def sample_extinction(outputMat, traitDistr, sample_idx, currT, stateVarPlus):
    # store time
    outputMat['time'][sample_idx] = currT

    # calc total population sizes
    for varname in stateVarPlus:
        outname = varname + '_mav'
        outputMat[varname][sample_idx] = 0
        outputMat[outname][sample_idx] = 0

    # calc distribution groupsizes
    traitDistr[sample_idx, :, :] = 0
    sample_idx += 1

    return sample_idx


"""============================================================================
Sub functions group dynamics 
============================================================================"""

# remove group from group matrix
@jit(Tuple((i8[:, ::1], f8[:, ::1]))(i8[:, ::1], f8[:, ::1], i8), nopython=True)
def remove_group(traitMat, grpMat, groupDeathID):
    #find group that died
    NGrp = grpMat.shape[1]
    hasDied = np.zeros(NGrp)
    hasDied[groupDeathID] = 1
    # copy remaining groups to new matrix
    grpMatNew = grpMat[:, hasDied == 0]
    grpMatNew = grpMatNew.copy()
   
    # copy traits of remaining groups to new matrix
    traitMatNew = traitMat[:, hasDied == 0]
    traitMatNew = traitMatNew.copy()
        
    return (traitMatNew, grpMatNew)


@jit(UniTuple(i8, 2)(f8, f8, i8, i8, f8[::1]), nopython=True)
def mutate_group(mutR_frac, mutR_size, fracIdx, sizeIdx, rand):
    #check for mutation in offspring size
    if rand[0].item() < mutR_size / 2:  # offspring size mutates to lower value
        offsprSizeIdx = max(0, sizeIdx - 1)
    elif rand[0].item() < mutR_size:  # offspring size mutates to lower value
        offsprSizeIdx = min(nBinOffsprSize - 1, sizeIdx + 1)
    else:
        offsprSizeIdx = sizeIdx

    #check for mutation in offspring fraction
    if rand[1].item() < mutR_frac / 2:  # offspring size mutates to lower value
        offsprFracIdx = max(0, fracIdx - 1)
    elif rand[1].item() < mutR_frac:  # offspring size mutates to lower value
        offsprFracIdx = min(nBinOffsprFrac - 1, fracIdx + 1)
    else:
        offsprFracIdx = fracIdx

    #make sure we stay inside allowed trait space
    offsprFracIdx = offsprFracMatrix[offsprFracIdx, offsprSizeIdx]
    
    return (offsprFracIdx, offsprSizeIdx)
    
@jit(Tuple((i8[:, ::1], f8[:, ::1]))(i8[:, ::1], f8[:, ::1], i8, f8, f8), nopython=True)
def fission_group(traitMat, grpMat, eventGroup, mutR_frac, mutR_size):
    #get parent
    parentGroup = grpMat[:, eventGroup].copy()
    
    #get parent properties
    fracIdx = traitMat[0, eventGroup]
    sizeIdx = traitMat[1, eventGroup]
    offspr_frac = binCenterOffsprFrac[fracIdx]
    offspr_size = binCenterOffsprSize[sizeIdx]
    NCellPar = int(parentGroup.sum())
    
    #distribute cells   
    destIdx, nOffspring = mls.distribute_offspring(offspr_size,
                                                   offspr_frac, 
                                                   NCellPar)  
    
    if nOffspring > 0: 
        if np.sum(destIdx==-1) > 0:
            #consider parent to be new group, remove old parent
            destIdx += 1
            nPar = 1
        else:
            nPar = 0
                
        #init new 2D array  
        matShape2D = grpMat.shape
        nGrpAdded = nOffspring + nPar
        nGrpNew = matShape2D[1] + nGrpAdded - 1
        isParent = np.zeros(matShape2D[1])
        isParent[eventGroup] = 1
        
        grpMatNew = np.zeros((matShape2D[0], nGrpNew))
        traitMatNew = np.zeros((2, nGrpNew), dtype=i8)
        
        #store existing groups at end, exclude parent  
        grpMatNew[:, nGrpAdded::] = grpMat[:, isParent==0]
        grpMatNew = grpMatNew.copy()
        
        traitMatNew[:, nGrpAdded::] = traitMat[:, isParent==0]
        traitMatNew = traitMatNew.copy()
        
        #find non zero elements
        ttIDxTuple = np.nonzero(parentGroup)
        ttIDxVec = ttIDxTuple[0]
        
        #loop all cells in parentgroup and assign to new group
        idx = 0
        for ttIDx in ttIDxVec:
            numCell = parentGroup[ttIDx]
            while numCell>0:
                currDest = destIdx[idx]
                grpMatNew[ttIDx, currDest] += 1
                numCell -= 1
                idx += 1   
        
        #mutate offspring groups
        rndMat = np.random.random((nOffspring,2))
        idx = 0
        if nPar==1: #re-assign parent traits 
            traitMatNew[0, idx] = fracIdx
            traitMatNew[1, idx] = sizeIdx
            idx += 1
        for n in range(nOffspring):
            #mutate offspring
            offsprFracIdx, offsprSizeIdx = mutate_group(mutR_frac, mutR_size, 
                                                        fracIdx, sizeIdx, 
                                                        rndMat[n,:])
             
            traitMatNew[0, idx] = offsprFracIdx
            traitMatNew[1, idx] = offsprSizeIdx
            idx += 1
                                             
    else:
        #nothing happens
        grpMatNew = grpMat
        traitMatNew = traitMat
                
    return (traitMatNew, grpMatNew)

# process group level events
@jit(Tuple((i8[:, ::1], f8[:, ::1]))(i8[:, ::1], f8[:, ::1], f8[::1], f8[::1], f8, f8), nopython=True)
def process_group_event(traitMat, grpMat, grpRate, rand, mutR_frac, mutR_size):
    # get number of groups
    NGroup = grpMat.shape[1]

    # select random event based on propensity
    eventID = util.select_random_event(grpRate, rand[0])
    # get event type
    eventType = math.floor(eventID/NGroup)
    # get event group
    eventGroup = eventID % NGroup  # % is modulo operator
    
    if eventType < 1:
        # fission event - add new group and split cells
        traitMat, grpMat = fission_group(traitMat, grpMat, 
                                        eventGroup,
                                        mutR_frac, mutR_size)
    else:
        # extinction event - remove group
        traitMat, grpMat = remove_group(traitMat, grpMat, eventGroup)
    return (traitMat, grpMat)


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
    gr_CFis    = float(model_par['gr_CFis'])
    gr_SFis    = float(model_par['gr_SFis']) / indv_K
    alpha_Fis  = float(model_par['alpha_Fis'])
    K_grp      = float(model_par['K_grp'])
    K_tot      = float(model_par['K_tot'])
    delta_grp  = float(model_par['delta_grp'])
    delta_tot  = float(model_par['delta_tot'])
    delta_size = float(model_par['delta_size'])
    indv_tau   = float(model_par['indv_tau'])

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
    traitMat, grpMat = init_traitMat(model_par)
    NGroup = grpMat.shape[1]

    #init dynamic helper vectors
    onesNGrp, onesIndR, onesGrR, indvRate, grpRate = mls.create_helper_vector(
        NGroup, NType)

    # get first sample of init state
    sampleIdx = sample_model(traitMat, grpMat, 
                             outputMat, traitDistr, 
                             sampleIdx, currT, mavInt, rmsInt, 
                             stateVarPlus)

    # loop time steps
    while currT <= maxT:
        # reset rand matrix when used up
        if ttR >= rndSize0:
            randMat = util.create_randMat(rndSize0, rndSize1)
            ttR = 0 

        #calc group state
        grSizeVec, NTot = mls.calc_group_state(grpMat, 
                                               onesNType, onesNGrp)    

        # calc rates of individual level events
        mls.calc_indv_rates(indvRate, grpMat, grSizeVec, birthRVec,
                            deathR, delta_indv, NType, NGroup)
        
        
        # calc rates of group events
        mls.calc_group_rates(grpRate, grpMat, grSizeVec, NTot, NGroup,
                            gr_CFis, gr_SFis, alpha_Fis, K_grp, K_tot,
                            delta_grp, delta_tot, delta_size)

        # calculate total propensities
        indvProp = indv_tau * (onesIndR @ indvRate)
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
            groupDeathID = mls.process_indv_event(grpMat, indvRate, 
                                                  mutR_type, randMat[ttR, 2:4], 
                                                  NType, NGroup)
            if groupDeathID > -1:  # remove empty group
                traitMat, grpMat = remove_group(traitMat, grpMat, groupDeathID)
                groupsHaveChanged = True
        elif rescaledRand < (indvProp + migrProp):
            # migration event - select and process migration event
            groupDeathID = mls.process_migration_event(grpMat, grSizeVec,
                                                       NGroup, NType, 
                                                       randMat[ttR, 2:5])
            if groupDeathID > -1:  # remove empty group
                traitMat, grpMat = remove_group(traitMat, grpMat, groupDeathID)
                groupsHaveChanged = True
        else:
            # group level event - select and process group level event
            traitMat, grpMat = process_group_event(traitMat, grpMat, 
                                                    grpRate, randMat[ttR, 2:4],
                                                    mutR_frac, mutR_size)
            groupsHaveChanged = True
         
        # update group matrices if needed    
        if groupsHaveChanged:
            NGroup = grpMat.shape[1]
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
            sampleIdx = sample_model(traitMat, grpMat, 
                             outputMat, traitDistr, 
                             sampleIdx, currT, mavInt, rmsInt, 
                             stateVarPlus)
            
    # cut off non existing time points at end
    outputMat = outputMat[0:sampleIdx]
    traitDistr = traitDistr[0:sampleIdx, :, :]
    
    if outputMat['NCoop'][-1] == 0:
        outputMat['NCoop_mav'][-1] = 0
    
    return (outputMat, traitDistr)


"""============================================================================
Code that calls model and plots results
============================================================================"""

def single_run_save(model_par, mainName):
    """[Runs evolution model and saves results to disk in .npz file]
    
    Arguments:
        model_par {[dictionary]} -- [model parameters] 
        
        mainName {[string]} -- [filename for data file, appended with parameter settings]
    Returns:
        [numpy array] -- [trait distribution at last timepoint]
    """
    #create file name, append mainName with parameter settings
    parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_CFis'       : 'fisC',
                'gr_SFis'       : 'fisS',
                'alpha_Fis'     : 'fisA',
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
                'offspr_fracInit':'frIn',
                'indv_tau'      : 'tInd'}
    
    parListName = ['gr_SFis', 'indv_cost', 'mutR_type',
                   'mutR_size', 'mutR_frac', 'offspr_sizeInit',
                   'offspr_fracInit', 'indv_K',
                   'indv_migrR','indv_tau']
    parName = ['_%s%.0g' %(parNameAbbrev[x], model_par[x]) for x in parListName]
    parName = ''.join(parName)
    fileName = mainName + parName + '.npz'
    
    #run model and save data to disk
    #try: 
    outputMat, traitDistr = run_model(model_par)  
    np.savez(fileName, output=outputMat, traitDistr=traitDistr,
             model_par=[model_par])
#    except:
#        print("error with run")
#        traitDistr = np.full((1, nBinOffsprFrac, nBinOffsprSize), np.nan)
    
    return traitDistr
    

# this piece of code is run only when this script is executed as the main
if __name__ == "__main__":
    print("running with default parameter")

    model_par = {
        #time and run settings
        "maxT":             5,  # total run time
        "maxPopSize":       0,  #stop simulation if population exceeds this number
        "minT":             10,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       5,    # average over this time window
        "rms_window":       5,    # calc rms change over this time window
        "rms_err_trNCoop":  0,   # when to stop calculations
        "rms_err_trNGr":    0,   # when to stop calculations
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
        "indv_migrR":       0,   # mutation rate to cheaters
        # set mutation rates
        'mutR_type':        1E-3,
        'mutR_size':        2E-2, 
        'mutR_frac':        2E-2, 
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          1/100,
        'gr_SFis':          4,
        'alpha_Fis':        1,
        'indv_tau':         0.1,
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,    # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            5000,   # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # initial settings for fissioning
        'offspr_sizeInit':  0.05,  # offspr_size <= 0.5 and
        'offspr_fracInit':  0.9  # offspr_size < offspr_frac < 1-offspr_size'
    }
    
    start = time.time()
    outputMat, traitDistr = run_model(model_par)    
    end = time.time()
    print(end - start)
    print('done')
