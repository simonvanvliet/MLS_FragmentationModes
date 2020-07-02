#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 23 2020

Implements version of MLS model of group dynamics (no evolution) with rate functions from 
Pichugin, Peña, Rainey, Traulsen; PLOS CompBio; 2017

Only individual and group level rate functions are different, rest of model remains unchanged 


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
import scipy.stats as stats
import MlsGroupDynamics_utilities as util
import MlsGroupDynamics_main as mls
import time

#output variables to store
stateVar = ['NTot', 'fCoop', 'NGrp', 'groupSizeAv', 'groupSizeMed']

"""============================================================================
Pichugin et al like rate functions  
============================================================================"""

# calculate birth and death rate for all groups and types
# @jit provides speedup by compling this function at start of execution
# To use @jit provide the data type of output and input, nopython=true makes compilation faster
@jit(void(f8[::1], f8[:, ::1], f8[::1], f8, f8, i8, i8), nopython=True)
def calc_indv_rates(rates, groupMat, grSizeVec, indv_K, alpha_b, NType, NGrp):
    
    #loop cell types
    for tt in range(NType):
        #setup indices
        cIdx = 2 * tt
        dIdx = 2 * tt + 1
        bIdxC1 = cIdx * NGrp 
        bIdxD1 = dIdx * NGrp
        dIdxC1 = bIdxC1 + 2 * NType * NGrp
        dIdxD1 = bIdxD1 + 2 * NType * NGrp
                        
        # calc rates
        # to simulate results from Pichugin et al
        # implements group size dependent birth rate grpBEf = 1 +  M*(Ni-1/Kind-2)^alpha_b
        # we keep M=1
        grpBEf = ((grSizeVec - 1) / (indv_K - 2)) ** alpha_b
        rates[bIdxC1: bIdxC1 + NGrp] = (1 + grpBEf) * groupMat[cIdx, :]
        rates[bIdxD1: bIdxD1 + NGrp] = 0 
        
        rates[dIdxC1: dIdxC1 + NGrp] = 0
        rates[dIdxD1: dIdxD1 + NGrp] = 0
        
    return None

# calculate fission and extinction rate of all groups
@jit(void(f8[::1], f8[::1], f8, f8, i8), nopython=True)
def calc_group_rates(grpRate, grSizeVec, gr_CFis, K_ind, NGrp):
    # calc fission rate
    fissionR = (grSizeVec >= K_ind) * gr_CFis 
    extinctR = np.zeros_like(fissionR)

    # combine all rates in single vector
    grpRate[0:NGrp] = fissionR
    grpRate[NGrp::] = extinctR

    return None

 

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
    inv_migrR  = float(model_par['indv_migrR'])
    indv_mutR  = float(model_par['indv_mutR'])
    indv_K     = float(model_par['indv_K'])
    if 'indv_tau' in model_par:
        indv_tau   = float(model_par['indv_tau'])
    else:
        indv_tau = 1
        
    #get group rates
    gr_CFis    = float(model_par['gr_CFis'])
    alpha_b    = float(model_par['alpha_b'])

    #get group reproduction traits
    offspr_size = float(model_par['offspr_size'])
    offspr_frac = float(model_par['offspr_frac'])
    
    #check rates
    if NType > 1: 
        print('cannot do that, code only supports 1 species')
        raise ValueError
    if offspr_size > 0.5: 
        print('cannot do that: offspr_size < 0.5 and offspr_size < offspr_frac < 1')
        raise ValueError
    elif offspr_frac < offspr_size or offspr_frac > (1-offspr_size):
        print('cannot do that: offspr_frac should be offspr_size < offspr_frac < 1-offspr_size')
        raise ValueError
    
    # Initialize model, get rates and init matrices
    maxT, minTRun, sampleInt, mavInt, rmsInt = mls.calc_time_steps(model_par)
                        
    # init counters
    currT = 0
    ttR = 0
    sampleIdx = 0
    #counters to count group birth and death events
    NBGrp = 0
    NDGrp = 0

    #init static helper vectors
    oneVecType = np.ones(2)
    
    # initialize output matrix
    output, distFCoop, binFCoop, distGrSize, binGrSize = mls.init_output_matrix(model_par)
    
    # initialize group matrix
    groupMat = mls.init_groupMat(model_par)
    NGrp     = groupMat.shape[1]

    # creates matrix with rndSize0 entries, it is recreated if needed
    rndSize0 = int(1E6)
    rndSize1 = 5 
    randMat = util.create_randMat(rndSize0, rndSize1)

    #init dynamic helper vectors
    onesGrp, onesIndR, onesGrpR, indvRate, grpRate = \
        mls.create_helper_vector(NGrp, NType)

    # get first sample of init state
    sampleIdx = mls.sample_model(groupMat, output, distFCoop, binFCoop,
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
        grSizeVec, NTot = mls.calc_group_state(groupMat, oneVecType, onesGrp)

        # calc rates of individual level events
        calc_indv_rates(indvRate, groupMat, grSizeVec,
                        indv_K, alpha_b,
                        NType, NGrp)
        
        # calc rates of group events
        calc_group_rates(grpRate, grSizeVec, gr_CFis, indv_K, NGrp)

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
            groupDeathID = mls.process_indv_event(groupMat, indvRate, 
                                              indv_mutR, randMat[ttR, 2:4], NType, NGrp)
            if groupDeathID > -1:  # remove empty group
                groupMat, NGrp = mls.remove_group(groupMat, groupDeathID)
                NDGrp += 1
                groupsHaveChanged = True
        elif rescaledRand < (indvProp + migrProp):
            # migration event - select and process migration event
            groupDeathID = mls.process_migration_event(groupMat, grSizeVec, 
                                                   NGrp, NType, randMat[ttR, 2:5])
            if groupDeathID > -1:  # remove empty group
                groupMat, NGrp = mls.remove_group(groupMat, groupDeathID)
                NDGrp += 1
                groupsHaveChanged = True
        else:
            # group level event - select and process group level event
            groupMat, NGrp, NBGrp, NDGrp = mls.process_group_event(groupMat, grpRate, randMat[ttR, 2:4], 
                                                 offspr_size, offspr_frac, NBGrp, NDGrp)
            groupsHaveChanged = True

        if groupsHaveChanged:
            if NGrp == 0:  # if all groups have died, end simulation
                sampleIdx = mls.sample_extinction(output, distFCoop, binFCoop,
                                              distGrSize, sampleIdx, currT, stateVarPlus)
                break
            else: #otherwise, recreate helper vectors
                onesGrp, onesIndR, onesGrpR, indvRate, grpRate = mls.create_helper_vector(NGrp, 
                                                                                      NType)

        # update time
        currT += dt
        ttR += 1
        # sample model at intervals
        nextSampleT = sampleInt * sampleIdx
        if currT >= nextSampleT:
            sampleIdx = mls.sample_model(groupMat, output, distFCoop, binFCoop,
                                     distGrSize, binGrSize, sampleIdx, currT, 
                                     mavInt, rmsInt, stateVarPlus,
                                     NBGrp, NDGrp)
            
            # check if final population size has been reached 
            if output['NTot'][sampleIdx - 1] > model_par['maxPopSize']:
                break
            
            if currT > model_par['maxT']:
                output = None
                break
            

    # cut off non existing time points at end
    if output is not None:
        output = output[0:sampleIdx]

    return output


"""============================================================================
Code that calls model and plots results
============================================================================"""

#run model store only final state 
def single_run_trajectories(model_par):
    """[Runs MLS model and stores trajectories state]
    
    Parameters
    ----------
    model_par : [Dictionary]
        [Stores model parameters]
        
    """    
    # run model
    start = time.time()
    output = run_model(model_par)    
    end = time.time()
    
    #Fit growth rates
    if output is not None:
        startFit = model_par['startFit']
        toFit = output['NTot'] >= startFit
        tFit = output['time'][toFit]
        logNFit = np.log10(output['NTot'][toFit])
        logNGrpFit = np.log10(output['NGrp'][toFit])
        
        r_tot, _, corr_coef_tot, _, _ = stats.linregress(tFit, logNFit)
        r_groups, _, corr_coef_groups, _, _ = stats.linregress(tFit, logNGrpFit)
    else:
        r_tot = np.nan
        r_groups = np.nan
        corr_coef_tot = np.nan
        corr_coef_groups = np.nan


    # init output matrix
    dType = np.dtype([
        ('indv_K', 'f8'),
        ('alpha_b', 'f8'),
        ('offspr_size', 'f8'),
        ('offspr_frac', 'f8'),
        ('r_groups', 'f8'),
        ('r_tot', 'f8'),
        ('corr_coef_groups', 'f8'),
        ('corr_coef_tot', 'f8')])
    output = np.zeros(1, dType)
    
    #store output
    output['indv_K'] = model_par['indv_K']
    output['alpha_b'] = model_par['alpha_b']
    output['offspr_size'] = model_par['offspr_size']
    output['offspr_frac'] = model_par['offspr_frac']
    output['r_groups'] = r_groups
    output['r_tot'] = r_tot
    output['corr_coef_groups'] = corr_coef_groups
    output['corr_coef_tot'] = corr_coef_tot

    return (output)


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
        'grp_tau':          1,
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
