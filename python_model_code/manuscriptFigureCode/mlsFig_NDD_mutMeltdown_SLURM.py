#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on 2021-07-09

- For each strategy, maximum mutation rate, no migration, no density dependence
- Fission slope increased to 1E9 when Ni > K_ind

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""

import sys
sys.path.insert(0, '..')

from mainCode import MlsGroupDynamics_main as mls
from mainCode import MlsGroupDynamics_utilities as util
import numpy as np
import pandas as pd
import itertools
import os
#import plotParScan

"""============================================================================
Define parameters
============================================================================"""

#running with minT = 200, Ktot = 4E4, maxPopSize = 1E5

#SET OUTPUT FILENAME
mainName = 'mutMelt_NoDensDep-2021-07-09'

#SET nr of replicates
nReplicate = 7

#set  mutation rates to try
mu_Vec = np.logspace(0,-7,20)

#setup 2D parameter grid
offspr_size_Vec = np.arange(0.01, 0.5, 0.034)
offspr_frac_Vec = np.arange(0.01, 1, 0.07)

#set model mode settings (slope and migration rate)
mode_set = np.array([[np.inf],
                     [0]])
mode_vec = np.arange(mode_set.shape[1])

#set number of types to scan
nType_vec = np.array([1, 2, 3, 4]) #parameter values

#SET rest of model parameters
model_par_def = {
          #time and run settings
        "maxRunTime":       900,     #max cpu time in seconds
        "maxT":             5000,  # total run time
        "maxPopSize":       2E5,  #stop simulation if population exceeds this number
        "minT":             200,    # min run time
        "sampleInt":        1,      # sampling interval
        "mav_window":       200,    # average over this time window
        "rms_window":       200,    # calc rms change over this time window
        "rms_err_trNCoop":  1E-2,   # when to stop calculations
        "rms_err_trNGr":    5E-2,   # when to stop calculations
        # settings for initial condition
        "init_groupNum":    100,     # initial # groups
        "init_fCoop":       1,
        "init_groupDens":   50,     # initial total cell number in group
        # settings for individual level dynamics
        # complexity
        "indv_NType":       1,
        "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
        # mutation load
        "indv_cost":        0.01,   # cost of cooperation
        "indv_mutR":        1E-3,   # mutation rate to cheaters
        "indv_migrR":       0,      # mutation rate to cheaters
        # group size control
        "indv_K":           100,     # total group size at EQ if f_coop=1
        "delta_indv":       0,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0.05,
        'gr_SFis':          np.inf,     # measured in units of 1 / indv_K
        'grp_tau':          1,     # constant multiplies group rates
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            1E4,    # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.01,  # offspr_size <= 0.5 and
        'offspr_frac':      0.01,    # offspr_size < offspr_frac < 1-offspr_size'
        # extra settings
        'run_idx':          1,
        'replicate_idx':    1,
        'perimeter_loc':    0
    }


"""============================================================================
Define functions
============================================================================"""

parNameAbbrev = {
                'delta_indv'    : 'dInd',
                'delta_grp'     : 'dGrp',
                'delta_tot'     : 'dTot',
                'delta_size'    : 'dSiz',
                'gr_CFis'       : 'fisC',
                'gr_SFis'       : 'fisS',
                'indv_NType'    : 'nTyp',
                'indv_asymmetry': 'asym',
                'indv_cost'     : 'cost',
                'mutR_type'     : 'muTy',
                'mutR_size'     : 'muSi',
                'mutR_frac'     : 'muFr',
                'indv_migrR'    : 'migR',
                'indv_mutR'     : 'mutR',
                'indv_K'        : 'kInd',
                'K_grp'         : 'kGrp',
                'K_tot'         : 'kTot',
                'offspr_sizeInit':'siIn',
                'offspr_fracInit':'frIn',
                'indv_tau'      : 'tInd',
                'replicate_idx' : 'repN'}

parListName = ['indv_NType',
               'indv_migrR',
               'gr_SFis',
               'replicate_idx']

#create list of main parameters to scan, each will run as separate SLURM job
modelParListGlobal = [x for x in itertools.product(mode_vec, nType_vec, range(nReplicate))]

#report number of array jobs needed
def report_number_runs():
    return len(modelParListGlobal)

# create parameter list for current job
def create_model_par_list_local(runIdx):
    #create model paremeter list for current parameter scan
    modelParListLocal = []
    #get current global parameters
    curGlobalPar = modelParListGlobal[runIdx]
    for offspr_size in offspr_size_Vec:
        for offspr_frac in offspr_frac_Vec:
            inBounds = offspr_frac >= offspr_size and \
                    offspr_frac <= (1 - offspr_size)

            if inBounds:
                settings = {'gr_SFis'        : mode_set[0, curGlobalPar[0]],
                            'indv_migrR'     : mode_set[1, curGlobalPar[0]],
                            'indv_NType'     : curGlobalPar[1],
                            'replicate_idx'  : curGlobalPar[2],
                            'offspr_size'    : offspr_size,
                            'offspr_frac'    : offspr_frac,
                            }
                curPar = util.set_model_par(model_par_def, settings)
                modelParListLocal.append(curPar)

    return modelParListLocal

#run mutational meltdown scan
def run_meltdown_scan(model_par):
    #input parameters to store
    parList = ['run_idx','replicate_idx',
               'indv_NType', 'indv_asymmetry', 'indv_cost',
               'indv_mutR','indv_migrR', 'gr_SFis', 'gr_CFis',
               'offspr_size','offspr_frac',
               'indv_K', 'K_tot',
               'delta_indv', 'delta_tot', 'delta_size', 'delta_grp', 'K_grp']

    stateVar = ['maxMu', 'NTot','fCoop','NGrp']

    dTypeList = [(x, 'f8') for x in parList] + [(x, 'f8') for x in stateVar]
    dType = np.dtype(dTypeList)

    outputMat = np.full(1, np.nan, dType)
    for par in parList:
        outputMat[par] = model_par[par]

    hasMeltdown = True
    idx = 0
    #reduce mutation rate till community can survive
    while hasMeltdown:
        model_par['indv_mutR'] = mu_Vec[idx]
        output = mls.run_model_steadyState_fig(model_par)

        #if community survives, found max mutation burden
        if output['NTot'][-1] > 0:
            hasMeltdown = False
            outputMat['maxMu'] = model_par['indv_mutR']
            outputMat['NTot'] = output['NTot_mav']
            outputMat['fCoop'] = output['fCoop_mav']
            outputMat['NGrp'] = output['NGrp_mav']

        else:
            idx += 1

        #end when lowest possible mutation rate has been reached
        if idx >= mu_Vec.size:
            hasMeltdown = False

    return outputMat



# run model code
def run_model(runIdx, folder):
    #get model parameters to scan
    modelParListLocal = create_model_par_list_local(runIdx)

    # run model, use serial processing
    dfList = []
    for par in modelParListLocal:
        output_matrix = run_meltdown_scan(par)
        #convert to pandas data frame, add to list
        dfList.append(pd.DataFrame.from_records(output_matrix))

    #create output name
    parName = ['_%s%.0g' %(parNameAbbrev[x], modelParListLocal[0][x]) for x in parListName]
    parName = ''.join(parName)
    fileNamePkl = folder + mainName + parName + '.pkl'

    #merge results in single data frame
    df = pd.concat(dfList, axis=0, ignore_index=True, sort=True)
    #store results on disk
    df.to_pickle(fileNamePkl)

    return None

#run parscan
if __name__ == "__main__":
    runIdx = sys.argv[1]
    folder = sys.argv[2]
    runFolder = folder + '/home/' + mainName + '/'

    runIdx = int(runIdx)
    runFolder = '/scicore/home/jenal/vanvli0000/home/NDD_mutMelt/'
    if not os.path.exists(runFolder):
        try:
            os.mkdir(runFolder)
        except:
            print('skip folder creation')
    run_model(runIdx, runFolder)
