#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021-07-13

Code for SI FIGURE
- for archetype fragmentation modes we run simulations and extract group size and cooperator frequency


@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

============================================================================
Run Model and plot results
============================================================================"""

import sys
sys.path.insert(0, '..')

#load code
from mainCode import MlsGroupDynamics_main as mls
from mainCode import MlsGroupDynamics_utilities as util
import pandas as pd
import numpy as np
import itertools
import os

"""============================================================================
SET MODEL SETTINGS
============================================================================"""


#SET OUTPUT FILENAME
mainName = 'groupPropSteadyState'

#SET fragementation modes to scan
init_Aray = np.array([[0.05,0.95],[0.05,0.05],[0.5,0.5]])
numInit = init_Aray.shape[0]

indv_mutR_vec = np.array([0.001, 0.01, 0.025, 0.05, 0.1])
indv_NType_vec = np.array([1, 2])

#SET nr of replicates
nReplicate = 10

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
        "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
        # setting for group rates
        # fission rate
        'gr_CFis':          0.05,
        'gr_SFis':          0,     # measured in units of 1 / indv_K
        'grp_tau':          1,     # constant multiplies group rates
        # extinction rate
        'delta_grp':        0,      # exponent of density dependence on group #
        'K_grp':            0,      # carrying capacity of groups
        'delta_tot':        1,      # exponent of density dependence on total #individual
        'K_tot':            1E5,      # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      0.5,  # offspr_size <= 0.5 and
        'offspr_frac':      0.5,    # offspr_size < offspr_frac < 1-offspr_size'
        # extra settings
        'run_idx':          1,
        'replicate_idx':    1,
        'perimeter_loc':    0
    }


"""============================================================================
CODE TO MAKE FIGURE
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
                'replicate_idx' : 'repN',
                'offspr_size'   : 's',
                'offspr_frac'   : 'n'}

parListName = ['offspr_size',
               'offspr_frac',
               'replicate_idx',
               'indv_mutR',
               'indv_NType']

#create list of main parameters to scan, each will run as separate SLURM job
modelParListGlobal = [x for x in itertools.product(range(numInit), range(nReplicate), indv_mutR_vec, indv_NType_vec)]

#report number of array jobs needed
def report_number_runs():
    return len(modelParListGlobal)

# run model code
def run_model(runIdx, folder):
    #get model parameters to scan

    curGlobalPar = modelParListGlobal[runIdx]
    settings = {'offspr_size'    : init_Aray[curGlobalPar[0]][0],
                'offspr_frac'    : init_Aray[curGlobalPar[0]][1],
                'replicate_idx'  : curGlobalPar[1],
                'indv_mutR'      : curGlobalPar[2],
                'indv_NType'     : curGlobalPar[3]
                }
    par = util.set_model_par(model_par_def, settings)

    _, _, _, grSizeVec, fCoop_group = mls.run_model(par)

    frag_mode_idx_vec   = np.ones_like(grSizeVec) * curGlobalPar[0]
    offspr_size_vec     = np.ones_like(grSizeVec) * par['offspr_size']
    offspr_frac_vec     = np.ones_like(grSizeVec) * par['offspr_frac']
    replicate_idx_vec   = np.ones_like(grSizeVec) * par['replicate_idx']
    indv_mutR_vec       = np.ones_like(grSizeVec) * par['indv_mutR']
    indv_NType_vec      = np.ones_like(grSizeVec) * par['indv_NType']


    data = {'frag_mode_idx'     : frag_mode_idx_vec,
            'offspr_size'       : offspr_size_vec,
            'offspr_frac'       : offspr_frac_vec,
            'replicate_idx'     : replicate_idx_vec,
            'indv_mutR'         : indv_mutR_vec,
            'indv_NType'        : indv_NType_vec,
            'group_size'        : grSizeVec,
            'coop_freq'         : fCoop_group}

    df = pd.DataFrame(data=data)

    parName = ['_%s%.0g' %(parNameAbbrev[x], par[x]) for x in parListName]
    parName = ''.join(parName)
    fileNamePkl = folder + mainName + parName + '.pkl'
    #store results on disk
    df.to_pickle(fileNamePkl)

    return None

#run parscan
if __name__ == "__main__":
    runIdx = sys.argv[1]
    folder = sys.argv[2]
    runFolder = folder + '/home/' + mainName + '/'

    runIdx = int(runIdx)
    runFolder = '/scicore/home/jenal/vanvli0000/home/MLSGroupProp/'
    if not os.path.exists(runFolder):
        try:
            os.mkdir(runFolder)
        except:
            print('skip folder creation')
    run_model(runIdx, runFolder)
