#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-07-03

Code for figure 4C
- For the one optimal strategy, we show on x-axis the mutation rate,
and on y-axis the population size, for different levels of complexity

Runs model to steady state for different mutation rates
Outputs population size as well as other data.


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
mainName = 'mutRvsPopSizeWComplexity_NoDensDep'

#SET mutation rates to scan
mutR_vec = 10*np.logspace(-3,-1,20) #np.logspace(-3,-1,15)

#SET number of types to scan
indv_NType_vec = np.array([1,2,3,4])

#SET XY Coordinates in parameter space
offspr_size = 0.1
offspr_frac = 0.9

#SET nr of replicates
nReplicate = 5

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
        'K_tot':            1E4,      # carrying capacity of total individuals
        'delta_size':       0,      # exponent of size dependence
        # settings for fissioning
        'offspr_size':      offspr_size,  # offspr_size <= 0.5 and
        'offspr_frac':      offspr_frac,    # offspr_size < offspr_frac < 1-offspr_size'
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
                'replicate_idx' : 'repN'}

parListName = ['indv_NType',
               'replicate_idx']

#create list of main parameters to scan, each will run as separate SLURM job
modelParListGlobal = [x for x in itertools.product(indv_NType_vec, range(nReplicate))]

#report number of array jobs needed
def report_number_runs():
    return len(modelParListGlobal)

# create parameter list for current job
def create_model_par_list_local(runIdx):
    #create model paremeter list for current parameter scan
    modelParListLocal = []
    #get current global parameters
    curGlobalPar = modelParListGlobal[runIdx]
    for mutR in mutR_vec:
        settings = {'indv_NType'     : curGlobalPar[0],
                    'indv_mutR'      : mutR,
                    'replicate_idx'  : curGlobalPar[1],
                    }
        curPar = util.set_model_par(model_par_def, settings)
        modelParListLocal.append(curPar)

    return modelParListLocal


# run model code
def run_model(runIdx, folder):
    #get model parameters to scan
    modelParListLocal = create_model_par_list_local(runIdx)

    # run model, use serial processing
    dfList = []
    for par in modelParListLocal:
        output_matrix = mls.run_model_steadyState_fig(par)
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
    runFolder = '/scicore/home/jenal/vanvli0000/home/NDD_NvsMut/'
    if not os.path.exists(runFolder):
        try:
            os.mkdir(runFolder)
        except:
            print('skip folder creation')
    run_model(runIdx, runFolder)
