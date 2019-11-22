"""
Created on Tue Oct 21 2019

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

import itertools
import MlsGroupDynamics_main as mls
import datetime
from joblib import Parallel, delayed, parallel_backend
import numpy as np

"""============================================================================
Define parameters
============================================================================"""

numCore = 22 #number of cores to run code on
numThread = 2 #number o threads per core
#where to store output?
mainName = 'vanVliet_scanFrac'

#setup variables to scan
offspr_fracVec = np.arange(0.1, 0.9, 0.025)
gr_SfissionVec = np.array([0, 1./100])
gr_SextinctVec = np.array([0, -1./5, -1./10, -1./50])

nSim = offspr_fracVec.size * gr_SfissionVec.size * gr_SextinctVec.size

#set other parameters
model_par = {
    # solver settings
    "maxT":             3000,  # total run time
    "minT":             400,   # min run time
    "sampleInt":        1,     # sampling interval
    "mav_window":       200,   # average over this time window
    "rms_window":       200,   # calc rms change over this time window
    "rms_err_trNCoop":  2E-2,  # when to stop calculations
    "rms_err_trNGr":    0.1,  # when to stop calculations
    # settings for initial condition
    "init_groupNum":    50,  # initial # groups
    # initial composition of groups (fractions)
    "init_fCoop":       1,
    "init_groupDens":   50,  # initial total cell number in group  
    # settings for individual level dynamics
    "indv_NType":       2,
    "indv_cost":        0.05,  # cost of cooperation
    "indv_K":           100,  # total group size at EQ if f_coop=1
    "indv_mutationR":   1E-3,  # mutation rate to cheaters
    # difference in growth rate b(j+1) = b(j) / asymmetry
    "indv_asymmetry":    1,
    # setting for group rates
    'gr_Sfission':       0.,    # fission rate = (1 + gr_Sfission * N)/gr_tau
    'gr_Sextinct':      0.,    # extinction rate = (1 + gr_Sextinct * N)*gr_K/gr_tau
    'gr_K':             100,   # total carrying capacity of cells
    'gr_tau':           100,   # relative rate individual and group events
    # settings for fissioning
    'offspr_size':      0.1,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5    # offspr_size < offspr_frac < 1-offspr_size'
}

#setup name to save files
parName = '_off_size%.2g_cost%.0g_indvK%.0e_grK%.0g_sFis%.0g_sExt%.0g' % (
    model_par['offspr_size'], model_par['indv_cost'], 
    model_par['indv_K'], model_par['gr_K'],
    model_par['gr_Sfission'], model_par['gr_Sextinct'])
dataFileName = mainName + parName 


"""============================================================================
Define functions
============================================================================"""
#set model parameters for fission mode
def set_fission_mode(model_par, offspr_frac, gr_Sfission, gr_Sextinct):
    #copy model par (needed because otherwise it is changed in place)
    model_par_local = model_par.copy()
    model_par_local['offspr_frac'] = offspr_frac
    model_par_local['gr_Sfission'] = gr_Sfission
    model_par_local['gr_Sextinct'] = gr_Sextinct


    return model_par_local

# run model
def run_model(model_par, dataFileName):
    #create model paremeter list for all valid parameter range
    # *x unpacks variables stored in tuple x e.g. if x = (a1,a2,a3) than f(*x) = f(a1,a2,a3)
    # itertools.product creates all possible combination of parameters
    modelParList = [set_fission_mode(model_par, *x)
                   for x in itertools.product(*(offspr_fracVec, gr_SfissionVec, gr_SextinctVec))]
    
    #modelParList = [set_fission_mode(model_par, x) for x in offspr_fracVec]
    modelParList[-2*numCore:]
    # run model, use parallel cores 
    nJobs = min(len(modelParList), numCore)
    print('starting with %i jobs' % len(modelParList))

    with parallel_backend("loky", inner_max_num_threads=numThread):
        results = Parallel(n_jobs=nJobs, verbose=10, timeout=1.E8)(
            delayed(mls.single_run_finalstate)(par) for par in modelParList)
    
    np.savez(dataFileName, results=results,
             offspr_fracVec = offspr_fracVec,
             gr_SfissionVec = gr_SfissionVec,
             gr_SextinctVec = gr_SextinctVec,
             modelParList = modelParList, date=datetime.datetime.now())
    
    return None

def run_setGroupRate(gr_Sfission, gr_Sextinct):    
    model_par['gr_Sfission'] = gr_Sfission
    model_par['gr_Sextinct'] = gr_Sextinct
    
    #setup name to save files
    parName = '_off_size%.2g_cost%.0g_indvK%.0e_grK%.0g_sFis%.0g_sExt%.0g' % (
        model_par['offspr_size'], model_par['indv_cost'], 
        model_par['indv_K'], model_par['gr_K'],
        model_par['gr_Sfission'], model_par['gr_Sextinct'])
    dataFileName = mainName + parName     
    
    run_model(model_par, dataFileName)
    
    return None


#run parscan and make figure
if __name__ == "__main__":
    run_model(model_par, dataFileName)

