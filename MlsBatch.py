#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:58:05 2020

Code runs multiple 2D parameter space scans, each run is stored on disk independently
Results can be plotted using plotParScan

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""
import MlsGroupDynamics_scan2D as mls
import numpy as np
import MlsGroupDynamics_utilities as util

""" 
SET SETTINGS
"""
#SET mainName is appended to file name
mainName = 'evol2D_March6'
#SET number of cores to use
numCore = 10;
#SET group fission rates to scan
gr_Sfission_Vec = np.array([4])
#SET parName and par0Vec to scan over any parameter of choice
par0Name = 'indv_tau'
par0Vec = np.array([1, 0.1, 0.01])
#SET parameter space to scan
offspr_sizeVec = np.arange(0.01, 0.5, 0.017)
offspr_fracVec = np.arange(0.01, 1, 0.035) 

#SET Population size
K_tot_def = 20000

#SET Model default settings
model_par = {
    #time and run settings
    "maxT":             10000,  # total run time
    "maxPopSize":       100000,  #stop simulation if population exceeds this number
    "minT":             200,    # min run time
    "sampleInt":        1,      # sampling interval
    "mav_window":       400,    # average over this time window
    "rms_window":       400,    # calc rms change over this time window
    "rms_err_trNCoop":  1E-1,   # when to stop calculations
    "rms_err_trNGr":    5E-1,   # when to stop calculations
    # settings for initial condition
    "init_groupNum":    10,     # initial # groups
    "init_fCoop":       1,
    "init_groupDens":   50,     # initial total cell number in group
    # settings for individual level dynamics
    # complexity
    "indv_NType":       2,
    "indv_asymmetry":   1,      # difference in growth rate b(j+1) = b(j) / asymmetry
    # mutation load
    "indv_cost":        0.01,  # cost of cooperation
    "indv_migrR":       0,   # mutation rate to cheaters
    # set mutation rates
    'indv_mutR':        1E-3,
    'indv_tau':         0.1,
    # group size control
    "indv_K":           100,     # total group size at EQ if f_coop=1
    "delta_indv":       1,      # zero if death rate is simply 1/k, one if death rate decreases with group size
    # setting for group rates
    # fission rate
    'gr_CFis':          1/100,
    'gr_SFis':          0,
    # extinction rate
    'delta_grp':        0,      # exponent of denisty dependence on group #
    'K_grp':            0,    # carrying capacity of groups
    'delta_tot':        1,      # exponent of denisty dependence on total #indvidual
    'K_tot':            5000,   # carrying capacity of total individuals
    'delta_size':       0,      # exponent of size dependence
    # initial settings for fissioning
    'offspr_size':      0.25,  # offspr_size <= 0.5 and
    'offspr_frac':      0.5,  # offspr_size < offspr_frac < 1-offspr_size'
    # extra settings
    'run_idx':          1,
    'perimeter_loc':    0
    }
  
          
def run_batch():
    """[Runs batch of 2D parameter scans]
    
    Returns:
        None
    """
    for gr_Sfission in gr_Sfission_Vec:
        for par0 in par0Vec:
            if gr_Sfission == 0:
                K_tot = K_tot_def * 6
            else:
                K_tot = K_tot_def
                
            settings = {'gr_SFis' : gr_Sfission,
                        par0Name  : par0,
                        'K_tot'   : K_tot}
                        
            modelParCur = util.set_model_par(model_par, settings)
            _ = mls.run_model(mainName, modelParCur, numCore, 
                                      offspr_sizeVec, offspr_fracVec)
            
    return None

#run parscan and make figure
if __name__ == "__main__":
    run_batch()
    

