#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:54:31 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
#from numba.types import Tuple, UniTuple
#from numba import jit, f8, i8
#import MlsGroupDynamics_utilities as util
#
#

NGroup = 20
grpMat = 1*np.ones((4,NGroup,100,100))
offspring = 100*np.ones((4,3,100,100))
parrentNew = 101*np.zeros((4,100,100))

for i in range(NGroup):
    grpMat[:,i,:,:] = i
    
eventGroup = 5


def fission_group(parentGroup):   
    #get group properties
    offspr_size, offspr_frac, NCellPar = calc_mean_group_prop(parentGroup)
    NCellPar = int(NCellPar)
       
    #calc number of offspring, draw from Poisson distribution
    # <#offspring> = NCellToOffspr / sizeOfOffspr = offspr_frac / offspr_size
    expectNOffspr = offspr_frac / offspr_size
    # calc num cells per offsring
    NCellPerOffspr = round(offspr_size * NCellPar)
    #calc max num offspring group
    maxNOffspr = int(np.floor(NCellPar / NCellPerOffspr))
    
    #draw number of offspring from truncated Poission distribution
    NOffspr = util.truncated_poisson(expectNOffspr, maxNOffspr)
    NCellToOffspr = NOffspr * NCellPerOffspr
    
    #assign cells to offspring
    if NOffspr > 0:
        matShape = parentGroup.shape
        
        #vector with destination index for all cells
        #initialize to -1: stay with parent
        destinationIdx = np.full(NCellPar, -1)
        #assign indices 0 to N-1 for offspring
        offsprIdx = np.kron(np.arange(NOffspr), np.ones(NCellPerOffspr))
        destinationIdx[0:NCellToOffspr] = offsprIdx
        
        #random shuffle matrix 
        destinationIdx = np.random.permutation(destinationIdx)
        
        #now identify all cells in parent group
        #create vector with type idx, offspr_frac idx, offspr_size idx
        parCellProp = np.ones((NCellPar, 3), dtype=int) #CHNAGE TO I8 FOR NUMBA
        
        #find non zero elements
        ttIDx, ffIdx, ssIdx = np.nonzero(parentGroup)
        #loop all cells in parentgroup and store properties
        idx = 0
        for ii in range(ttIDx.size):
            numCell = parentGroup[ttIDx[ii], ffIdx[ii], ssIdx[ii]]
            while numCell>0:
                parCellProp[idx, 0] = ttIDx[ii]
                parCellProp[idx, 1] = ffIdx[ii]
                parCellProp[idx, 2] = ssIdx[ii]
                numCell -= 1
                idx += 1
    
        #assign cells to offspring
        #init offspring and parremt array
        offspring = np.zeros((matShape[0], NOffspr, matShape[1], matShape[2]))
        parrentNew = np.zeros((matShape[0], matShape[1], matShape[2]))
        
        for cc in range(NCellPar):
            currDest = destinationIdx[cc]
            if currDest == -1: #stays in parrent
                parrentNew[parCellProp[cc,0],
                           parCellProp[cc,1],
                           parCellProp[cc,2]] += 1
            else:
                offspring[parCellProp[cc,0],
                          currDest,
                          parCellProp[cc,1],
                          parCellProp[cc,2]] += 1
    else:
        #nothing happens
        parrentNew = parentGroup
        offspring = np.zeros((0, 0, 0, 0))
                
    return (parrentNew, offspring)