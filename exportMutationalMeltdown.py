#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Plots output of MlsGroupDynamics_scanMutationalMeltdown

Last Update Oct 23 2019

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

"""

"""============================================================================
Import dependencies & define global constants
============================================================================"""
import numpy as np
from pathlib import Path

#set name of file to load (no extension)
fileName = 'MutationMeltdown_March28_kInd1e+02_fisC0.01_kTot3e+04_asym1.npz'

#data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
pathLoad = Path(".")
pathSave = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/MutationalMeltdown/")


"""============================================================================
Define functions
============================================================================"""

    
#remove extension if needed
if fileName[-4:] == '.npz':
    fileName = fileName[:-4]

#set folders of data
figureName = pathSave / (fileName + '.pdf')

#load data
loadName   = pathLoad / (fileName + '.npz')
data_file  = np.load(loadName, allow_pickle=True)
statData   = data_file['statData']
maxMu      = data_file['maxMu']
NTot       = data_file['NTot']
NCoop      = data_file['NCoop']
NGrp       = data_file['NGrp']
numRepeat  = data_file['numRepeat']
offsprSize = data_file['offsprSize'] 
offsprFrac = data_file['offsprFrac']
mutR       = data_file['mutR']
mode_vec   = data_file['mode_vec']
par0_vec   = data_file['par0_vec']
par1_vec   = data_file['par1_vec']
mode_set   = data_file['mode_set']
modeNames  = data_file['modeNames']
parNames   = data_file['parNames']
parList    = data_file['parList']
data_file.close()
parList = parList[0]


#combine data
avMaxMu = np.reshape(np.nanmean(maxMu,axis=1), (-1, 1))
avNTot  = np.reshape(np.nanmean(NTot,axis=1), (-1, 1))
avNCoop = np.reshape(np.nanmean(NCoop,axis=1), (-1, 1))
avNGrp  = np.reshape(np.nanmean(NGrp,axis=1), (-1, 1))

parData = statData.view(np.float64)
data = np.concatenate((avMaxMu,avNTot,avNCoop,avNGrp,parData), axis=1)
dataNames = ('av_maxMu', 'av_NTot_mav','av_NCoop_mav','av_NGrp_mav') + statData.dtype.names

   
#save main data file
dataName = pathSave / (fileName + '_data' + '.csv')
header=','.join(dataNames)
np.savetxt(dataName, data, delimiter=',', header=header, comments='')

#save meta data file
metaDataName = pathSave / (fileName + '_metaData' + '.txt')
fid = open(metaDataName, 'w') 

header = 'Data for %s \n' % fileName
fid.write(header)
fid.write('numRepeat = %i\n' % numRepeat)
fid.write('offsprSize vector values are:\n')
fid.write(np.array2string(offsprSize) + '\n')
fid.write('offsprFrac vector values are:\n')
fid.write(np.array2string(offsprFrac) + '\n')
fid.write('modeVec index values are:\n')
fid.write(np.array2string(mode_vec) + '\n')
fid.write('modeVec %s values are:\n' % modeNames[0])
fid.write(np.array2string(mode_set[0,:]) + '\n')
fid.write('modeVec %s values are:\n' % modeNames[1])
fid.write(np.array2string(mode_set[1,:]) + '\n')
fid.write('par0 = %s values are:\n' % parNames[0])
fid.write(np.array2string(par0_vec) + '\n')
fid.write('par1 = %s values are:\n' % parNames[1])
fid.write(np.array2string(par1_vec) + '\n')


fid.write('\nOther Model Parameters are:\n')

for key, val in parList.items():
    fid.write('%s = %f\n' % (key, val))

fid.close()

header=','.join(statData.dtype.names)
np.savetxt(dataName, statData.view(np.float64), delimiter=',', header=header, comments='')
