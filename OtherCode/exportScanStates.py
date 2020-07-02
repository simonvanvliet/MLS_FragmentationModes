#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Last Update Oct 23 2019

Plots output of MlsGroupDynamics_scanStates


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
fileName = 'March28_kInd1e+02_fisC0_kTot1e+04_nTyp1_asym1.npz'

#data_folder = Path(str(Path.home())+"/Desktop/MLS_GroupDynamics-MultipleTypes/Data/")
pathLoad = Path(".")
pathSave = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/scanStates")


"""============================================================================
Define functions
============================================================================"""


#remove extension if needed
if fileName[-4:] == '.npz':
    fileName = fileName[:-4]
        
#load data
loadName   = pathLoad / (fileName + '.npz')
data_file  = np.load(loadName, allow_pickle=True)
results    = data_file['results']
offsprSize = data_file['offsprSize'] 
offsprFrac = data_file['offsprFrac']
par1       = data_file['par1']
par2       = data_file['par2']
par3       = data_file['par3']
parNames   = data_file['parNames']
parList    = data_file['parList']
data_file.close()


# process output
statData, _, _ = zip(*results)
statData = np.vstack(statData)
parList = parList[0]

   
#save main data file
dataName = pathSave / (fileName + '_data' + '.csv')
header=','.join(statData.dtype.names)
np.savetxt(dataName, statData.view(np.float64), delimiter=',', header=header, comments='')

#save meta data file
metaDataName = pathSave / (fileName + '_metaData' + '.txt')
fid = open(metaDataName, 'w') 

header = 'Data for %s \n' % fileName
fid.write(header)
fid.write('offsprSize vector values are:\n')
fid.write(np.array2string(offsprSize) + '\n')
fid.write('offsprFrac vector values are:\n')
fid.write(np.array2string(offsprFrac) + '\n')
fid.write('par1 = %s values are:\n' % parNames[0])
fid.write(np.array2string(par1) + '\n')
fid.write('par2 = %s values are:\n' % parNames[1])
fid.write(np.array2string(par2) + '\n')
fid.write('par3 = %s values are:\n' % parNames[2])
fid.write(np.array2string(par3) + '\n')
fid.write('\nOther Model Parameters are:\n')

for key, val in parList.items():
    fid.write('%s = %f\n' % (key, val))

fid.close()

header=','.join(statData.dtype.names)
np.savetxt(dataName, statData.view(np.float64), delimiter=',', header=header, comments='')
