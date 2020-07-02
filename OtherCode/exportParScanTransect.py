#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 2019

Plots output of MlsGroupDynamics_scanTransects


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

#set name of file to load
fileName = 'transect_March1_cost0.01_migR0_kInd1e+02_kGrp0_kTot3e+04_asym1_dInd1_dGrp0_dTot1_dSiz0_fisC0.01.npz'

#fileName = 'transect_Feb28_cost0.01_migR0_kInd1e+02_kGrp0_kTot3e+04_asym1_dInd1_dGrp0_dTot1_dSiz0_fisC0.01_newFormat.npz'

#set Folder
pathLoad = Path("./Data/")
pathSave = Path("/Users/simonvanvliet/ownCloud/MLS_GroupDynamics_shared/Figures/Transects/")


"""============================================================================
Define functions
============================================================================"""


    #set folders of data

#remove extension if needed
if fileName[-4:] == '.npz':
    fileName = fileName[:-4]
 
#load data
loadName = pathLoad / (fileName + '.npz')
data_file = np.load(loadName, allow_pickle=True)
results = data_file['results']
par0_vec = data_file['par0_vec']
par1_vec = data_file['par1_vec']
par2_vec = data_file['par2_vec']
parNames = data_file['parNames']
modelParList = data_file['modelParList']
perimeter_loc_vec = data_file['perimeter_loc_vec']
data_file.close()

Output, _, _ = zip(*results)

stateVar = ['NTot', 'NCoop', 'fCoop',
        'NGrp', 'groupSizeAv', 'groupSizeMed']

#input parameters to store
parList = ['indv_NType', 'indv_cost', 'indv_K', 'indv_mutR', 'indv_migrR', 'indv_asymmetry', 'delta_indv',
           'gr_SFis', 'gr_CFis', 'K_grp', 'K_tot', 'delta_grp', 'delta_tot', 'delta_size',
           'offspr_size', 'offspr_frac', 'run_time']

# init output matrix
dTypeList1 = [(x, 'f8') for x in stateVar]
dTypeList2 = [(x+'_mav', 'f8') for x in stateVar]
dTypeList3 = [(x, 'f8') for x in parList]
dTypeList = dTypeList1 + dTypeList2 + dTypeList3 + [('perimeter_loc', 'f8')]
dType = np.dtype(dTypeList)
statData = np.zeros(len(Output), dType)

# store final state
i = 0
for data in Output:
    OffsprFrac = data['offspr_frac']
    OffsprSize = data['offspr_size']
    statData['perimeter_loc'][i] = OffsprSize if OffsprFrac >=0.5 else (1-OffsprSize)
    for var in stateVar:
        statData[var][i] = data[var]
        var_mav = var + '_mav'
        statData[var_mav][i] = data[var_mav]
    for par in parList:
        statData[par][i] = data[par]
    i += 1

parList = modelParList[0]

#convert to 1d coulmn vector
statData = np.reshape(statData,(-1,1))   

#save main data file
dataName = pathSave / (fileName + '_data' + '.csv')
header=','.join(statData.dtype.names)
np.savetxt(dataName, statData.view(np.float64), delimiter=',', header=header, comments='')

#save meta data file
metaDataName = pathSave / (fileName + '_metaData' + '.txt')
fid = open(metaDataName, 'w') 

header = 'Data for %s \n' % fileName
fid.write(header)
perimeter_loc_vec
fid.write('perimeter_loc_vec values are:\n')
fid.write(np.array2string(perimeter_loc_vec) + '\n')
fid.write('par0 = %s values are:\n' % parNames[0])
fid.write(np.array2string(par0_vec) + '\n')
fid.write('par1 = %s values are:\n' % parNames[1])
fid.write(np.array2string(par1_vec) + '\n')
fid.write('par2 = %s values are:\n' % parNames[2])
fid.write(np.array2string(par2_vec) + '\n')
fid.write('\nOther Model Parameters are:\n')

for key, val in parList.items():
    fid.write('%s = %f\n' % (key, val))

fid.close()

header=','.join(statData.dtype.names)
np.savetxt(dataName, statData.view(np.float64), delimiter=',', header=header, comments='')

