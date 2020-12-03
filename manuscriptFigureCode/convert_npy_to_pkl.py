#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#opens numpy file with name fileName_temp.npy and converts to pandas dataframe
#dataframe stored with name fileName.pkl

import pandas as pd
import numpy as np

fileName = 'mutRvsPopSizeWComplexity'

#load data
fileNameTemp = fileName + '_temp' + '.npy'
results = np.load(fileNameTemp, allow_pickle=True)

#convert to pandas dataframe and export
fileNameFull = fileName + '.pkl'
outputComb = np.hstack(results)
df = pd.DataFrame.from_records(outputComb)
df.to_pickle(fileNameFull)