#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

results = np.load('Data_FigSX_Pichugin_temp.npy')


#convert to pandas dataframe and export
fileNameFull = fileName + '.pkl'
outputComb = np.reshape(results, (-1))
df = pd.DataFrame.from_records(outputComb)
df.to_pickle(fileNameFull)