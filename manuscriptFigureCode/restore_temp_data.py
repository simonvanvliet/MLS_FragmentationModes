#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020-13-03

Converts temp.npy into proper pandas dataframe
Used to fix bug in data export (now fixed) that occured in initial run

@author: Simon van Vliet & Gil Henriques
Department of Zoology
University of Britisch Columbia
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca
"""

import sys
sys.path.insert(0, '..')

#load code
from mainCode import MlsGroupDynamics_main as mls
import pandas as pd
import numpy as np
import itertools


def convert_data(modelParList, fileName):
    """
    Converts filename_temp.npy data dump to proper panda dataframe
    stored as filename_corrected.pkl

    Parameters
    ----------
    modelParList : List of dictionaries
        List of model parameters used for each run.
    fileName : string
        file name.

    Returns
    -------
    None.

    """
    #get model parameters to scan
    columnNames = [mls.columnNames_steadyState_fig(par) for par in modelParList]
    
    fileNameTemp = fileName + '_temp' + '.npy'
    fileNameFull = fileName + '_corrected.pkl'
    
    results = np.load(fileNameTemp, allow_pickle=True)
    
    dfList = [pd.DataFrame.from_records(r, columns=c) for (r, c) in zip(results, columnNames)]    
    df = pd.concat(dfList, axis=0, ignore_index=True)    
    df.to_pickle(fileNameFull)

    return None