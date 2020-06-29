#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:33:00 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

import numpy as np
import pandas as pd

df = pd.DataFrame.from_records(output)

df.to_pickle("test.pkl")