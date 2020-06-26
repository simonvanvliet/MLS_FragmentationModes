#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:33:00 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

dType = np.dtype([
        ('indv_K', 'f8'),
        ('alpha_b', 'f8'),
        ('offspr_size', 'f8'),
        ('offspr_frac', 'f8'),
        ('growthRate', 'f8'),
        ('intercept_fit', 'f8'),
        ('r_value_fit', 'f8'),
        ('p_value_fit', 'f8'),
        ('std_err_fit', 'f8')])
output = np.zeros(1, dType)