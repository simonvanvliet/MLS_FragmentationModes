#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:54:31 2020

@author: simonvanvliet
vanvliet@zoology.ubc.ca
"""

DefName = 'Default'

def printName(name=DefName):
    print(name)
    return None

printName()

printName('Something else')