# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 08:54:10 2021

@author: easpi
"""


import os
from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy
import numpy as np
from numpy import isnan, ma
import tifffile
from matplotlib import pyplot as plt
import tifffile
# where did you put the modules
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')
import module
import Input
os.chdir(Input.inputDir)





def index_filter(spatial_unit):
    """
    Parameters
    ----------
    spatial_unit : TYPE array (lat,lon)
        DESCRIPTION. array containing the ID of the catchments starting with ID = 0

    Returns
    -------
    index_filter : TYPE list of lists
    idlsit : TYPE array where the ID of the catchment is stored in the same position  as index_filter row number
        DESCRIPTION. the list contains for each catchmetn ID, the list of latitude index in position 0 and the list of longitude indexes for position 1

    """
    index_filter = []
    # select ID of spatial units present in the map
    idlist = np.unique(spatial_unit)
    for n in idlist:
        a = np.where(spatial_unit == n)
        index_filter.append(a)
    return index_filter, idlist



