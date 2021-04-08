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

def index_outlet(index_filter, flowacc):
    """
    Parameters
    ----------
    
    """
    index = []
    
    
    for n in idlist:
        a = np.where(spatial_unit == n)
        index_filter.append(a)
    return index_filter, idlist


# open basin delineation: the spatial resolution can be changed here.

# catchment map
# spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
# spatial_unit =pcr2numpy(spatial_unit_map,-2147483648)#nan = -2147483648

# #finer spatial resolution
# spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
# spatial_unit =pcr2numpy(spatial_unit_map,-2147483648)#nan = -2147483648
# spatial_unit_mask = ma.masked_values(spatial_unit, -2147483648)#we mask  nan values

# catchment map correceted by ecoregion
#d = Dataset(Input.inputDir + '/' + Input.fn_catchmentsldd)
#spatial_unit = d.variables["catchment"]
#spatial_unit_mask = ma.masked_values(spatial_unit, 65535)  # we mask  nan values
#spatial_unit_mask.shape

spatial_unit = tifffile.imread(Input.inputDir + '/' + 'basins_5min_valerio.tif')
spatial_unit = ma.masked_equal(spatial_unit, -2147483647)

idlist = np.unique(spatial_unit)


# the pointer is a list of the coordinates indexes for each spatial unit
# recalculate for each delineation map when testing the effect of scale
pointer = index_filter(spatial_unit)

# array containing the coordinates of each spatial unit with data
pointer_array = np.array(pointer[0], dtype=list)  # conversion to array
pointer_array[pointer[1] == 8506][0][0]
pointer_array1 = np.asarray(pointer[0], dtype = list)
pointer_array1[pointer[1] == 8506][0][1]
# list of the spatial unit ids reported in the pointer array

idlist = pointer[1]

idlist == np.unique(spatial_unit)

np.save(Input.outputDir + '/filter_array_catchments_valerio', pointer_array)

#np.save('output/idlist_catchments_ldd_rev1', ma.getdata(idlist))

f = np.load(Input.outputDir + '/' + 'filter_array_catchments_valerio.npy', allow_pickle = True)

np.where(f==pointer_array) #from some reason it is different

#test ok
ID = pointer[2]

ID==idlist

np.where(idlist == 8506)

pointer_array[idlist == 8506]

np.where(spatial_unit == 8506)

f[idlist == 8506]


# to open this filter later, use command np.load

#pointer_array = np.load('output_catchments_in_ecoregions/filter_array_catchment_scale.npy', allow_pickle = True)
# list ordered



#pointer_array = np.load('output/filter_array_catchment_scale_new.npy', allow_pickle = True)
#idlist = np.load('output/idlist_catchment_scale_new.npy')


#pointer_array = np.load('output/filter_array_catchment_scale.npy', allow_pickle = True)
#idlist = np.load('output/idlist_catchment_scale.npy')

#pointer_array = np.load('output/filter_array_catchmentinecoregions_scale.npy', allow_pickle = True)
#idlist = np.load('output/idlist_catchmentinecoregions_scale.npy')






#extraction of outlet of watersheds

flowacc = tifffile.imread(Input.inputDir + '/' + 'flowAcc_5min.tif')
flowacc = ma.masked_equal(flowacc, -2147483647)
plt.imshow(flowacc)

f = np.load(Input.outputDir + '/' + 'filter_array_catchments_ldd_rev1.npy', allow_pickle = True)

# outlet  = []
# for i in f.shape[0]:#for each catchment
#     maxflow = np.max(flowacc[f[i][0],f[i][1]])#flow accumulation values in the catchment i
#     coord = np.where(flowacc = np.max)
#     outlet.apped(coord)


# catchment = flowacc[f[12735][0],f[12735][1]]
# maxflow = np.max(catchment)#flow accumulation values in the catchment i
# coord1 = np.argmax(catchment, axis=None)
# #coordinates of the outlet in the reference of the catchment, not in the qworld! flattenned array!
# maxflow
# catchment[coord1]
# coord1
# catchment
# #test ok. 