# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:09:44 2021

@author: easpi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:32:31 2021
the purpose of this main program is to calculate the stressor variation and export results to netcdf.

@author: easpi
"""


import os
from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy
import numpy as np
from numpy import isnan, ma, load
from matplotlib import pyplot as plt
from scipy import stats
import tifffile
#import time


#os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')#where did you put the modules


import Input
import module

#set directories

# input directory
#os.chdir(Input.inputDir)


#spatial unit definition----------------------------------------------------

# def set_spatial_unit(filename, var_name):

#     dataset1 = Dataset(filename)
#     print(dataset1)
#     basin = module.spatial_unit(dataset1.variables[var_name][:])
#     pointer_array = basin.index_filter()
    
#     return basin,pointer_array

# #pointer_array = np.load(Input.outputDir +'/'+ Input.fn_filter_catchmentsinecoregions, allow_pickle = 1) 
 


def aggregate(idlist,pointer_array):
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit
        correspondence between catchment ID and spatial_unit row is does using the idlist 

    Returns timeseries of the groundwater head average over each spatial unit for all timesteps 
    -------
    None.

    '''

#the flow accumulation file is used to identify the outlet point
    flowacc = tifffile.imread(Input.inputDir + '/' + 'flowAcc_5min.tif')
    flowacc = ma.masked_equal(flowacc, -2147483647)

    d = Dataset(Input.inputDir + '/' + Input.fn_discharge)
    #area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    print(d)
    discharge = d.variables["discharge"]#m3/s
    times = d.variables["time"][:]#month
    #lat = d.variables["latitude"][:]
    #lon = d.variables["longitude"][:]

    d1 = Dataset(Input.inputDir + '/' + Input.fn_discharge_natural)
    discharge_natural = d1.variables['discharge']

    
#aggrgeate 
    ntime, nlat, nlon = discharge.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        temp = discharge[t,:,:]-discharge_natural[t,:,:]
        for k in range(n_spatial_unit):
            coord = np.argmax(flowacc[pointer_array[k][0],pointer_array[k][1]], axis=None)#returns the index of the max value of the flattenned array.
            s_aggregated[k,t] = np.ravel(temp[pointer_array[k][0],pointer_array[k][1]])[coord]#select the value at the coordinate point
            #s_aggregated[k,t] = np.max(temp[pointer_array[k][0],pointer_array[k][1]])
    
    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)
    
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+ 'discharge_outlet_human-natural-rev'  + Input.name_timeperiod +'_'+ Input.name_scale, s_aggregated[:-1,:], ID, times, 'discharge flow', 'month', 'm3/s') 

    world_Q = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world_Q, label='human - natural')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('m3/s')  # Add a y-label to the axes.
    ax.set_title("World discharge at outlet 1960/2010")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated

def aggregate_human(idlist,pointer_array):
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit
        correspondence between catchment ID and spatial_unit row is does using the idlist 

    Returns timeseries of the groundwater head average over each spatial unit for all timesteps 
    -------
    None.

    '''

#the flow accumulation file is used to identify the outlet point
    flowacc = tifffile.imread(Input.inputDir + '/' + 'flowAcc_5min.tif')
    flowacc = ma.masked_equal(flowacc, -2147483647)

    d = Dataset(Input.inputDir + '/' + Input.fn_discharge)
    #area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    print(d)
    discharge = d.variables["discharge"]#m3/s
    times = d.variables["time"][:]#month
    #lat = d.variables["latitude"][:]
    #lon = d.variables["longitude"][:]

    #d1 = Dataset(Input.inputDir + '/' + Input.fn_discharge_natural)
    #discharge_natural = d1.variables['discharge']

    
#aggrgeate 
    ntime, nlat, nlon = discharge.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        temp = discharge[t,:,:]
        for k in range(n_spatial_unit):
            coord = np.argmax(flowacc[pointer_array[k][0],pointer_array[k][1]], axis=None)#returns the index of the max value of the flattenned array.
            s_aggregated[k,t] = temp[pointer_array[k][0],pointer_array[k][1]][coord]#select the value at the coordinate point
            #s_aggregated[k,t] = np.max(temp[pointer_array[k][0],pointer_array[k][1]])
    
    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)
    
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+ 'discharge_outlet_human_'  + Input.name_timeperiod +'_'+ Input.name_scale, s_aggregated[:-1,:], ID, times, 'discharge flow', 'month', 'm3/s') 

    world_Q = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world_Q, label='q human')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('m3/s')  # Add a y-label to the axes.
    ax.set_title("World discharge at outlet 1960/2010")  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated
