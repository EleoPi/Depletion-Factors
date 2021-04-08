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
from numpy import isnan, ma, trapz, load
from matplotlib import pyplot as plt
from scipy import stats
#import time


#os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')#where did you put the modules


import Input
import module

#set directories

# input directory
#os.chdir(Input.inputDir)


def aggregate(spatial_unit, pointer_array):
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit

    Returns timeseries of the groundwater head average over each spatial unit for all timesteps 
    -------
    None.

    '''

#groundwater head-----------------------------------------------------------
#we have to do it manually because the gwd file is to big to load in the memory.
    d = Dataset(Input.inputDir + '/' + Input.fn_ground_water_depth)
    gwd = d.variables[Input.var_groundwater_depth]
    times = d.variables["time"][:]
    # lat = d.variables["lat"][:]
    # lon = d.variables["lon"][:]


    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)


    sy = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_aquifer_yield), mv = 1e20)
    sy = ma.masked_values(sy, 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    #aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        temp =  (dem - gwd[t,:,:])*sy*area
        for k in range(n_spatial_unit):
            s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])
    
    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)  
    ID = ma.getdata(np.unique(spatial_unit)[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'test_groundwater_storage_' + Input.name_timeperiod +'_'+ Input.name_scale, s_aggregated[:-1], ID, times, 'groundwater storage', 'month', 'm3') 
    
    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='groundwater storage')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World groundwater storage "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated

def aggregate_weight(spatial_unit, pointer_array):#this should not be used becasue the area is already accounted in the ocnversion to m3
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit

    Returns timeseries of the groundwater head average over each spatial unit for all timesteps 
    -------
    None.

    '''

#groundwater head-----------------------------------------------------------
#we have to do it manually because the gwd file is to big to load in the memory.
    d = Dataset(Input.inputDir + '/' + Input.fn_ground_water_depth)
    gwd = d.variables[Input.var_groundwater_depth]
    times = d.variables["time"][:]
    # lat = d.variables["lat"][:]
    # lon = d.variables["lon"][:]


    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)


    sy = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_aquifer_yield), mv = 1e20)
    sy = ma.masked_values(sy, 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    #aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        temp =  (dem - gwd[t,:,:])*sy*area
        for k in range(n_spatial_unit):
            s_aggregated[k,t] = np.average(temp[pointer_array[k][0],pointer_array[k][1]], weights = area[pointer_array[k][0],pointer_array[k][1]])
    
    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)  
    ID = ma.getdata(np.unique(spatial_unit)[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'test_groundwater_storage_weighed_avg' + Input.name_timeperiod +'_'+ Input.name_scale, s_aggregated[:-1], ID, times, 'groundwater storage', 'month', 'm3') 
    
    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='groundwater storage')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World groundwater storage "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated

def calculate_variation(aggregated_mean, spatial_unit, pointer_array):
    '''
    

    Parameters
    ----------
     aggregated_mean : TYPE groundwater aggregated array (spatial_units, time)
        DESCRIPTION.represents the mean GWH for each catchments defined in spatial_unit
    basin : TYPE array
        DESCRIPTION. spatial_unit delineation and clonemap
    pointer_array : TYPE array
        DESCRIPTION.filter array that identified the coordinates of each spatial units

    Returns map of the variation of the groundwater head over the period correpsonding to spatial units, and values for each catchemnts
    -------
    None.

    '''

    d = Dataset(Input.inputDir + '/' + Input.fn_ground_water_depth)
#    gwd = d.variables[Input.var_groundwater_depth]
#    t = d.variables["time"][:]
    lat = d.variables["lat"][:]
    lon = d.variables["lon"][:]
    
#average FF
    var = np.sum(np.diff(aggregated_mean, axis = 1), axis = 1)
    
    map_var = module.make_map(var, pointer_array, spatial_unit)
    
    plt.matshow(map_var, vmin = stats.scoreatpercentile(var,5), vmax = stats.scoreatpercentile(var,95))#ok
    
    module.new_map_netcdf(Input.outputDir +'/'+"test_groundwater_storage_variation_" + Input.name_timeperiod + '_' + Input.name_scale, map_var, "groundwater storage", "m3", lat, lon)
        
    return map_var, var 

def calculate_grad_avg(aggregated_mean, spatial_unit, pointer_array):
    """
    

    Parameters
    ----------
    aggregated_mean : TYPE groundwater aggregated array (spatial_units, time)
        DESCRIPTION.represents the mean GWH for each catchments defined in spatial_unit
    basin : TYPE array
        DESCRIPTION. spatial_unit delineation and clonemap
    pointer_array : TYPE array
        DESCRIPTION.filter array that identified the coordinates of each spatial units

    Returns map of the annual gradient average over the last 10 years in each spatial unit,  gradient values for each spatial unit
    -------
    None.

    """
    d = Dataset(Input.inputDir + '/' + Input.fn_ground_water_depth)
#    gwd = d.variables[Input.var_groundwater_depth]
#    times = d.variables["time"][:]
    lat = d.variables["lat"][:]
    lon = d.variables["lon"][:]
     
    #adjust interval for gradient calculation
    # intervals_year = np.zeros(times.shape[0]//12)
    
    # for i in range(times.shape[0]//12):
    #     intervals_year[i] = times[i+12]-times[i]

    aggregated_yr = module.convert_month_to_year_avg(aggregated_mean)
    
    aggregated_yr_mask = ma.masked_where(isnan(aggregated_yr) == 1, aggregated_yr)
    #mean of temporal gradient over the last 10 years.
    grad = np.mean(np.gradient(aggregated_yr_mask, axis = 1)[:, -10:], axis = 1)
    
    map_grad = module.make_map(grad, pointer_array, spatial_unit)
    
    map_grad_mask = ma.masked_where(isnan(map_grad), map_grad)
    
    plt.imshow(map_grad_mask)
    
    module.new_map_netcdf(Input.outputDir +'/'+ "test_groundwater_storage_gradient_"+ Input.name_timeperiod + '_' + Input.name_scale, map_grad_mask, "groundwater storage gradient", "m3/yr", lat, lon)

    return map_grad, grad