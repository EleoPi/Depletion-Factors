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


#spatial unit definition----------------------------------------------------

# def set_spatial_unit(filename, var_name):

#     dataset1 = Dataset(filename)
#     print(dataset1)
#     basin = module.spatial_unit(dataset1.variables[var_name][:])
#     pointer_array = basin.index_filter()
    
#     return basin,pointer_array

# #pointer_array = np.load(Input.outputDir +'/'+ Input.fn_filter_catchmentsinecoregions, allow_pickle = 1) 
 
def aggregate_mean(basin, pointer_array):
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

#groundwater head-----------------------------------------------------------
#we have to do it manually because the gwd file is to big to load in the memory.
    d = Dataset(Input.inputDir + '/' + Input.fn_ground_water_depth)
    gwd = d.variables[Input.var_groundwater_depth]
    times = d.variables["time"][:]
    # lat = d.variables["lat"][:]
    # lon = d.variables["lon"][:]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)


#aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    gwh_aggregated_mean = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        #s =  module.calculate_groundwater_head(gwd, dem, t)
        s = dem - gwd[t,:,:]
        for k in range(n_spatial_unit):
            gwh_aggregated_mean[k,t] = np.mean(s[pointer_array[k][0],pointer_array[k][1]])
    
    gwh_aggregated_mean = ma.masked_where(isnan(gwh_aggregated_mean), gwh_aggregated_mean)
    
    ID = ma.getdata(np.unique(basin)[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'test_groundwater_head_mean_' + Input.name_timeperiod +'_'+ Input.name_scale, gwh_aggregated_mean[:-1], ID, times, 'groundwater head', 'month', 'm') 

    # world = np.mean(gwh_aggregated_mean, axis = 0)#should be weighted by the area of the cell
    # fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.plot(world, label='groundwater head')  # Plot some data on the axes.
    # ax.set_xlabel('month')  # Add an x-label to the axes.
    # ax.set_ylabel('m')  # Add a y-label to the axes.
    # ax.set_title("World groundwater head "+Input.name_timeperiod)  # Add a title to the axes.
    # ax.legend()  # Add a legend.
    
    return gwh_aggregated_mean

def aggregate_wavg(basin, pointer_array):
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

#groundwater head-----------------------------------------------------------
#we have to do it manually because the gwd file is to big to load in the memory.
    d = Dataset(Input.inputDir + '/' + Input.fn_ground_water_depth)
    gwd = d.variables[Input.var_groundwater_depth]
    times = d.variables["time"][:]
    # lat = d.variables["lat"][:]
    # lon = d.variables["lon"][:]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)


#aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    gwh_aggregated_mean = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        #s =  module.calculate_groundwater_head(gwd, dem, t)
        s = dem - gwd[t,:,:]
        for k in range(n_spatial_unit):
            gwh_aggregated_mean[k,t] = np.average(s[pointer_array[k][0],pointer_array[k][1]], weights = area[pointer_array[k][0],pointer_array[k][1]])
    
    gwh_aggregated_mean = ma.masked_where(isnan(gwh_aggregated_mean), gwh_aggregated_mean)
    
    ID = ma.getdata(np.unique(basin)[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'test_groundwater_head_wavg_' + Input.name_timeperiod +'_'+ Input.name_scale, gwh_aggregated_mean[:-1], ID, times, 'groundwater head', 'month', 'm') 

    world = np.mean(gwh_aggregated_mean, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world, label='groundwater head')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('m')  # Add a y-label to the axes.
    ax.set_title("World groundwater head "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return gwh_aggregated_mean

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
    
    module.new_map_netcdf(Input.outputDir +'/'+"test_groundwater_head_variation_" + Input.name_timeperiod + '_' + Input.name_scale, map_var, "groundwater head", "m", lat, lon)
          
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
    
    aggregated_yr_mask = ma.masked_where(isnan(aggregated_yr) ==1, aggregated_yr)
    #mean of temporal gradient over the last 10 years.
    grad = np.mean(np.gradient(aggregated_yr_mask, axis = 1)[:, -10:], axis = 1)
    
    map_grad = module.make_map(grad, pointer_array, spatial_unit)
    
    map_grad_mask = ma.masked_where(isnan(map_grad), map_grad)
    
    plt.matshow(map_grad_mask)
    
    module.new_map_netcdf(Input.outputDir +'/'+ "test_groundwater_head_gradient_"+ Input.name_timeperiod + '_' + Input.name_scale, map_grad_mask, "groundwater head gradient", "m/yr", lat, lon)

    return map_grad, grad