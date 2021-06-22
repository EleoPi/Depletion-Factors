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
    
    d1 = Dataset(Input.fn_ground_water_depth_natural)
    gwd1 = d1.variables[Input.var_groundwater_depth_natural]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    
#aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    gwh_aggregated_mean = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        s = (dem - gwd[t,:,:])-(dem - gwd1[t//12,:,:])#the natural run is year average
        for k in range(n_spatial_unit):
            gwh_aggregated_mean[k,t] = np.mean(s[pointer_array[k][0],pointer_array[k][1]])
    
    gwh_aggregated_mean = ma.masked_where(isnan(gwh_aggregated_mean), gwh_aggregated_mean)
    
    ID = ma.getdata(np.unique(basin)[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'groundwater_head_mean_' + Input.name_timeperiod +'_'+ Input.name_scale, gwh_aggregated_mean[:-1], ID, times, 'groundwater head', 'month', 'm') 

    # world = np.mean(gwh_aggregated_mean, axis = 0)#should be weighted by the area of the cell
    # fig, ax = plt.subplots()  # Create a figure and an axes.
    # ax.plot(world, label='groundwater head')  # Plot some data on the axes.
    # ax.set_xlabel('month')  # Add an x-label to the axes.
    # ax.set_ylabel('m')  # Add a y-label to the axes.
    # ax.set_title("World groundwater head "+Input.name_timeperiod)  # Add a title to the axes.
    # ax.legend()  # Add a legend.
    
    return gwh_aggregated_mean

def aggregate_wavg(idlist, pointer_array):
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
    
    d1 = Dataset(Input.fn_ground_water_depth_natural)
    gwd1 = d1.variables[Input.var_groundwater_depth_natural]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)


#aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    gwh_aggregated_mean = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        #s =  module.calculate_groundwater_head(gwd, dem, t)
        s = (dem - gwd[t,:,:])-(dem - gwd1[t//12,:,:])
        for k in range(n_spatial_unit):
            gwh_aggregated_mean[k,t] = np.average(s[pointer_array[k][0],pointer_array[k][1]], weights = area[pointer_array[k][0],pointer_array[k][1]])
    
    gwh_aggregated_mean = ma.masked_where(isnan(gwh_aggregated_mean), gwh_aggregated_mean)
    
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'groundwater_head_wavg_' + Input.name_timeperiod +'_'+ Input.name_scale, gwh_aggregated_mean[:-1], ID, times, 'groundwater head', 'year', 'm') 

    world = np.mean(gwh_aggregated_mean, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world, label='groundwater head')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('m')  # Add a y-label to the axes.
    ax.set_title("World groundwater head human - natural "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return gwh_aggregated_mean

def aggregate_wavg_human(idlist, pointer_array):
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
    
    #d1 = Dataset(Input.fn_ground_water_depth_natural)
    #gwd1 = d1.variables[Input.var_groundwater_depth_natural]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)


#aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    gwh_aggregated_mean = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(ntime):
        #s =  module.calculate_groundwater_head(gwd, dem, t)
        s = (dem - gwd[t,:,:])
        for k in range(n_spatial_unit):
            gwh_aggregated_mean[k,t] = np.average(s[pointer_array[k][0],pointer_array[k][1]], weights = area[pointer_array[k][0],pointer_array[k][1]])
    
    gwh_aggregated_mean = ma.masked_where(isnan(gwh_aggregated_mean), gwh_aggregated_mean)
    
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'groundwater_head_wavg_human' + Input.name_timeperiod +'_'+ Input.name_scale, gwh_aggregated_mean[:-1], ID, times, 'groundwater head', 'year', 'm') 

    world = np.mean(gwh_aggregated_mean, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world, label='groundwater head')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('m')  # Add a y-label to the axes.
    ax.set_title("World groundwater head human "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return gwh_aggregated_mean
