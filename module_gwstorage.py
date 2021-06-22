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


def aggregate(idlist, pointer_array):
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
    
    #d1 = Dataset(Input.fn_ground_water_depth_natural)
    d1 = Dataset('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/data/groundwaterDepthLayer1_yearavg_natural_output_19602010.nc')
    gwd1 = d1.variables[Input.var_groundwater_depth_natural]
    #times = d1.variables["time"][:]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    sy = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_aquifer_yield), mv = 1e20)
    sy = ma.masked_values(sy, 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    #aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    # for t in range(0,ntime-1):
    #     temp =  (gwd[t+1,:,:] - gwd[t,:,:])*sy*area#here no need to convert yearly to monthly average
    #     for k in range(n_spatial_unit):
    #         s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])
    
    # s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)
    # gws_depletion_human = s_aggregated#monthly
    # gws_depletion_human = ma.masked_equal(gws_depletion_human, 1e20)

    # gws_depletion_human_yr = module.convert_month_to_year_sum(gws_depletion_human)
    
    
    
    # s_aggregated = np.full((n_spatial_unit, gwd1.shape[0]), 1e20)
    # for t in range(0,gwd1.shape[0]-1):
    #     temp =  (gwd1[t,:,:] - gwd1[t-1,:,:])*sy*area#here no need to convert yearly to monthly average
    #     for k in range(n_spatial_unit):
    #         s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])

    # s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)
    # gws_depletion_natural_yr = s_aggregated#yearly
    # gws_depletion_natural_yr = ma.masked_equal(gws_depletion_natural_yr, 1e20)
    
    # total_depletion = gws_depletion_human_yr - gws_depletion_natural_yr
    
    for t in range(ntime):
        temp =  (dem - gwd[t,:,:] - (dem - gwd1[t//12,:,:]))*sy*area#here no need to convert yearly to monthly average
        for k in range(n_spatial_unit):
            s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])
    
    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)  
    
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'groundwater_storage_' + Input.name_timeperiod +'_'+ Input.name_scale, s_aggregated[:-1], ID, times, 'groundwater storage', 'month', 'm3') 
    
    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='human-natural')  # Plot some data on the axes.
    #ax.set_xlabel('yr')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World groundwater storage "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated

def aggregate_human(idlist, pointer_array):
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
    
    #d1 = Dataset(Input.fn_ground_water_depth_natural)
    #d1 = Dataset('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/data/groundwaterDepthLayer1_yearavg_natural_output_19602010.nc')
    #gwd1 = d1.variables[Input.var_groundwater_depth_natural]
    #times = d1.variables["time"][:]
    
    dem = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_dem), mv = 1e20)

    sy = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_aquifer_yield), mv = 1e20)
    sy = ma.masked_values(sy, 1e20)

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    #aggrgeate 
    ntime, nlat, nlon = gwd.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    # for t in range(0,ntime-1):
    #     temp =  (gwd[t+1,:,:] - gwd[t,:,:])*sy*area#here no need to convert yearly to monthly average
    #     for k in range(n_spatial_unit):
    #         s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])
    
    # s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)
    # gws_depletion_human = s_aggregated#monthly
    # gws_depletion_human = ma.masked_equal(gws_depletion_human, 1e20)

    # gws_depletion_human_yr = module.convert_month_to_year_sum(gws_depletion_human)
    
    
    
    # s_aggregated = np.full((n_spatial_unit, gwd1.shape[0]), 1e20)
    # for t in range(0,gwd1.shape[0]-1):
    #     temp =  (gwd1[t,:,:] - gwd1[t-1,:,:])*sy*area#here no need to convert yearly to monthly average
    #     for k in range(n_spatial_unit):
    #         s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])

    # s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)
    # gws_depletion_natural_yr = s_aggregated#yearly
    # gws_depletion_natural_yr = ma.masked_equal(gws_depletion_natural_yr, 1e20)
    
    # total_depletion = gws_depletion_human_yr - gws_depletion_natural_yr
    
    for t in range(ntime):
        temp =  (dem - gwd[t,:,:])*sy*area#here no need to convert yearly to monthly average
        for k in range(n_spatial_unit):
            s_aggregated[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])
    
    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)  
    
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'groundwater_storage_human_' + Input.name_timeperiod +'_'+ Input.name_scale, s_aggregated[:-1], ID, times, 'groundwater storage', 'month', 'm3') 
    
    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='human')  # Plot some data on the axes.
    #ax.set_xlabel('yr')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World groundwater storage "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated