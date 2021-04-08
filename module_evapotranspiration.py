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
 


def aggregate(spatial_unit, pointer_array):
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
    
    area = pcr2numpy(readmap(Input.inputDir +'/'+ Input.fn_area_map), mv = 1e20)

    d = Dataset('D:/fate/data/totalEvaporation_annuaTot_output_1960to2010.nc')
    s = d.variables['total_evaporation'][:]
    times = d.variables["time"][:]
    # lat = d.variables["latitude"][:]
    # lon = d.variables["longitude"][:]
    
    d1 = Dataset('D:/fate/data/totalEvaporation_monthTot_output_1960to2004_natural.nc')#title is inaccurate, the timespan is identical to hu,man run 1960-2012 - 52 years of register
    s1 = d.variables['total_evaporation'][:]

    
#aggrgeate 
    ntime, nlat, nlon = s.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    
    for t in range(ntime):
        temp =  (s[t,:,:] - s1[t,:,:])*area#conversion to m3
        for i in range(n_spatial_unit):#select catchment in the same order as the pointer array sot hat the idlist is still valid
            s_aggregated[i,t] = np.sum(temp[pointer_array[i][0], pointer_array[i][1]])

    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)

    idlist = np.unique(spatial_unit)
    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'evapotranspiration_human_natural_1960_2012'  + '_' + Input.name_scale, s_aggregated[:-1], ID, times, 'evapotranspiration', 'year', 'm3') 

    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='evapotranspiration')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World evapotranspiration "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated

def calculate_variation(aggregated, spatial_unit, pointer_array):
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
    var = np.sum(np.diff(aggregated, axis = 1), axis = 1)
    
    map_var = module.make_map(var, pointer_array, spatial_unit)
    
    plt.matshow(map_var, vmin = stats.scoreatpercentile(var,5), vmax = stats.scoreatpercentile(var,95))#ok
    
    module.new_map_netcdf(Input.outputDir +'/'+"test_evapotranspiration_variation_" + Input.name_timeperiod + '_' + Input.name_scale, map_var, "total evaporation", "m3", lat, lon)
      
    return map_var, var



def calculate_grad_avg(aggregated, spatial_unit, pointer_array):
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
     
    grad = np.mean(np.gradient(aggregated, axis = 1)[:, -10:], axis = 1)
    
    map_grad = module.make_map(grad, pointer_array, spatial_unit)
    
    map_grad_mask = ma.masked_where(isnan(map_grad), map_grad)
    
    plt.matshow(map_grad_mask)
    
    module.new_map_netcdf(Input.outputDir +'/'+ "test_evaporation_gradient_" + Input.name_timeperiod + '_' + Input.name_scale, map_grad, "total evaporation gradient", "m3/yr", lat, lon)

    return map_grad, grad