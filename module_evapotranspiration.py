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
 


def aggregate(idlist, pointer_array):
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

    d = Dataset(Input.inputDir +'/'+'totalEvaporation_annuaTot_output_1960to2010_human.nc')
    s = d.variables['total_evaporation'][:]
    #times = d.variables["time"][:]#year
    # lat = d.variables["latitude"][:]
    # lon = d.variables["longitude"][:]
    
    d1 = Dataset(Input.inputDir +'/'+'totalEvaporation_monthTot_output_1960to2004_natural.nc')#title is inaccurate, the timespan is identical to hu,man run 1960-2012 - 52 years of register
    s1 = d1.variables['total_evaporation']#month
    times1 = d1.variables['time'][:]

    
#aggrgeate 
    ntime, nlat, nlon = s1.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    
    for t in range(ntime):
        temp =  (s[t//12,:,:]/12 - s1[t,:,:])*area#conversion to m3, the annual human run ET us converted to monthly ET
        for i in range(n_spatial_unit):#select catchment in the same order as the pointer array sot hat the idlist is still valid
            s_aggregated[i,t] = np.sum(temp[pointer_array[i][0], pointer_array[i][1]])

    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)

    ID = ma.getdata(idlist[:-1])#exclude where the idlist does not correspondto a catchment
    
    module.new_stressor_out_netcdf(Input.outputDir + '/'+'evapotranspiration_human-natural_1960_2004_1'  + '_' + Input.name_scale, s_aggregated[:-1,:], ID, times1, 'evapotranspiration', 'yr', 'm3') 
    
    
    
    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='evapotranspiration')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World evapotranspiration human - natural 1960-2004"+Input.name_timeperiod)  # Add a title to the axes.
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
        correspondence between catchment ID and spatial_unit row is does using the idlist 

    Returns timeseries of the groundwater head average over each spatial unit for all timesteps 
    -------
    None.

    '''

#groundwater head-----------------------------------------------------------
#we have to do it manually because the gwd file is to big to load in the memory.
    
    area = pcr2numpy(readmap(Input.inputDir +'/'+ Input.fn_area_map), mv = 1e20)

    d = Dataset(Input.inputDir +'/'+ 'totalEvaporation_annuaTot_output_1960to2010_human.nc')
    s = d.variables['total_evaporation'][:]
    times = d.variables["time"][:]#year
    # lat = d.variables["latitude"][:]
    # lon = d.variables["longitude"][:]
    
    #d1 = Dataset('D:/fate/data/totalEvaporation_monthTot_output_1960to2004_natural.nc')#title is inaccurate, the timespan is identical to hu,man run 1960-2012 - 52 years of register
    #s1 = d1.variables['total_evaporation']#month
    #times1 = d1.variables['time'][:]

    
#aggrgeate 
    ntime, nlat, nlon = s.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    
    for t in range(ntime):
        temp =  (s[t,:,:])*area#conversion to m3, the annual human run ET us converted to monthly ET
        for i in range(n_spatial_unit):#select catchment in the same order as the pointer array sot hat the idlist is still valid
            s_aggregated[i,t] = np.sum(temp[pointer_array[i][0], pointer_array[i][1]])

    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)

    ID = ma.getdata(idlist[:-1])#exclude where the idlist does not correspondto a catchment
    
    module.new_stressor_out_netcdf(Input.outputDir + '/'+'evapotranspiration_human'  + '_' + Input.name_scale, s_aggregated[:-1,:], ID, times, 'evapotranspiration', 'yr', 'm3') 
    
    
    
    world = np.sum(s_aggregated, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='evapotranspiration')  # Plot some data on the axes.
    ax.set_xlabel('yr')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World evapotranspiration_human_1960-2004"+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return s_aggregated

