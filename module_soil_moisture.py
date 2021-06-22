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



from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy
import numpy as np
from numpy import isnan, ma
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

#-----------------------------------------------------------
    
    area = pcr2numpy(readmap(Input.inputDir +'/'+ Input.fn_area_map), mv = 1e20)
    
    d = Dataset(Input.inputDir +'/'+ Input.fn_soil_low)
    slow = d.variables["lower_soil_storage"][:]
    
    #lat = d.variables["latitude"][:]
    #lon = d.variables["longitude"][:]
    d.close()
    
    d = Dataset(Input.inputDir +'/'+ Input.fn_soil_upp)
    sup = d.variables["upper_soil_storage"][:]
    d.close()
    
    soil_moisture = slow + sup
    
    d1 = Dataset(Input.inputDir +'/'+ Input.fn_soil_low_nat)
    times = d1.variables["time"][:]
    slow1 = d1.variables["lower_soil_storage"][:]
    #lat = d.variables["latitude"][:]
    #lon = d.variables["longitude"][:]
    d1.close()
    
    d1 = Dataset(Input.inputDir +'/'+ Input.fn_soil_upp_nat)
    sup1 = d1.variables["upper_soil_storage"][:]
    d1.close()
    
    soil_moisture1 = slow1 + sup1
    
    s = soil_moisture[45,:,:] - soil_moisture1
    
    
    
#aggrgeate 
    ntime, nlat, nlon = s.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    
    for t in range(ntime):
        temp =  s[t,:,:]*area#conversion to m3
        for i in range(n_spatial_unit):#select catchment in the same order as the pointer array sot hat the idlist is still valid
            s_aggregated[i,t] = np.sum(temp[pointer_array[i][0], pointer_array[i][1]])

    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)


    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'soil_moisture_human-natural_' + Input.name_timeperiod + '_' + Input.name_scale, s_aggregated[:-1], ID, times, 'soil moisture', 'year', 'm3') 
    
    world_SM = np.sum(s_aggregated, axis = 0)
    
    
    # Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world_SM/1e9, label='soil moisture')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World soil moisture human - natural "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    np.sum(world_SM)/1e9
    
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

#-----------------------------------------------------------
    
    area = pcr2numpy(readmap(Input.inputDir +'/'+ Input.fn_area_map), mv = 1e20)
    
    d = Dataset(Input.inputDir +'/'+Input.fn_soil_low)
    slow = d.variables["lower_soil_storage"][:]
    times = d.variables["time"][:]
    #lat = d.variables["latitude"][:]
    #lon = d.variables["longitude"][:]
    d.close()
    
    d = Dataset(Input.inputDir +'/'+Input.fn_soil_upp)
    sup = d.variables["upper_soil_storage"][:]
    d.close()
    
    soil_moisture = slow + sup
    
    s = soil_moisture
    
    
    
#aggrgeate 
    ntime, nlat, nlon = s.shape
    n_spatial_unit = pointer_array.shape[0]
    s_aggregated = np.full((n_spatial_unit, ntime), 1e20)
    
    
    for t in range(ntime):
        temp =  s[t,:,:]*area#conversion to m3
        for i in range(n_spatial_unit):#select catchment in the same order as the pointer array sot hat the idlist is still valid
            s_aggregated[i,t] = np.sum(temp[pointer_array[i][0], pointer_array[i][1]])

    s_aggregated = ma.masked_where(isnan(s_aggregated), s_aggregated)


    ID = ma.getdata(idlist[:-1])

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'test_soil_moisture_human_' + Input.name_timeperiod + '_' + Input.name_scale, s_aggregated[:-1], ID, times, 'soil moisture', 'year', 'm3') 
    
    world_SM = np.sum(s_aggregated, axis = 0)
    
    
    # Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world_SM/1e9, label='soil moisture')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World soil moisture "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    np.sum(world_SM)/1e9
    
    return s_aggregated

