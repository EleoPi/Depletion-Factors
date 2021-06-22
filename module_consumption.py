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

def aggregate_return_flow_irrigation(pointer_array):
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit

   Returns timeseries of the return flows from irrigation, summed over the gridcells of each spatial  units. negative values are filtered
      -------
    None.

    '''

#irrigation retunr flows calculation-----------------------------------------------------------
#we have to do it manually because the gwd file is to big to load in the memory.


    d1 = Dataset('D:/Fate/data/gwRecharge_monthTot_output_1960to2010_human.nc')
    
    recharge_human = d1.variables['groundwater_recharge']#monthly
    time_human = ma.getdata(d1.variables['time'][:])
    
    
    d2 = Dataset('D:/Fate/data/gwRecharge_monthTot_output_1960to2010_natural.nc')
    
    recharge_natural = d2.variables['groundwater_recharge']#monthly. reprated times
    time_natural = ma.getdata(d2.variables['time'][:])

    
    time_unique = ma.getdata(np.unique(time_natural, return_index = 1))

    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)

    #aggrgeate 
    ntime = time_human.shape[0]
    n_spatial_unit = pointer_array.shape[0]
    return_flow_irrigation = np.full((n_spatial_unit, ntime), 1e20)
    
    for t in range(len(time_unique[1])):
        temp =  (recharge_human[t, :, :] - recharge_natural[time_unique[1][t], :, :] )*area
        for k in range(n_spatial_unit):
            return_flow_irrigation[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])#should i exclude negatives values here?
    
    return_flow_irrigation = ma.masked_where(isnan(return_flow_irrigation), return_flow_irrigation)
    
    return_flow_irrigation_yr = module.convert_month_to_year_avg(return_flow_irrigation)

    return_flow_irrigation_yr_filtered = np.where(return_flow_irrigation_yr >= 0, return_flow_irrigation_yr, 0)

#    module.new_stressor_out_netcdf(Input.outputDir + '/'+'test_net_return_flow_irrigation_'+ Input.name_timeperiod + '_' + Input.name_scale, return_flow_irrigation_yr_filtered[:-1,:] , ID, time3, 'total return flows irrigation  and non irrigation', 'year', 'm3') 
    
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(np.sum(return_flow_irrigation_yr_filtered/1e9, axis = 0), label='filter>0')  # Plot some data on the axes.
    ax.plot(np.sum(return_flow_irrigation_yr/1e9, axis = 0), label='all values')
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World return flows from irrigation "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
    
    return return_flow_irrigation, return_flow_irrigation_yr_filtered


def aggregate_return_flow_non_irrigation(pointer_array):
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit

    Returns timeseries of the return flows from industry and domestic uses, summed over the gridcells of each spatial  units. negative values are filtered
    
    None.

    '''
    d3 = Dataset('D:/Fate/data/nonIrrWaterConsumption_annualTot_1960to2010b.nc')
    
    consumption_non_irrigation = d3.variables['consumptive_water_use_for_non_irrigation_demand']
    time3 = d3.variables['time'][:]
    
    d4 = Dataset('D:/Fate/data/industry_Withdrawal_annualTot_out_1960to2010_b.nc')
    
    withdrawal_industry = d4.variables['industry_water_withdrawal'][:]
    #time4 = d4.variables['time'][:]
    
    d5 = Dataset('D:/Fate/data/domesticWithdrawal_annualTot_out_1960to2010_b.nc')
    
    withdrawal_domestic = d5.variables['domestic_water_withdrawal'][:]
    #time5 = d5.variables['time'][:]
    
    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)
    
    #time3 == time 4 and time4 == time 5 ok
    n_spatial_unit = pointer_array.shape[0]
    
    return_flow_non_irrigation = np.full((n_spatial_unit, time3.shape[0]), 1e20)
    
    for t in range(len(time3)):
        temp = ( - consumption_non_irrigation[t,:,:] + withdrawal_industry[t,:,:] + withdrawal_domestic[t,:,:] )*area
        for k in range(n_spatial_unit):
            return_flow_non_irrigation[k,t] = np.sum(temp[pointer_array[k][0],pointer_array[k][1]])
    
    return_flow_non_irrigation = ma.masked_where(isnan(return_flow_non_irrigation), return_flow_non_irrigation)
    
    return_flow_non_irrigation_filtered = np.where(return_flow_non_irrigation >= 0, return_flow_non_irrigation, 0)
    
    return_flow_non_irrigation_filtered  = ma.masked_where(isnan(return_flow_non_irrigation_filtered), return_flow_non_irrigation_filtered)
    
    world = np.sum(return_flow_non_irrigation_filtered, axis = 0)
    
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='return flow>0')  # Plot some data on the axes.
    ax.plot(np.sum(return_flow_non_irrigation, axis = 0)/1e9, label='all values')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World return flows from households and industry "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.

    
    return return_flow_non_irrigation, return_flow_non_irrigation_filtered

def aggregate_abstraction(pointer_array):
    '''
    

    Parameters
    ----------
    basin : TYPE scalar array
        DESCRIPTION. spatial unit IDs
    pointer_array : TYPE array (spatialunit,indexes)
        DESCRIPTION. for each spatial unit (rows) there is an index filter pointing at the coordinates of the spatial unit

    Returns timeseries of the world water abstraction summed over the gridcells of each spatial  units 
    -------
    None.

    '''
    
    
    d = Dataset('D:/Fate/data/totalAbstractions_annuaTot_1960to2010.nc')

    abstraction = d.variables['total_abstraction'][:]# m/yr
 
    times = d.variables['time'][:]
    
    area = pcr2numpy(readmap(Input.inputDir + '/' + Input.fn_area_map), mv = 1e20)
    
    n_spatial_unit = pointer_array.shape[0]
    
    abstraction_aggregated = np.full((n_spatial_unit, times.shape[0]), 1e20)
    
    for t in range(times.shape[0]):
        temp =  abstraction[t,:,:]*area #conversion to m3
        for k in range(n_spatial_unit):
            abstraction_aggregated[k,t] = np.sum(temp[pointer_array[k][0], pointer_array[k][1]])
    
    abstraction_aggregated_filtered = ma.masked_where(isnan(abstraction_aggregated), abstraction_aggregated)
     
    world = np.sum(abstraction_aggregated_filtered, axis = 0)
    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='world abstractions')  # Plot some data on the axes.
    ax.set_xlabel('year')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World abstractions "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.
   
    return abstraction_aggregated_filtered

def calculate_net_consumption(abstraction, return_flow_irrigation, return_flow_non_irrigation, idlist):
 
    d = Dataset('D:/Fate/data/totalAbstractions_annuaTot_1960to2010.nc')

    times = d.variables['time'][:]
     
    net_consumption = abstraction - return_flow_irrigation - return_flow_non_irrigation
    
    net_consumption = ma.masked_where(isnan(net_consumption), net_consumption)
    

    
    
    total_return_flows = return_flow_irrigation + return_flow_non_irrigation

    total_return_flows = ma.masked_where(isnan(total_return_flows), total_return_flows)

    world = np.sum(net_consumption, axis = 0)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(world/1e9, label='consumption')  # Plot some data on the axes.
    ax.plot(np.sum(total_return_flows, axis =0)/1e9, label='return flows')  # Plot some data on the axes.
    ax.set_xlabel('month')  # Add an x-label to the axes.
    ax.set_ylabel('km3')  # Add a y-label to the axes.
    ax.set_title("World consumption and retunr flows "+Input.name_timeperiod)  # Add a title to the axes.
    ax.legend()  # Add a legend.


    ID = ma.getdata(idlist)[:-1]
    #ID = ma.getdata(np.unique(spatial_unit)[3:-1])#for valerios catchment
    
    
    module.new_stressor_out_netcdf(Input.outputDir + '/'+'consumption_' + Input.name_timeperiod + '_' + Input.name_scale, net_consumption[:-1,:], ID, times, 'net water consumption', 'year', 'm3') 

    module.new_stressor_out_netcdf(Input.outputDir + '/'+'return_flows_'+ Input.name_timeperiod + '_' + Input.name_scale, total_return_flows[:-1,:] , ID, times, 'total return flows irrigation  and non irrigation', 'year', 'm3') 

    return net_consumption


# def calculate_variation(aggregated, spatial_unit, pointer_array):
#     '''
    

#     Parameters
#     ----------
#      aggregated_mean : TYPE groundwater aggregated array (spatial_units, time)
#         DESCRIPTION.represents the mean GWH for each catchments defined in spatial_unit
#     basin : TYPE array
#         DESCRIPTION. spatial_unit delineation and clonemap
#     pointer_array : TYPE array
#         DESCRIPTION.filter array that identified the coordinates of each spatial units

#     Returns map of the sum of differences between timesteps over the period correpsonding to spatial units, and values for each catchemnts
#     -------
#     None.

#     '''

#     d = Dataset('D:/Fate/data/totalAbstractions_annuaTot_1960to2010.nc')

#    # time = d.variables["time"][:]
#     lat = d.variables["latitude"][:]
#     lon = d.variables["longitude"][:]
    
# #average FF
#     var = np.sum(np.diff(aggregated, axis = 1), axis = 1)
#     ref = np.mean(aggregated, axis = 1)
    
#     if ref !=0:
#         change = var/ref
#     else: change = var
    
#     map_var = module.make_map(change, pointer_array, spatial_unit)
    
#     plt.matshow(map_var, vmin = stats.scoreatpercentile(var,5), vmax = stats.scoreatpercentile(var,95))#ok
    
#     module.new_map_netcdf(Input.outputDir +'/'+"test_net_consumption_change_normalized_" + Input.name_timeperiod + '_' + Input.name_scale, map_var, "net consumption variation", "m3", lat, lon)

#     return map_var, var


    


# def calculate_grad_avg(aggregated, spatial_unit, pointer_array):
#     """
    

#     Parameters
#     ----------
#     aggregated_mean : TYPE groundwater aggregated array (spatial_units, time)
#         DESCRIPTION.represents the mean GWH for each catchments defined in spatial_unit
#     basin : TYPE array
#         DESCRIPTION. spatial_unit delineation and clonemap
#     pointer_array : TYPE array
#         DESCRIPTION.filter array that identified the coordinates of each spatial units

#     Returns map of the annual gradient average over the last 10 years in each spatial unit,  gradient values for each spatial unit
#     -------
#     None.

#     """
#     d = Dataset('D:/Fate/data/totalAbstractions_annuaTot_1960to2010.nc')

#    # time = d.variables["time"][:]
#     lat = d.variables["latitude"][:]
#     lon = d.variables["longitude"][:]
    
      
#     grad = np.mean(np.gradient(aggregated, axis = 1)[:, -10:], axis = 1)
    
#     ref = np.mean(np.gradient(aggregated, axis = 1), axis = 1)
    
#     change = grad/ref
    
#     stats.scoreatpercentile(grad,5)
#     stats.scoreatpercentile(grad,50)
#     stats.scoreatpercentile(grad,95)
    
#     map_grad = module.make_map(change, pointer_array, spatial_unit)
    
#     plt.imshow(map_grad)
#     #plt.imshow(s_map_grad/1e6, vmin = stats.scoreatpercentile(s_grad/1e6,5), vmax = stats.scoreatpercentile(s_grad/1e6,95))
#     #yellow is max, blue is min
    
#     module.new_map_netcdf(Input.outputDir +'/'+ "test_net consumption_gradient_1950_to_2010"+ Input.name_scale,  map_grad, "net consumption gradient", "m3/yr", lat, lon)

#     return map_grad, grad