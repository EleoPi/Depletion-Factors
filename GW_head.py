# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:44:22 2021

@author: easpi
"""

#import pcraster
import os
from pcraster import readmap, pcr2numpy
from netCDF4 import Dataset
import numpy as np
#import gdal

#function
def new_netcdf(filename, stressor,spatial_unit, latitude,longitude, time, stressor_name, unit_time,unit_stressor):
    """
    Parameters
    ----------
    filename : TYPE string
        DESCRIPTION.  name of the output file, the full location needed
    stressor: TYPE array
        DESCRIPTION. array (spatial unit id number, time in number)
        containing the value of the stressor  in each spatial unit
    spatial_unit : TYPE array
        DESCRIPTION. array (lat, lon) containing the spatial unit id numbers
    stressor_string : TYPE string to define thenameof variable into netcdf
        DESCRIPTION.
    unit_time : TYPE string
        DESCRIPTION. define theunit of the time variable. e.g. month
    unit_stressor : TYPE string 
        DESCRIPTION. to define the unit of the stressor variable. eg. ET
    latitude : TYPE array 0D
        DESCRIPTION: latitude variable values

    Returns
    -------
    None. Netcdf file is written based on the inputs. and the dataset is 
    returned to main program.

    """
    dataset_out = Dataset(filename +'.nc','w',format = 'NETCDF4')

    dataset_out.createDimension("time", stressor.shape[1])
    dataset_out.createDimension("spatial_unit_id",stressor.shape[0])
    dataset_out.createDimension("lat", spatial_unit.shape[0])
    dataset_out.createDimension("lon", spatial_unit.shape[1])
    

    time_var = dataset_out.createVariable("time","f4",("time",))
    time_var.units = unit_time
    lat_var = dataset_out.createVariable("lat","f4",("lat",))
    lon_var = dataset_out.createVariable("lon","f4",("lon",))
    spatial_unit_id_var = dataset_out.createVariable("spatial_unit_id","f4",("spatial_unit_id",))
    spatial_unit_map_var = dataset_out.createVariable("spatial_unit_map","f4",("lat","lon"))
    stressor_aggregated_timeserie_var = dataset_out.createVariable(stressor_name,"f4",("spatial_unit_id","time"))
    stressor_aggregated_timeserie_var.units = unit_stressor

    #fill NETCDF with results
    time_var[:] = time[:]
    lat_var[:] = latitude[:]
    lon_var[:] = longitude[:]
    spatial_unit_map_var[:] = spatial_unit[:]
    spatial_unit_id_var[:] = np.array(range(stressor.shape[0]))
    stressor_aggregated_timeserie_var[:] = stressor[:]

    dataset_out.sync()#write into the  saved file
    print(dataset_out)
    dataset_out.close()
    return "netcdf created, check folder"



#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')

#open DEM model with pcraster library and conversion to array.
dem_map = readmap('data/DEM_05min.map')
dem =pcr2numpy(dem_map,1e20)
dem.shape

#fn = 'data/new_DEM_05min.tif'
#dem_temp = gdal.Open(fn)
#print(dem_temp)
#dem=dem_temp.GetRasterBand(1)
#print(dem)

#open basin delineation: the spatial resolution can be changed here.
spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
spatial_unit =pcr2numpy(spatial_unit_map,1e20)
spatial_unit.shape

#import GW table depth dataset and extract variables groundwater depth
fn ='data/groundwaterDepthLayer1_monthEnd_1960to2010.nc'
dataset = Dataset(fn)
print(dataset)
stressor=dataset.variables["groundwater_depth_for_layer_1"]
#the groundwater depth for layer 1 array is too big to be loaded in 1 
#variable. I had to slice the years.


#calculation of groundwater head using groundwaterdepth and dem


ntime, nlat, nlon = stressor.shape
n_spatial_unit = np.max(spatial_unit)#number of spatial units
#initialization
n = 10 #number of catchments to make a test
stressor_aggregated_timeserie = np.zeros((n, ntime),dtype = 'float16')
#test for 2 basins, use n_spatial_unit for full picture

#loop to fill the aggregated stressor matrix. 
#for full dataset, use ntime in for t in range (ntime)
#this loop is designed to calculate the average groundwater head over the 
#spatial units for each timestep. The result is a time serie of stressor vals.

for t in range(ntime):#put ntime for full range
    stressor_time = stressor[t,:]#selectmap for time index = t
    for k in range(n):#putn_spatial_unit for full range
        dem_space = dem.copy()[spatial_unit == k] # select spatial unit k
        stressor_time_space = stressor_time.copy()[spatial_unit == k]
        #select stressor subset for spatial unit= k
        stressor_temp = dem_space - stressor_time_space
        #calculate something: groundwater head = DEM - Groundwater depth
        stressor_aggregated_timeserie[k,t] = np.average(stressor_temp)
        #sperform zonal aggregation

stressor_aggregated_timeserie.shape
print(stressor_aggregated_timeserie)
#verify that the zeros resulting from the iteration of initial condition 
#np.zeros() are gone whhen running the full loop.
#all gridcells have to be filled with data.
# for n=10 time 15h36 to 15h45 more or less 1 minute for each catchment
#total catchment = approx. 23000 !


#create NCDF with the aggregatedstressor and spatial unit information
time1 = dataset.variables["time"]#all timesteps
lat1 = dataset.variables["lat"]
lon1 = dataset.variables["lon"]

new_netcdf('output/test_groundwater_head_%n_catchments_1950_to_2010', stressor_aggregated_timeserie, spatial_unit, lat1, lon1, time1, "groundwater_head", "month", "m")

# stressor_dataset_out = Dataset('output/groundwater_head_aggregated_2catchment_monthly_1960_test1.nc','w',format = 'NETCDF4')
# #create dimensions and variables

# stressor_dataset_out.createDimension("time", stressor_aggregated_timeserie.shape[1])
# stressor_dataset_out.createDimension("lat", spatial_unit.shape[0])
# stressor_dataset_out.createDimension("lon", spatial_unit.shape[1])
# stressor_dataset_out.createDimension("spatial_unit_id",stressor_aggregated_timeserie.shape[0])

# time_var = stressor_dataset_out.createVariable("time","f4",("time",))
# #number of timesteps for which the agregated stressor was calculated
# lat_var = stressor_dataset_out.createVariable("lat","f4",("lat",))
# lon_var = stressor_dataset_out.createVariable("lon","f4",("lon",))
# spatial_unit_map_var = stressor_dataset_out.createVariable("spatial_unit_map","f4",("lat","lon"))
# spatial_unit_id_var = stressor_dataset_out.createVariable("spatial_unit_id","f4")
# #number of spatial units over which the stressor was aggrgated
# stressor_aggregated_timeserie_var = stressor_dataset_out.createVariable("groundwater_head_aggregated_monthly","f4",("spatial_unit_id","time"))
# stressor_aggregated_timeserie_var.units = "m"

# #fill NETCDF with results
# time_var[:] = dataset.variables["time"][0:stressor_aggregated_timeserie.shape[1]]
# lat_var[:] = dataset.variables["lat"][:]
# lon_var[:] = dataset.variables["lon"][:]
# spatial_unit_map_var[:] = spatial_unit
# spatial_unit_id_var[:] = np.array(range(stressor_aggregated_timeserie.shape[0]))
# stressor_aggregated_timeserie_var[:] = stressor_aggregated_timeserie #ok

# stressor_dataset_out.sync()#write into the  saved file 
# print(stressor_dataset_out)
# stressor_dataset_out.close()
