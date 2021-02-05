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
#import numpy.ma




#functions




def new_netcdf(filename, stressor,spatial_unit, spatial_unit_id, latitude,longitude, time, stressor_name, unit_time,unit_stressor):
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
    spatial_unit id : TYPE array 1D
        DESCRIPTION. array containing the spatial unit id values where the stressor was aggregated
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
    

    time_var = dataset_out.createVariable("time","f4",("time",), zlib=True)
    time_var.units = unit_time
    lat_var = dataset_out.createVariable("lat","f4",("lat",), zlib=True)
    lon_var = dataset_out.createVariable("lon","f4",("lon",), zlib=True)
    spatial_unit_id_var = dataset_out.createVariable("spatial_unit_id","f4",("spatial_unit_id",), zlib=True)
    spatial_unit_map_var = dataset_out.createVariable("spatial_unit_map","f4",("lat","lon"), zlib=True)
    stressor_aggregated_timeserie_var = dataset_out.createVariable(stressor_name,"f4",("spatial_unit_id","time"), zlib=True)
    stressor_aggregated_timeserie_var.units = unit_stressor

    #fill NETCDF with results
    time_var[:] = time[:]
    lat_var[:] = latitude[:]
    lon_var[:] = longitude[:]
    spatial_unit_map_var[:] = spatial_unit[:]
    spatial_unit_id_var[:] = spatial_unit_id
    #spatial_unit_id_var[:] = np.array(range(stressor.shape[0]))
    stressor_aggregated_timeserie_var[:] = stressor[:]

    dataset_out.sync()#write into the  saved file
    print(dataset_out)
    dataset_out.close()
    return "netcdf created, check folder"


def new_netcdf_simple(filename, stressor, latitude,  longitude, stressor_name, unit_stressor):
    """
    Parameters
    ----------
    filename : TYPE string
        DESCRIPTION.  name of the output file, the full location needed
    stressor: TYPE array
        DESCRIPTION. array (spatial unit id number, time in number)
        containing the value of the stressor  in each spatial unit
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

    dataset_out.createDimension("lat", stressor.shape[0])
    dataset_out.createDimension("lon", spatial_unit.shape[1])
    
    lat_var = dataset_out.createVariable("lat","f4",("lat",), zlib=True)
    lon_var = dataset_out.createVariable("lon","f4",("lon",), zlib=True)
    out_var = dataset_out.createVariable(stressor_name,"f4",("lat","lon"), zlib=True)
    out_var.units = unit_stressor

    #fill NETCDF with results
    lat_var[:] = latitude[:]
    lon_var[:] = longitude[:]
    #spatial_unit_id_var[:] = np.array(range(stressor.shape[0]))
    out_var[:] = stressor[:]

    dataset_out.sync()#write into the  saved file
    print(dataset_out)
    dataset_out.close()
    return "netcdf created, check folder"


#calculation of groundwater head using groundwaterdepth and dem

def calculate_groundwater_head(s, dem, t):
    '''
    
    Parameters
    ----------
    s : TYPE netcdf variable  or array (time, lat, lon)
        DESCRIPTION. groundwater depth 
    dem : TYPE array (lat, lon)
        DESCRIPTION. digital elevation model
    t : TYPE integer
        DESCRIPTION. timestep index in s

    Returns
    -------
    out : TYPE array(lat,lon)
        DESCRIPTION. groundwater head at the given timestep t

    '''
    s_t = s[t,:,:]
    out = dem - s_t
    return out




#inputs(factorize across scripts in the future)





#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')

#open DEM model with pcraster library and conversion to array.
dem_map = readmap('data/DEM_05min.map')
dem =pcr2numpy(dem_map,1e20)
dem.shape

#open basin delineation: the spatial resolution can be changed here.
spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
spatial_unit =pcr2numpy(spatial_unit_map,1e20)
spatial_unit.shape


#import GW table depth dataset and extract variables groundwater depth
fn ='data/groundwaterDepthLayer1_monthEnd_1960to2010.nc'
dataset = Dataset(fn)
print(dataset)
stressor = dataset.variables["groundwater_depth_for_layer_1"]
time = dataset.variables["time"][:]
#the groundwater depth for layer 1 array is too big to be loaded in 1 
#variable. I had to slice the years.







#extraction of aggregated GWH for sample catchments





#initialization

ntime, nlat, nlon = stressor.shape

n_spatial_unit = np.max(spatial_unit)#number of spatial units

spatial_unit_sample = np.array([21447,21229, 21588, 21723, 19373, 8120, 9132, 8156, 8275, 7383])#in Brazil and in France,wth different sizes

stressor_sample_aggregated_timeserie = np.zeros((len(spatial_unit_sample), ntime))
stressor_sample_aggregated_timeserie.shape

for t in range(ntime):
    stressor_t = calculate_groundwater_head(stressor, dem, t)
    for k in range(len(spatial_unit_sample)):
          a = np.extract(spatial_unit == spatial_unit_sample[k], stressor_t)
          stressor_sample_aggregated_timeserie[k,t] = np.mean(a)

stressor_sample_aggregated_timeserie
#95s for 10 catchments and 612 timesteps. total estiamted for 23117 catchments : 62h



#create NCDF with the aggregatedstressor and spatial unit information
time1 = dataset.variables["time"]#all timesteps
lat1 = dataset.variables["lat"]
lon1 = dataset.variables["lon"]

new_netcdf('output/test_groundwater_head_10sample_catchments_1950_to_2010_rev', stressor_sample_aggregated_timeserie, spatial_unit, spatial_unit_sample, lat1, lon1, time1, "groundwater_head", "month", "m")






# calculation of full dataset






#initfull loop

ntime, nlat, nlon = stressor.shape

n_spatial_unit = np.max(spatial_unit)
spatial_unit_ID = np.array(range(n_spatial_unit))
spatial_unit_ID# list of spatial unit IDs

stressor_aggregated_timeserie = np.zeros((n_spatial_unit, ntime))
stressor_aggregated_timeserie.shape

for t in range(ntime):
    stressor_t = calculate_groundwater_head(stressor, dem, t)
    for k in range(n_spatial_unit):#here run the loop for samples of catchments to // calculation
        #a =  numpy.extract(spatial_unit == spatial_unit_ID[k], stressor_t)
        a = np.extract(spatial_unit == k, stressor_t)
        stressor_aggregated_timeserie[k,t] = np.mean(a)
        #perform zonal aggregation ignoring NAN

stressor_aggregated_timeserie.shape
print(stressor_aggregated_timeserie)
#code not running it is too long-. go paralel making group of catchments.
#define function to paralelize calculation per catchment samples
#spatial_unit_ID = np.array(range(n_spatial_unit))
# sample size = total catchment number /number threads
# sample vector = [[catchment ID 0 : 10][10 : 20] .... [23107 : 23117]]
# run the loop for all samples in //




#new_netcdf('output/test_groundwater_head_catchments_1950_to_2010', stressor_aggregated_timeserie, spatial_unit, lat1, lon1, time1, "groundwater_head", "month", "m")
#save when calulcation succeed.



