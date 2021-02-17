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
import time
#from numpy import nditer, ma
#import numpy.ma




#functions-----------------------------------------------------




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
    

    time_var = dataset_out.createVariable("time","f4",("time",), zlib = True)
    time_var.units = unit_time
    lat_var = dataset_out.createVariable("lat","f4",("lat",), zlib = True)
    lon_var = dataset_out.createVariable("lon","f4",("lon",), zlib = True)
    spatial_unit_id_var = dataset_out.createVariable("spatial_unit_id","f4",("spatial_unit_id",), zlib = True)
    spatial_unit_map_var = dataset_out.createVariable("spatial_unit_map","f4",("lat","lon"), zlib = True)
    stressor_aggregated_timeserie_var = dataset_out.createVariable(stressor_name,"f4",("spatial_unit_id","time"), zlib = True)
    stressor_aggregated_timeserie_var.units = unit_stressor

    #fill NETCDF with results
    time_var[:] = time[:]
    lat_var[:] = latitude[:]
    lon_var[:] = longitude[:]
    spatial_unit_map_var[:] = spatial_unit[:]
    spatial_unit_id_var[:] = spatial_unit_id[:]
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


# extract coordinate indexes for all catchments 
def index_filter(spatial_unit):
    """
    Parameters
    ----------
    spatial_unit : TYPE array (lat,lon)
        DESCRIPTION. array containing the ID of the catchments starting with ID = 0

    Returns
    -------
    index_filter : TYPE list of lists
        DESCRIPTION. the list contains for each catchmetn ID, the list of latitude index in position 0 and the list of longitude indexes for position 1

    """
    index_filter = []
    nmax = np.max(spatial_unit) + 1
    for n in range(nmax):
        a = np.where(spatial_unit == n)
        index_filter.append(a)
    
    return index_filter




#inputs(factorize across scripts in the future)---------------------




#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')

#open DEM model with pcraster library and conversion to array.
dem_map = readmap('data/DEM_05min.map')
dem =pcr2numpy(dem_map,1e20)
dem.shape

#import GW table depth dataset and extract variables groundwater depth
fn ='data/groundwaterDepthLayer1_monthEnd_1960to2010.nc'
dataset = Dataset(fn)
print(dataset)
stressor = dataset.variables["groundwater_depth_for_layer_1"]
tim = dataset.variables["time"][:]
lat = dataset.variables["lat"][:]
lon = dataset.variables["lon"][:]
#mask = ma.getmask(stressor[0,:,:])
#the groundwater depth for layer 1 array is too big to be loaded in 1 
#variable. I had to slice the years.

#open basin delineation: the spatial resolution can be changed here.
spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
spatial_unit =pcr2numpy(spatial_unit_map,1e20)
spatial_unit.shape
#spatial_unit_masked = ma.masked_outside(spatial_unit, 0, np.max(spatial_unit))
#spatial_unit_masked_1 = ma.masked_where(mask == True, spatial_unit, copy = True)




#aggregated stressor timseries calculations----------------------




#the pointer is a list of the coordinates indexes for each catchment 
start = time.time()
pointer = index_filter(spatial_unit) 
elapsed_time = (time.time() - start)#result = 663 s

pointer_array = np.array(pointer, dtype = list)# conversion to array
pointer_array.shape
np.save('output/full_catchment_filter_array', pointer_array)
# to open this filter later, use command np.load
#pointer_array = np.load('output/full_catchment_filter_array.npy', allow_pickle = True)
#index list ([array], [array]) of cells in catchment 2

#initatlization

ntime, nlat, nlon = stressor.shape

n_spatial_unit = np.max(spatial_unit) + 1 #number of spatial units from0 to 23117. total 23118

stressor_aggregated_timeserie = np.zeros((n_spatial_unit, ntime))

#calculation of aggregated stressor timeseries

start1 = time.time()
for t in range(ntime):
    s =  calculate_groundwater_head(stressor, dem, t)
    for k in range(n_spatial_unit):
        stressor_aggregated_timeserie[k,t] = np.mean(s[pointer[k][0],pointer[k][1]])
elapsed_time1 = (time.time() - start) 
#result = 1628s for all catchments and all time steps!

stressor_aggregated_timeserie

#save results to netcdf

spatial_unit_ID = np.array(range(n_spatial_unit))
np.max(spatial_unit_ID)#correct number of catchments
new_netcdf('output/test_groundwater_head_all_catchments_1950_to_2010', stressor_aggregated_timeserie, spatial_unit, spatial_unit_ID, lat, lon, tim, "groundwater_head", "month", "m")



#nan problem analysis-------------------------------------------------------
#extract nan locations
nan_problem = np.isnan(stressor_aggregated_timeserie)
nan_problem_ID = np.where(nan_problem == True)
stressor_aggregated_timeserie_noNAN = stressor_aggregated_timeserie[nan_problem == False]

nan_ID = np.unique(nan_problem_ID[0])#remove duplicates ID
len(nan_ID) #3143 catchments have nan problems

stressor_aggregated_timeserie_NAN_catchments = stressor_aggregated_timeserie[nan_ID, :]
np.where(np.isnan(stressor_aggregated_timeserie_NAN_catchments) == False)
#when 1 catchment is NAN, the whole timeserie is nan

pointer_nan = pointer_array[nan_ID,:]
pointer_nan #cell coordinates where we have nan problem potentially
pointer_nan.shape
np.save('output/test_groundwater_head_all_catchments_1950_to_2010_NANfilter',pointer_nan)

stressor[0,pointer_nan[2,0],pointer_nan[2,1]]
#there is no values in this catchment. 
#the problem is that the array is masked in this catchment...
#problem with catchment delineation?


#-------------------------------------------------------------------
#extraction of aggregated GWH for sample catchments





# #initialization

# ntime, nlat, nlon = stressor.shape

# n_spatial_unit = np.max(spatial_unit)#number of spatial units

# spatial_unit_sample = np.array([21447,21229, 21588, 21723, 19373, 8120, 9132, 8156, 8275, 7383])

# stressor_sample_aggregated_timeserie1 = np.zeros((len(spatial_unit_sample), ntime))
# stressor_sample_aggregated_timeserie1.shape

# for t in range(ntime):
#     stressor_t = calculate_groundwater_head(stressor, dem, t)
#     for k in range(len(spatial_unit_sample)):
#           a = np.extract(spatial_unit == spatial_unit_sample[k], stressor_t)
#           stressor_sample_aggregated_timeserie1[k,t] = np.mean(a)

# stressor_sample_aggregated_timeserie1
# #95s for 10 catchments and 612 timesteps. total estiamted for 23117 catchments : 62h



# #create NCDF with the aggregatedstressor and spatial unit information


# new_netcdf('output/test_groundwater_head_10sample_catchments_1950_to_2010_rev', stressor_sample_aggregated_timeserie1, spatial_unit, spatial_unit_sample, lat, lon, tim, "groundwater_head", "month", "m")

# # s0 =stressor[0,:,:]
# # s0.shape
# #spatial_unit_ID = np.array(range(n_spatial_unit))
# # time2 = time[0]
# # new_netcdf_simple('output/test_groundwater_head_all_catchments_1_timestep_5', s0, lat1, lon1, "groundwater_head", "m")




# # calculation of full dataset




# #initfull loop

# ntime, nlat, nlon = stressor.shape

# n_spatial_unit = np.max(spatial_unit) + 1
# # spatial_unit_ID = np.array(range(n_spatial_unit))
# # spatial_unit_ID# list of spatial unit IDs

# stressor_aggregated_timeserie = np.zeros((n_spatial_unit, ntime))
# stressor_aggregated_timeserie.shape

# for t in range(ntime):
#     s = calculate_groundwater_head(stressor, dem, t)
#     for k in range(n_spatial_unit):#here run the loop for samples of catchments to // calculation
#         a =  np.extract(spatial_unit == k, s)
#         stressor_aggregated_timeserie[k,t] = np.mean(a)
#         #perform zonal aggregation ignoring NAN
# #too long to run
# stressor_aggregated_timeserie.shape
# print(stressor_aggregated_timeserie)





#new_netcdf('output/test_groundwater_head_catchments_1950_to_2010', stressor_aggregated_timeserie, spatial_unit, lat1, lon1, time1, "groundwater_head", "month", "m")
#save when calulcation succeed.


