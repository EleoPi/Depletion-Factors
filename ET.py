#import pcraster
import os
from pcraster import readmap, pcr2numpy
from netCDF4 import Dataset
import numpy as np

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

#open basin delineation: the spatial resolution can be changed here.
spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
spatial_unit =pcr2numpy(spatial_unit_map,1e20)
spatial_unit.shape

#Et data upload and array extraction
fn ='data/totalEvaporation_annuaTot_output_1960to2010.nc'
ET_data = Dataset(fn)
print(ET_data)
ET = ET_data.variables['total_evaporation'][:]


#surface of gridcell
area_map = readmap('data/Global_CellArea_m2_05min.map')
area =pcr2numpy(area_map,1e20)
area.shape
print(area)


#calculation of stressor over spatial unit

#initialization
ntime, nlat, nlon = ET.shape
n_spatial_unit = np.max(spatial_unit)#number of spatial units

stressor = ET.copy()
#n = 10 #number of catchments to make a test
stressor_aggregated_timeserie = np.zeros((n_spatial_unit, ntime))
#test for n basins, use n_spatial_unit for full picture

for t in range(ntime):#put ntime for full range
    stressor[t,:,:] = np.multiply(ET[t,:,:], area)
    s = stressor[t,:,:]
    #selectmap for time index = t
    for k in range(n_spatial_unit):#putn_spatial_unit - 1 for full range, the count starts in 1 if the basins id starts in 1
        stressor_aggregated_timeserie[k,t] = np.sum(s[spatial_unit == k], dtype = 'float32')
        #perform zonal aggregation
        #for ET the aggregation procedure is sum

stressor_aggregated_timeserie.shape

print(stressor_aggregated_timeserie)
