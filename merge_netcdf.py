# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:54:20 2021

@author: easpi
"""

import os
from pcraster import readmap, pcr2numpy, setclone
from netCDF4 import Dataset
import netCDF4 as nc
import numpy as np

#create dimensions and variables

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


# #test function that creates new netcdf with 3 dimensions based on a filename

# def createNetCDF(self, ncFileName, varName, varUnits, longName = None, standardName= None):
#     rootgrp = nc.Dataset(ncFileName,'w',format= self.format)
# #-create dimensions - time is unlimited, others are fixed
#     rootgrp.createDimension('time',None)
#     rootgrp.createDimension('lat',len(self.latitudes))
#     rootgrp.createDimension('lon',len(self.longitudes))

#     date_time = rootgrp.createVariable('time','f4',('time',))
#     date_time.standard_name = 'time'
#     date_time.long_name = 'Days since 1901-01-01'

#     date_time.units = 'Days since 1901-01-01' 
#     date_time.calendar = 'standard'

#     lat= rootgrp.createVariable('lat','f4',('lat',))
#     lat.long_name = 'latitude'
#     lat.units = 'degrees_north'
#     lat.standard_name = 'latitude'

#     lon= rootgrp.createVariable('lon','f4',('lon',))
#     lon.standard_name = 'longitude'
#     lon.long_name = 'longitude'
#     lon.units = 'degrees_east'

#     lat[:]= self.latitudes
#     lon[:]= self.longitudes

#     shortVarName = varName
#     longVarName  = varName
#     standardVarName = varName
#     if longName != None: longVarName = longName
#     if standardName != None: standardVarName = standardName()
    
#     var = rootgrp.createVariable(shortVarName,'f4',('time','lat','lon',) ,fill_value=1e20,zlib=self.zlib)
#     var.standard_name = standardVarName
#     var.long_name = longVarName
#     var.units = varUnits

#     attributeDictionary = self.attributeDictionary
#     for k, v in list(attributeDictionary.items()): setattr(rootgrp,k,v)

#     rootgrp.sync()
#     rootgrp.close()