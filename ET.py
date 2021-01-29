# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:44:22 2021

@author: easpi
"""
#import dataset
import netCDF4 as nc
import numpy as np
import pcraster as pcr
from pcraster import setclone, readmap, pcr2numpy, collection
import os
import geopandas as gps
import rasterio as rio
import rioxarray as rxr
import earthpy as et
import earthpy.plot as ep
from osgeo import gdal

#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations')

pcr.setclone('data/clone_global_05min.map')

#Et data upload and array extraction
fn ='data/totalEvaporation_annuaTot_output_1960to2010.nc'
ET_aux = nc.Dataset(fn)
print(ET_aux)
ET=ET_aux.variables['total_evaporation'][:]
ET.shape

#load basins as arrays
basins_aux=readmap('data/catchments.map')
basins=pcr2numpy(basins_aux,1e20)
print(basins)
basins.shape
