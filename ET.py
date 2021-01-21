# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:44:22 2021

@author: easpi
"""

#import dataset
import netCDF4 as nc
import pcraster as pcr
import os

#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations')


#data upload
fn ='data/totalEvaporation_annuaTot_output_1960to2010.nc'
ET = nc.Dataset(fn)
print(ET)

basins='data/basins_valerio_5min.tif'


pcr.setclone('data/clone_global_05min.map')
