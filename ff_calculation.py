# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 13:38:28 2021

@author: easpi
"""

import os
#from pcraster import readmap, pcr2numpy
from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy, numpy2pcr, setclone
import numpy as np
from numpy import isnan, ma, trapz
from matplotlib import pyplot as plt
from scipy import stats
#import scipy # more intergation tools

#get hlep np.info(obj)

#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')

#import aggregated stressor timseries
fn ='output/test_groundwater_head_all_catchments_1950_2010_rev3.nc'#aggregated for catchment.map

dataset = Dataset(fn)

print(dataset)

stressor_tss = dataset.variables["groundwaterhead"][:]

time = dataset.variables["time"][:]

#we have missing values problem, lets put a mask on them
mask = np.isnan(stressor_tss)

stressor_tss_mask = ma.masked_where(mask, stressor_tss)

nonan_ID = np.unique(np.where(mask[:,0] == False))#catchment id with data
len(nonan_ID)
nan_ID = np.unique(np.where(mask[:,0] == True))#catchment id where no data
len(nan_ID)
#stressor_tss_nonan = ma.getdata(stressor_tss[nonan_ID,:])#extract the basins with data 

#some plots to visualize what is happening in several catchments
plt.plot(stressor_tss[21229,:])#nagtive values....
plt.plot(stressor_tss[21723,:])#basin of Paraná river, increase GWH over time
plt.plot(stressor_tss[21181,:])#basin of rio dos sinos? increase over time
plt.plot(stressor_tss[17918,:])#basin of rio Amazonas decrease onver time
plt.plot(stressor_tss[8506,:])#basin of Seine river increase over time
plt.plot(stressor_tss[9528,:])#basin of Rhone river decrease over time
plt.plot(stressor_tss[8156,:])#basin of cotentin catchment is overall the same over time, negative values...
plt.plot(stressor_tss[8068,:])#1 cell basin on the sea shore in netherlands, negative values, somehow constant
world = np.nanmean(stressor_tss,axis = 0)
plt.plot(time, world[:])#timeserie for the whole world




#we want to estimate stressor = f(t) and Dstressor over the period
#use numerical integration = calculate gradient and afterwards numerical integral with trapezoidal rule. seems useless. 
#refined solution: gaussian process regression + integration?
#we want to develop a surrogate of the response function of the stressor to time in the catchment. with uncertainty estimate

#simplest method = calculate the cumulated variation over time. 

#stressor_nonan_diff = np.sum(np.diff(stressor_tss_nonan, axis = 1), axis = 1)
stressor_diff = np.sum(np.diff(stressor_tss_mask, axis = 1), axis = 1)

plt.plot(stressor_diff)

#verificiation: seems correct
#plt.plot(stressor_tss[9528,:])#basin of Rhone river decrease over time
#stressor_diff[9528]

#visualization of the distribution of stressor variation values accross spatial units
np.mean(stressor_diff)

stats.scoreatpercentile(stressor_diff,95)

stats.scoreatpercentile(stressor_diff,5)

histogram_m = plt.hist(stressor_diff, bins = np.arange(-25,25,1))

#alternative approach to calculate  the stressor variation: numerical integration
# conclusion: we obtain the same results as previous method
gradient = np.gradient(stressor_tss, axis = 1)

integral_trapz = np.trapz(gradient, axis = 1)

plt.plot(integral_trapz)

histogram_m = plt.hist(integral_trapz)

boxplot = plt.boxplot(integral_trapz)

np.nanmean(integral_trapz)

stats.scoreatpercentile(integral_trapz,95)

stats.scoreatpercentile(integral_trapz,5)

#we could use the gradient array to interpolate a model and integrate it over the time period



#create a map to visualize the GHW variation 




#open basin delineation: the spatial resolution can be changed here.
spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
#setclone('data/catchments.map')

spatial_unit =pcr2numpy(spatial_unit_map,-2147483648)#this filling value is the filling value resulting from direct transformation of the map into array

spatial_unit_mask = ma.masked_values(spatial_unit, -2147483648)

#we want to fill this map with integral values
map_Dstressor = spatial_unit_mask.copy()

map_Dstressor.shape

#we need the pointer array to fill the map
pointer_array = np.load('output/full_catchment_filter_array.npy', allow_pickle = True)

#we put in the map the values corresponding for each catchments usign the pointer
n_max = pointer_array.shape[0]
for i in range(n_max):
    map_Dstressor[pointer_array[i,0],pointer_array[i,1]] = stressor_diff[i]

#verification ok
# integral_trapz[9528]
# map_Dstressor[pointer_array[9528,0],pointer_array[9528,1]]


#we  create the map with stressor variation values for the spatial units
plt.imshow(map_Dstressor, vmin = stats.scoreatpercentile(stressor_diff,5), vmax = stats.scoreatpercentile(stressor_diff,95))


#export results (does not work)
# setclone('data/catchments.map')
# map_out = numpy2pcr(Scalar, ma.getdata(map_Dstressor), mv = -2147483648)
# report(map_out, 'groundwaterhead_variation_catchment_scale.map')