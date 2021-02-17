# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:16:40 2021

@author: easpi
"""


import os
#from pcraster import readmap, pcr2numpy
#from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy, numpy2pcr, setclone
import numpy as np
from numpy import isnan, ma, trapz
from matplotlib import pyplot as plt
from scipy import stats
from pandas import read_excel, to_numpy
#import scipy # more intergation tools

#get hlep np.info(obj)

#input directory
os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')

#import aggregated stressor timseries
fn ='output/GW_ALL.xlsx'#aggregated for catchment.map

stressor_tss = read_excel(fn, "Sheet1", header = 1, names = None, keep_default_na=1)
stressor_tss_big = np.asarray(stressor_tss)
spatial_unit_ID = stressor_tss_big[:,0].copy()


fn1 ='output/GW_ALL_IND.xlsx'#aggregated for catchment.map
stressor_tss_small_frame = read_excel(fn1, "Sheet1", header = 1, names = None, keep_default_na=1)
stressor_tss_small = np.asarray(stressor_tss_small_frame)
stressor_tss_small.shape#2568 small catchments
stressor_tss_big.shape#19974 big catchments
stressor_tss = np.concatenate((stressor_tss_big, stressor_tss_small),  axis = 0)
stressor_tss.shape#completearray  with all catchments. still some catchments are missing.

#we have to put a mask to avoid problems with missing values

mask = np.isnan(stressor_tss)

stressor_tss_mask = ma.masked_where(mask, stressor_tss)

nanID = np.unique(np.where(mask == 1)[0])
len(nanID)#1573
ID = np.unique(np.where(mask == 0)[0])
len(ID)#22542

#some plots to visualize what is happening in several catchments

plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 21723)][0,1:])#basin of Paraná river, increase GWH over time
plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 21181)][0,1:])#basin of rio dos sinos? increase over time
plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 17918)][0,1:])#basin of rio Amazonas decrease onver time
plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 8506)][0,1:])#basin of Seine river increase over time
plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 9528)][0,1:])#basin of Rhone river decrease over time
plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 8156)][0,1:])#basin of cotentin catchment is overall the same over time, negative values...
plt.plot(stressor_tss[np.where(stressor_tss[:,0] == 8068)][0,1:])#1 cell basin on the sea shore in netherlands, negative values, somehow constant

#we want to estimate stressor = f(t) and Dstressor over the period
#use numerical integration = calculate gradient and afterwards numerical integral with trapezoidal rule. seems useless. 
#refined solution: gaussian process regression + integration?
#we want to develop a surrogate of the response function of the stressor to time in the catchment. with uncertainty estimate

#simplest method = calculate the cumulated variation over time. 

#stressor_nonan_diff = np.sum(np.diff(stressor_tss_nonan, axis = 1), axis = 1)

stressor_diff = np.sum(np.diff(stressor_tss_mask[:,1:], axis = 1), axis = 1)
#the id was removed, think about putting it back
plt.plot(stressor_diff)


#visualization of the distribution of stressor variation values accross spatial units
np.nanmean(stressor_diff)

stats.scoreatpercentile(stressor_diff,95)

stats.scoreatpercentile(stressor_diff,40)

stats.scoreatpercentile(stressor_diff,5)

histogram_m = plt.hist(stressor_diff, bins = np.arange(-25,25,1))

#alternative approach to calculate  the stressor variation: numerical integration
# conclusion: we obtain the same results as previous method
gradient = np.gradient(stressor_tss_mask[:,1:], axis = 1)

integral_trapz = np.trapz(gradient, axis = 1)

plt.plot(integral_trapz)

histogram_m = plt.hist(integral_trapz)

boxplot = plt.boxplot(integral_trapz)

np.nanmean(integral_trapz)

stats.scoreatpercentile(integral_trapz,95)

stats.scoreatpercentile(integral_trapz,5)

#we could use the gradient array to interpolate a model and integrate it over the time period



#create a map to visualize the GHW variation 




#we add the id index at the end of the stressor diff array
s = np.stack((stressor_tss[:,0],stressor_diff), axis = 1)
#a[catchment, :] = [catchment ID, Dstressor]


#open basin delineation: the spatial resolution can be changed here.
spatial_unit_map= readmap('data/catchments.map')#the catchment map corresponds to the river basin
#setclone('data/catchments.map')

spatial_unit =pcr2numpy(spatial_unit_map,-2147483648)#this filling value is the filling value resulting from direct transformation of the map into array

spatial_unit_mask = ma.masked_values(spatial_unit, -2147483648)

#we want to fill this map with integral values

#map_init = np.zeros(spatial_unit.shape)

#map_Dstressor = ma.array(map_init, mask = ma.getmask(spatial_unit_mask))

map_Dstressor = spatial_unit

map_Dstressor.shape

#we need the pointer array to fill the map
pointer_array = np.load('output/full_catchment_filter_array.npy', allow_pickle = True)

#we put in the map the values corresponding for each catchments usign the pointer
n_max = pointer_array.shape[0]

for i in range(n_max):
    if i in s[:,0]:#for what ever catchment, search if catchment ID exist
        map_Dstressor[pointer_array[i,0],pointer_array[i,1]] = s[np.where(s[:,0] == i)][0,1]
    else:
        map_Dstressor[pointer_array[i,0],pointer_array[i,1]] = -2147483648
#verification ok
# map_Dstressor[pointer_array[9528,0],pointer_array[9528,1]]
# s[np.where(s[:,0] == 9528)][0,1]

map_Dstressor_mask_all = ma.masked_values(map_Dstressor, -2147483648)


#we  create the map with stressor variation values for the spatial units
plt.imshow(map_Dstressor_mask_all, vmin = stats.scoreatpercentile(stressor_diff,5), vmax = stats.scoreatpercentile(stressor_diff,95))


#export results (does not work)
# setclone('data/catchments.map')
# map_out = numpy2pcr(Scalar, ma.getdata(map_Dstressor), mv = -2147483648)
# report(map_out, 'groundwaterhead_variation_catchment_scale.map')