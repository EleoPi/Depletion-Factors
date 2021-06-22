# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:32:31 2021
the purpose of this main program is to calculate the stressor variation and export results to netcdf.

TODO
- make map  consumption integral - export map  ok

- remove natural run contribution to stressors tss: SM is missing.
- check reference value

- calculate uncertainty values for the average FF : calcualte the trapz error for consumption and stressors with timeseries e = ndimt.dt^3.(max (grad3 S)) create formula

- calculate uncertainty for marginal with standard deviation of the mean
- save fate factors marginal map, matrix with the catchment ID, low and high estimate? or maps - > save to tiff??



@author: easpi
"""


import os
from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy
import numpy as np
from numpy import isnan, ma, load, trapz
from matplotlib import pyplot as plt
import tifffile
from scipy import stats
from tabulate import tabulate
import plotly.express as px#package uninstalled
import pandas as pd
import seaborn as sns
# import time

os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')#where did you put the modules
import Input
import module
import module_gwhead
import module_gwstorage
import module_gw_depletion
import module_evapotranspiration
import module_soil_moisture
import module_discharge
import module_consumption
import spatial_unit_delineation
#set directories

# input directory
os.chdir(Input.inputDir)


#spatial unit definition----------------------------------------------------

d = Dataset(Input.inputDir +'/'+ "domesticWithdrawal_annualTot_out_1960to2010_b.nc")#only to extract lat and lon
latitude = d.variables['latitude'][:]
longitude = d.variables['longitude'][:]

spatial_unit = tifffile.imread(Input.inputDir + '/' + Input.fn_catchmentsValerio)
spatial_unit = ma.masked_values(spatial_unit, -3.40282e+38)#some values are weird, they were filtered from the index array

idlist= np.unique(spatial_unit)
idlist = idlist[3:]

#filter_array = np.load(Input.outputDir +'/'+ Input.fn_filter_catchments_valerio, allow_pickle = 1) 
filter_ = spatial_unit_delineation.index_filter(spatial_unit)
filter_array = np.array(filter_[0], dtype=list)  # conversion to array
filter_array= filter_array[3:,:]


#area
area = pcr2numpy(readmap(Input.inputDir +'/'+ Input.fn_area_map), mv = 1e20)
areac = module.weight(filter_array, idlist, spatial_unit)
weights = areac/np.sum(areac[:-1])
weights[-1]=0
np.sum(weights)







#calculate aggregated values-------------------------- ---
return_flow_irrigation = module_consumption.aggregate_return_flow_irrigation(filter_array)

return_flow_non_irrigation = module_consumption.aggregate_return_flow_non_irrigation(filter_array)

abstraction = module_consumption.aggregate_abstraction(filter_array)

#use filtered values
consumption = module_consumption.calculate_net_consumption(abstraction, return_flow_irrigation[1], return_flow_non_irrigation[1], idlist)



#filter out catchments with small consumption and create a boolean filter
integral = np.sum(consumption, axis = 1)#careful with the time conversion: consumption volume
threshold = stats.scoreatpercentile(integral[integral>0], per=25)
integraltemp = ma.masked_less_equal(integral, threshold)
np.sum(integraltemp.mask)
mask = ma.getmask(integraltemp)#this is what we filter out








#calculate aggregatedvalues groundwater head-----------------------------------
h = module_gwhead.aggregate_wavg(idlist, filter_array)
h_yr = module.convert_month_to_year_avg(h)







#calculate total evapotranspiration outputs---------------------------------

et = module_evapotranspiration.aggregate(idlist, filter_array)
et_yr = module.convert_month_to_year_sum(et)




#calculate aggregated values groundwater storage-------------------------- ---
gws = module_gwstorage.aggregate(idlist, filter_array)#monthly
gws_yr = module.convert_month_to_year_avg(gws)





#calculate aggregated values soil moisture-------------------------- ---
sm_yr = module_soil_moisture.aggregate(idlist, filter_array)




#calculate aggregated values discharge-------------------------- ---
q = module_discharge.aggregate(idlist, filter_array)
q_yr = module.convert_month_to_year_avg(q)






# #load results human natural----------------------------------------------------------


# d  = Dataset(Input.outputDir + '/'+ 'human-natural' + '/'+ 'consumption_1960to2010_basinV.nc')
# consumption = d.variables['net water consumption'][:]#m3/yr
# consumption = np.append(consumption, np.full((1,consumption.shape[1]), 9.96921e+36), axis = 0)
# consumption = ma.masked_values(consumption, 9.96921e+36)


# d  = Dataset(Input.outputDir + '/'+ 'human-natural' + '/'+ 'groundwater_storage_human-natural_1960to2010_basinV.nc')
# gws = d.variables['groundwater storage'][:]#m3
# gws = np.append(gws, np.full((1,gws.shape[1]), 9.96921e+36), axis = 0)
# gws = ma.masked_values(gws, 9.96921e+36)
# gws_yr = module.convert_month_to_year_avg(gws)#it is average not sum"""
# #depletion_yr = np.diff(gws_yr)#correct


# d1  = Dataset(Input.outputDir + '/'+ 'human-natural' + '/'+ 'discharge_outlet_human-natural1960to2010_basinV.nc')
# q1 = d1.variables['discharge flow'][:]#m3/s
# q1 = np.append(q1, np.full((1,q1.shape[1]), 9.96921e+36), axis = 0)
# q1 = ma.masked_values(q1, 9.96921e+36)
# q_yr1 = module.convert_month_to_year_avg(q1)#m3/s


# d  = Dataset(Input.outputDir + '/'+  'discharge_outlet_human-natural-rev1960to2010_basinV.nc')
# q = d.variables['discharge flow'][:]#m3/s
# q = np.append(q, np.full((1,q.shape[1]), 9.96921e+36), axis = 0)
# q = ma.masked_values(q, 9.96921e+36)
# q_yr = module.convert_month_to_year_avg(q)#m3/s


# d  = Dataset(Input.outputDir + '/'+ 'human-natural' + '/'+ 'groundwater_head_wavg_human-natural_1960to2010_basinV.nc')
# h = d.variables['groundwater head'][:]#m
# h = np.append(h, np.full((1,h.shape[1]), 9.96921e+36), axis = 0)
# h = ma.masked_values(h, 9.96921e+36)
# h_yr = module.convert_month_to_year_avg(h)



# d  = Dataset(Input.outputDir + '/'+'human-natural' + '/'+  'evapotranspiration_human-natural_1960_2004_basinV.nc')
# et = d.variables['evapotranspiration'][:]#m
# et = np.append(et, np.full((1,et.shape[1]), 9.96921e+36), axis = 0)
# et = ma.masked_values(et, 9.96921e+36)
# et_yr = module.convert_month_to_year_sum(et)


# d  = Dataset(Input.outputDir + '/'+'human-natural' + '/'+  'soil_moisture_human-natural_1960to2010_basinV.nc')
# sm = d.variables['soil moisture'][:]#m
# sm = np.append(sm, np.full((1,sm.shape[1]), 9.96921e+36), axis = 0)
# sm_yr = ma.masked_values(sm, 9.96921e+36)







#average map

consumption_var = np.sum(np.diff(consumption, axis = 1),axis = 1)
consumption_var = ma.masked_where(isnan(consumption_var) == 1, consumption_var)
consumption_ref = np.mean(consumption, axis = 1)
consumption_ref = ma.masked_where(isnan(consumption_ref) == 1, consumption_ref)
consumption_change = consumption_var / consumption_ref
np.average(consumption_change, weights = weights)
np.average(consumption_change[mask==0], weights = weights[mask==0])

map_consumption = module.make_map(consumption_change, filter_array, idlist, spatial_unit)
module.new_map_netcdf(Input.outputDir + '/'+ 'map_consumption_change_1960to2010', map_consumption, 'consumption_change', '%', latitude, longitude)
plt.imshow(map_consumption)
plt.hist(consumption_change)
np.max(consumption_change)
np.min(consumption_change)












#calcualte average fate factor with human - natural variables-------------------------------- 




c_dif = consumption[:,:]

q_dif_yr = q_yr[:,:]
q_ff_a= module.FF_sum(c_dif, q_dif_yr*365*24*3600, filter_array, spatial_unit, 'discharge_average_human-natural_', '(m3/yr)/(m3/yr)', latitude, longitude, idlist)

stats.scoreatpercentile(q_ff_a[1][q_ff_a[1].mask==0], 25)
stats.scoreatpercentile(q_ff_a[1][q_ff_a[1].mask==0], 50)
stats.scoreatpercentile(q_ff_a[1][q_ff_a[1].mask==0], 75)




et_dif_yr= et_yr[:,:45]
et_ff_a = module.FF_sum(c_dif[:,:45], et_dif_yr, filter_array, spatial_unit, 'et_average_human-natural_', 'm3/yr/m3/yr', latitude, longitude, idlist)

stats.scoreatpercentile(et_ff_a[1][et_ff_a[1].mask==0], 25)
stats.scoreatpercentile(et_ff_a[1][et_ff_a[1].mask==0], 50)
stats.scoreatpercentile(et_ff_a[1][et_ff_a[1].mask==0], 75)




gws_dif_yr = gws_yr[:,:]
gws_ff_a = module.FF_sum(c_dif[:,:], gws_dif_yr, filter_array, spatial_unit, 'gws_average_human-natural_', 'm3/m3/yr', latitude, longitude, idlist)

stats.scoreatpercentile(gws_ff_a[1][gws_ff_a[1].mask==0], 25)
stats.scoreatpercentile(gws_ff_a[1][gws_ff_a[1].mask==0], 50)
stats.scoreatpercentile(gws_ff_a[1][gws_ff_a[1].mask==0], 75)


h_dif_yr = h_yr[:,:]
h_ff_a = module.FF_mean(c_dif, h_dif_yr, filter_array, spatial_unit, 'gw_head_average_human-natural_', 'm/m3/yr', latitude, longitude, idlist)

stats.scoreatpercentile(h_ff_a[1][h_ff_a[1].mask==0], 25)
stats.scoreatpercentile(h_ff_a[1][h_ff_a[1].mask==0], 50)
stats.scoreatpercentile(h_ff_a[1][h_ff_a[1].mask==0], 75)


sm_dif = sm_yr[:,:]
sm_ff_a = module.FF_sum(c_dif[:,:45], sm_dif, filter_array, spatial_unit, 'soil_moisture_average_human-natural_', 'm3/m3yr', latitude, longitude, idlist)

stats.scoreatpercentile(sm_ff_a[1][sm_ff_a[1].mask==0], 25)
stats.scoreatpercentile(sm_ff_a[1][sm_ff_a[1].mask==0], 50)
stats.scoreatpercentile(sm_ff_a[1][sm_ff_a[1].mask==0], 75)




#remove catchments underthreshold
c = module.exclude(consumption, mask)
q=module.exclude(q_yr*365*24*3600, mask)
g= module.exclude(gws_yr, mask)
sm=module.exclude(sm_yr, mask)
h = module.exclude(h_yr, mask)
et= module.exclude(et_yr, mask)

c_w  =np.sum(c)
q_w =np.sum(q)
g_w = np.sum(g[isnan(g)==0])
sm_w = np.sum(sm)
h_w = np.mean(np.average(h, weights = weights, axis = 0))             
et_w = np.sum(et_yr)

#gobal depletion factors
data = [["", "World DF average"],
        ["Discharge", q_w/c_w],
        ["Groundwater storage",g_w/c_w],
        ["Groundwater head m", h_w/c_w],
        ["Soil Moisture",sm_w/c_w],
        ["Evapotranspiration",et_w/c_w]
        ]


print(tabulate(data,headers='firstrow'))





#slope study with human - natural variables-------------------------------------------------------------------


#original values
c_dif = consumption

q_dif_yr = q_yr
q_slope = module.FF_lm(c_dif, q_dif_yr*365*24*3600,filter_array, spatial_unit, 'discharge recent', '(m3/yr)/(m3/yr)', latitude, longitude, idlist)


ET_dif_yr= et_yr
et_slope = module.FF_lm(c_dif[:,:45], ET_dif_yr[:,:45], filter_array, spatial_unit, 'et recent', 'm3/m3/yr', latitude, longitude, idlist)


gws_dif_yr = gws_yr
gws_slope = module.FF_lm(c_dif, gws_dif_yr, filter_array, spatial_unit, 'gws recent', 'm3/m3/yr', latitude, longitude, idlist)


h_dif_yr = h_yr
h_slope = module.FF_lm(c_dif, h_dif_yr, filter_array, spatial_unit, 'gw head recent', 'm/m3/yr', latitude, longitude, idlist)


sm_dif = sm_yr
sm_slope = module.FF_lm(c_dif[:,:45], sm_dif[:,:45], filter_array, spatial_unit, 'soil moisture recent', 'm3/m3/yr', latitude, longitude, idlist)


#good results
q_map_r2 = module.make_map(q_slope[0][:,2], filter_array, idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'q_map_r2.tif',q_map_r2)
plt.imshow(q_map_r2)


q_map_p = module.make_map(q_slope[0][:,3], filter_array,idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'q_map_p.tif',q_map_p)
plt.imshow(q_map_p)

#goodresults
gws_map_r2 = module.make_map(gws_slope[0][:,2], filter_array,idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'gws_map_r2.tif',gws_map_r2)
plt.imshow(gws_map_r2)
gws_map_p = module.make_map(gws_slope[0][:,3], filter_array,idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'gws_map_p.tif',gws_map_p)
plt.imshow(gws_map_p)

#very bad results
et_map_r2 = module.make_map(et_slope[0][:,2], filter_array, idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'et_map_r2.tif',et_map_r2)
plt.imshow(et_map_r2)
et_map_p = module.make_map(et_slope[0][:,3], filter_array,idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'et_map_p.tif',et_map_p)
plt.imshow(et_map_p)

h_map_r2 = module.make_map(h_slope[0][:,2], filter_array, idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'h_map_r2.tif',h_map_r2)
plt.imshow(h_map_r2)
h_map_p = module.make_map(h_slope[0][:,3], filter_array, idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'h_map_p.tif',h_map_p)
plt.imshow(h_map_p)

sm_map_r2 = module.make_map(sm_slope[0][:,2], filter_array, idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'sm_map_r2.tif',sm_map_r2)
plt.imshow(sm_map_r2)
sm_map_p = module.make_map(sm_slope[0][:,3], filter_array, idlist, spatial_unit)
tifffile.imsave(Input.outputDir + '/'+ 'sm_map_p.tif',sm_map_p)
plt.imshow(sm_map_p)



#table with coefficients of regressions
data = [["Weighed medians","slope","R2"," p", "error"],
        ["Discharge",np.median(q_slope[1][isnan(q_slope[1]) ==0]), np.median(q_map_r2[isnan(q_map_r2)==0]),np.median(q_map_p[q_map_p.mask ==0]),np.median(q_slope[2][q_slope[2].mask ==0])],
        ["Groundwater storage",np.median(gws_slope[1][isnan(gws_slope[1]) ==0]),np.median(gws_map_r2[gws_map_r2.mask==0]),np.median(gws_map_p[gws_map_p.mask ==0]),np.median(gws_slope[2][gws_slope[2].mask ==0])],
        ["Groundwater head", np.median(h_slope[1][isnan(h_slope[1]) ==0]), np.median(h_map_r2[h_map_r2.mask==0]),np.median(h_map_p[h_map_p.mask ==0]),np.median(h_slope[2][h_slope[2].mask ==0])],
        ["Evapotranspiration",np.median(et_slope[1][isnan(et_slope[1])==0]), np.median(et_map_r2[et_map_r2.mask==0]),np.median(et_map_p[et_map_p.mask ==0]), np.median(et_slope[2][et_slope[2].mask ==0])],
        ["Soil moisture",np.median(sm_slope[1][isnan(sm_slope[1])==0]), np.median(sm_map_r2[sm_map_r2.mask==0]),np.median(sm_map_p[sm_map_p.mask ==0]), np.median(sm_slope[2][sm_slope[2].mask ==0])]
        ]

print(tabulate(data,headers='firstrow'))


#we have too many outliers to be able to use average
#add other values
data = [["Weighed averages","slope","R2"," p", "error"],
        ["Discharge",np.average(q_slope[0][:-1,0], weights = weights[:-1]), np.average(q_slope[0][:-1,3], weights = weights[:-1]), np.average(q_slope[0][:-1,3], weights = weights[:-1]),np.average(q_slope[0][:-1,4], weights = weights[:-1])],
        ["Groundwater storage",np.average(gws_slope[0][:-1,0], weights = weights[:-1]), np.average(gws_slope[0][:-1,2], weights = weights[:-1]),np.average(gws_slope[0][:-1,3], weights = weights[:-1]),np.average(gws_slope[0][:-1,4], weights = weights[:-1])],
        ["Groundwater head", np.average(h_slope[0][:-1,0], weights = weights[:-1]), np.average(h_slope[0][:-1,2], weights = weights[:-1]), np.average(h_slope[0][:-1,3], weights = weights[:-1]), np.average(h_slope[0][:-1,4], weights = weights[:-1])],
        ["Evapotranspiration",np.average(et_slope[0][:-1,0], weights = weights[:-1]), np.average(et_slope[0][:-1,2], weights = weights[:-1]), np.average(et_slope[0][:-1,3], weights = weights[:-1]), np.average(et_slope[0][:-1,4], weights = weights[:-1])],
        ["Soil moisture",np.average(sm_slope[0][:-1,0], weights = weights[:-1]), np.average(sm_slope[0][:-1,2], weights = weights[:-1]), np.average(sm_slope[0][:-1,3], weights = weights[:-1]), np.average(sm_slope[0][:-1,4], weights = weights[:-1])]
        ]

print(tabulate(data,headers='firstrow'))



#marginal ff with moving average window----------------------------------------
#the last value of the moving average gradient is selected
#threshold has to be selected beforehand








c_avg = module.moving_average(consumption, 10) 
c_avg = module.exclude(c_avg, mask)

q_avg = module.moving_average(q_yr, 10)
q_avg = module.exclude(q_avg,mask)

gws_avg = module.moving_average(gws_yr, 10)
gws_avg = module.exclude(gws_avg,mask)

h_avg = module.moving_average(h_yr, 10)
h_avg = module.exclude(h_avg,mask)

et_avg = module.moving_average(et_yr, 10)
et_avg = module.exclude(et_avg,mask)

sm_avg = module.moving_average(sm_yr, 10)
sm_avg = module.exclude(sm_avg,mask)



q_ff_m = module.FF_grad(c_avg, q_avg*365*24*3600, filter_array, spatial_unit, 'Q', 'm3/yr/m3/yr', latitude, longitude, idlist)#exclude
gws_ff_m = module.FF_grad(c_avg, gws_avg, filter_array, spatial_unit, 'GWS', 'm3/yr/m3/yr', latitude, longitude, idlist)#exclude
h_ff_m = module.FF_grad(c_avg, h_avg, filter_array, spatial_unit, 'GWH', 'm/yr/m3/yr', latitude, longitude, idlist)#exclude
et_ff_m = module.FF_grad(c_avg, et_avg, filter_array, spatial_unit, 'ET', 'm3/yr/m3/yr', latitude, longitude, idlist)#exclude
sm_ff_m = module.FF_grad(c_avg, sm_avg, filter_array, spatial_unit, 'SM', 'm3/yr/m3/yr', latitude, longitude, idlist)#exclude




fig, ax = plt.subplots()  # Create a figure and an axes.
#ax.plot(np.sum(gws_avg, axis = 0), label='GWS')  # Plot some data on the axes.
#ax.plot(np.sum(q_avg, axis = 0), label='Q')  # Plot some data on the axes.
#ax.plot(np.sum(sm_avg, axis = 0), label='SM')  # Plot some data on the axes.
#ax.plot(np.average(h_avg, axis = 0, weights= weights), label='GWH')  # Plot some data on the axes.
#ax.plot(np.sum(et_avg, axis = 0), label='ET')  # Plot some data on the axes.
ax.plot(np.sum(c_avg, axis = 0), label='consumption') 
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('m3')  # Add a y-label to the axes.
ax.set_title( "Global evolution 1960/2010 (10yr average)")  # Add a title to the axes.
ax.legend()



#world marginal

c_w  = np.gradient(np.sum(c_avg, axis = 0))
q_w =np.gradient(np.sum(q_avg*365*3600*24, axis = 0))
g_w = np.gradient(np.sum(gws_avg, axis =0))
sm_w = np.gradient(np.sum(sm_avg, axis = 0))
h_w = np.gradient(np.average(h_avg, axis = 0, weights = weights))
et_w = np.gradient(np.sum(et_avg, axis = 0))

#low consumption watersheds are masked
data = [["", "World DF trend"],
        ["Discharge", q_w[-1]/c_w[-1]],
        ["Groundwater storage",g_w[-1]/c_w[-1]],
        ["Groundwater head m", h_w[-1]/c_w[-1]],
        ["Soil Moisture",sm_w[-1]/c_w[-1]],
        ["Evapotranspiration",et_w[-1]/c_w[-1]]
        ]


print(tabulate(data,headers='firstrow'))






#water balance check------------------------------------------
#compare cumsum with integral---------------------------------




balance  = (np.sum(consumption[:,:45]) + np.sum(q_yr[:,:45]*365*24*3600) + np.sum(gws_yr[:,:45])+ np.sum(et_yr)+np.sum(sm_yr))

balance / (np.sum(consumption[:,:45]))
#depltion -1,75: the depletion is overestimated 75%


area = module.weight(filter_array, idlist, spatial_unit)
weights = area/np.sum(area[:-1])
weights[-1]=0
np.sum(weights)


#total change in variables
data = [["Depletion", "No consumption", "Initial state"],
        ["Consumption km3", np.sum(consumption)/1e9,np.sum(np.diff(consumption, axis = 1))/1e9 ],
        ["Discharge km3", np.sum(q_yr*365*24*3600)/1e9,np.sum(np.diff(q_yr*365*24*3600, axis = 1))/1e9 ],
        ["Groundwater storage km3",np.sum(gws_yr)/1e9 , np.sum(np.diff(gws_yr, axis = 1))/1e9],
        ["Groundwater head m", np.average(np.mean(h_yr,axis = 1), weights = weights), np.average(np.sum(np.diff(h_yr, axis = 1), axis = 1), weights = weights)],
        ["Soil Moisture km3",np.sum(sm_yr)/1e9, np.sum(np.diff(sm_yr, axis = 1))/1e9],
        ["Evapotranspiration km3", np.sum(et_yr)/1e9, np.sum(np.diff(et_yr, axis = 1))/1e9]
        ]


print(tabulate(data,headers='firstrow'))












#tables of results-----------------------------------------------------------

area = module.weight(filter_array, idlist, spatial_unit)
weights = area/np.sum(area[:-1])
weights[-1]=0
np.sum(weights)



data4 = [["depletion factor", "Weighed median", "Weighed average"],
        ["Discharge", np.median(q_ff_a[1][q_ff_a[1].mask==0]),np.average(q_ff_a[0][:-1], weights = weights[:-1])],
        ["Groundwater storage", np.median(gws_ff_a[1][[gws_ff_a[1].mask==0]]),np.average(gws_ff_a[0][:-1], weights = weights[:-1])],
        ["Groudnwater head", np.median(h_ff_a[1][h_ff_a[1].mask==0]),np.average(h_ff_a[0][:-1], weights = weights[:-1])],
        ["Evapotranspiration", np.median(et_ff_a[1][et_ff_a[1].mask==0]),np.average(et_ff_a[0][:-1], weights = weights[:-1])],
        ["Soil moisture", np.median(sm_ff_a[1][sm_ff_a[1].mask==0]), np.average(sm_ff_a[0][:-1], weights = weights[:-1])]
        ]
print(tabulate(data4,headers='firstrow'))



#Depletion trend smooth avg
data5 = [["depletion trend"," weighted median", "weighted average"],
        ["Discharge", np.median(q_ff_m[1][q_ff_m[1].mask==0]), np.average(q_ff_m[0][:-1], weights = weights[:-1])],
        ["Groundwater storage", np.median(gws_ff_m[1][[gws_ff_m[1].mask==0]]),np.average(gws_ff_m[0][:-1], weights = weights[:-1])],
        ["Groudnwater head", np.median(h_ff_m[1][h_ff_m[1].mask==0]),np.average(h_ff_m[0][:-1], weights = weights[:-1])],
        ["Evapotranspiration", np.median(et_ff_m[1][et_ff_m[1].mask==0]), np.average(et_ff_m[0][:-1], weights = weights[:-1])],
        ["Soil moisture", np.median(sm_ff_m[1][sm_ff_m[1].mask==0]),np.average(sm_ff_m[0][:-1], weights = weights[:-1])]
        ]
print(tabulate(data5,headers='firstrow'))









#DF analysis -------------------------------------------------------------- 


matrix = np.stack((areac[mask == 0], q_ff_a[0][mask == 0], gws_ff_a[0][mask == 0], sm_ff_a[0][mask == 0], et_ff_a[0][mask == 0]), axis=1)

df=pd.DataFrame(matrix, columns = ['area','df q','df gws','df sm','df et' ])

sns.pairplot(df)

plt.scatter(df["df et"]/1e9, df["df sm"]/1e9,marker ='o', color='black')
plt.plot(df["df gws"]/1e9, df["df sm"]/1e9,'o', color='black')
plt.scatter(df["df q"], df["df sm"]/1e9,marker ='o', color='black')

plt.scatter(df["df et"]/1e9, df["df gws"]/1e9,marker ='o', color='black')
plt.scatter(consumption_change[mask==0], df["df gws"],marker ='o', color='black')
plt.boxplot(df["df gws"])
plt.boxplot(df["area"])


np.average(q_ff_a[0][consumption_change>1])
np.average(sm_ff_a[0][et_ff_a[0]>0], weights = weights[et_ff_a[0]>0])
np.average(sm_ff_a[0][et_ff_a[0]>0])








# case studies 


n = 14842#3047
m = "Mississipi basin"



n = 26269#8380
m = "Paraná river"


n = 16987
m = "Rio Grande"



#location of the basin the the DF matrix
k = np.where(idlist == n)[0][0]

#catchment area
weights[k]

fig, ax = plt.subplots()  # Create a figure and an axes.
#ax.plot(q_aggregated[k], label='q human - natural')  # Plot some data on the axes.
ax.plot(q_yr[k]*365*24*3600/1e9, label='q human - natural')  # Plot some data on the axes.
#ax.plot(q_human_yr[k]*365*24*3600/1e9, label='q human')  # Plot some data on the axes.
#ax.plot(q_dif_yr[k]*365*24*3600/1e9, label='q dif human')  # Plot some data on the axes.
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()



fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(consumption[k]/1e9, label='consumption')  # Plot some data on the axes.
#ax.plot(c_dif[k]/1e9, label='consumption diff')
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()


fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(et_yr[k], label='ET human - natural')  # Plot some data on the axes.
#ax.plot(et_human_yr[k], label='ET human')  # Plot some data on the axes.
#ax.plot(ET_dif_yr[k], label='ET dif human')  # Plot some data on the axes.
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('m3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()


fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gws_yr[k], label='gws human - natural')  # Plot some data on the axes.
#ax.plot(gws_human_yr[k], label='gws human')  # Plot some data on the axes.
#ax.plot(gws_dif_yr[k], label='gws dif human')  # Plot some data on the axes.
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('m3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(sm_yr[k], label='sm human - natural')  # Plot some data on the axes.
#ax.plot(sm_human_yr[k], label='sm human')  # Plot some data on the axes.
#ax.plot(sm_dif_yr[k], label='sm dif human')  # Plot some data on the axes.
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('m3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(h_yr[k], label='h human - natural')  # Plot some data on the axes.
#ax.plot(h_human_yr[k], label='h human')  # Plot some data on the axes.
#ax.plot(h_dif_yr[k], label='h dif human')  # Plot some data on the axes.
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('m')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()





data = [["","DF average", "DF marginal","slope","p","R2","int consumption","grad consumption"],
        ["discharge",q_ff_a[0][k],q_ff_m[0][k],q_slope[0][k,0],q_slope[0][k,3], q_slope[0][k,2]],
        #["discharge bis",q_ff_a_bis[0][k],q_ff_m_yr[0][k],q_slope_bis[:,0][k],q_slope_bis[:,3][k], q_slope_bis[:,2][k]],
        ["GW storage",gws_ff_a[0][k],gws_ff_m[0][k],gws_slope[0][k,0],gws_slope[0][k,3], gws_slope[0][k,2]],
        ["GW head", h_ff_a[0][k],h_ff_m[0][k],h_slope[0][k,0],h_slope[0][k,3], h_slope[0][k,2]],
        ["evapotranspiration", et_ff_a[0][k],et_ff_m[0][k],et_slope[0][k,0],et_slope[0][k,3], et_slope[0][k,2]],
        ["soil moisture", sm_ff_a[0][k],sm_ff_m[0][k],sm_slope[0][k,0],sm_slope[0][k,3], sm_slope[0][k,2]]
        ]
print(tabulate(data,headers='firstrow'))


