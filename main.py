# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:32:31 2021
the purpose of this main program is to calculate the stressor variation and export results to netcdf.

TODO
- show change/mean for consumption at world scale - ok
- discharge aggregation: use outlet map to identify the discharge value - ok 
- remove natural run contribution
-- reminder to inge: send natural run results, send catchment maps, check return flows - ok
- run results at catchment scale with new catchment map: inge and valerio will provide ldd and subwatershed maps - ok valerio map
- run results at  subwatershed scale with new catchment map: inge and valerio will provide ldd and subwatershed maps - cancelled

- calcualte fate factors from 1961 till 2010 : filter 1960 and possibly 2005-2010 due to poor irrigaiton return flow
- calculate FF values: use timeseries  directly to adjust the timespans -> redo all calcs of maps!! -> put in the main it is better
- calcualte FF maps: divide stressor map by consumption map, export map

- calculate uncertainty values for the average FF : calcualte the trapz error for consumption and stressors with timeseries e = ndimt.dt^3.(max (gradgradgrad S)) create formula
- save fate factors average map, matrix with the catchment ID, low and high estimate? or maps -> define best way to save the results.

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
# import time

os.chdir('C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/')#where did you put the modules
import Input
import module
import module_gwhead
import module_gwstorage
import module_evapotranspiration
import module_soil_moisture
import module_discharge
import module_consumption
#set directories

# input directory
os.chdir(Input.inputDir)


#spatial unit definition----------------------------------------------------

d = Dataset(Input.inputDir +'/'+ Input.fn_catchmentsldd)#only to extract lat and lon
# temp_map = d.variables['Catch_IDD'][:]
latitude = d.variables['lat'][:]
longitude = d.variables['lon'][:]
#spatial_unit = temp_map

spatial_unit = tifffile.imread(Input.inputDir + '/' + Input.fn_catchmentsValerio)
spatial_unit = ma.masked_equal(spatial_unit, -2147483647)

idlist= np.unique(spatial_unit)

# spatial_unit = Dataset(Input.fn_catchmentsinecoregions).variables["Catchments_Extract_FEOW.tif"][:]

# filter_array, idlist = index_filter(spatial_unit)

# np.save(Input.outputDir + '/'+  'filter_array_catchment_ldd', filter_array)

 
filter_array = np.load(Input.outputDir +'/'+ Input.fn_filter_catchments_valerio, allow_pickle = 1) 

area = pcr2numpy(readmap(Input.fn_area_map), mv = 1e20)


#calculate aggregated values discharge-------------------------- ---
return_flow_irrigation = module_consumption.aggregate_return_flow_irrigation(spatial_unit, filter_array)

return_flow_non_irrigation = module_consumption.aggregate_return_flow_non_irrigation(spatial_unit, filter_array)

abstraction = module_consumption.aggregate_abstraction(spatial_unit, filter_array)

#use filtered values
consumption = module_consumption.calculate_net_consumption(abstraction, return_flow_irrigation[1], return_flow_non_irrigation[1], spatial_unit)

consumption_int = np.trapz(consumption, axis = 1)

#consumption_variation = module.calculate_variation(net_consumption, spatial_unit, filter_array, latitude, longitude, "consumption variation", "m3")

#consumption_gradient = module.calculate_grad_avg(net_consumption, spatial_unit, filter_array, latitude, longitude, "consumption gradient", "m3")

#plt.imshow(consumption_gradient[0])


#module.new_map_netcdf(Input.outputDir +'/'+"test2_net_consumption_change_normalized_" + Input.name_timeperiod + '_' + Input.name_scale, map_change, "net consumption variation normlalized", "m3", latitude, longitude)


world_abstraction = np.sum(abstraction, axis = 0)
world_return_irr = np.sum(return_flow_irrigation[1], axis = 0)
world_return_non_irr = np.sum(return_flow_non_irrigation[1], axis = 0)
world_return_flow = world_return_irr + world_return_non_irr
world_consumption = np.sum(consumption, axis = 0)

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(world_abstraction/1e9, label='abstraction')  # Plot some data on the axes.
#ax.plot(world_return_flow/1e9, label='return_flow')  # Plot some data on the axes.
ax.plot(world_return_flow/1e9, label='return flow total')  # Plot some data on the axes.
ax.plot(world_return_irr/1e9, label='return flow irrigation')  # Plot some data on the axes.
ax.plot(world_return_non_irr/1e9, label='return flow non-irrigation')  # Plot some data on the axes.
ax.plot(world_consumption/1e9, label='consumption')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title("World consumption 1960/2010")  # Add a title to the axes.
ax.legend()  # Add a legend.






#to betested
# # #test map ok
# c_var = consumption_variation[1]
# c_var_map = consumption_variation[0] 
# c_var_map[filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]/1e9
# c_var[idlist == 8506]/1e9

# # #test map  ok
# c_grad = consumption_gradient[1]
# c_grad[idlist == 8506]
# consumption_gradient[0][filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]

# # #test value ok
# np.mean(np.gradient(net_consumption[idlist ==8506][0])[-10:])/1e9
# plt.plot(net_consumption[idlist == 8506][0])

# np.mean(c_var)/1e9




#calculate aggregatedvalues groundwater head
gwh_aggregated = module_gwhead.aggregate_wavg(spatial_unit, filter_array)

#gwh_variation = module_gwhead.calculate_variation(gwh_aggregated, spatial_unit, filter_array)

#gwh_gradient = module_gwhead.calculate_grad_avg(gwh_aggregated, spatial_unit, filter_array)

#gwh_variation =  module.calculate_variation(gwh_aggregated, spatial_unit, filter_array, latitude, longitude, 'groundwater head variation wavg', 'm')

#gwh_gradient =  module.calculate_grad_avg(gwh_aggregated, spatial_unit, filter_array, latitude, longitude, 'groundwater head gradient wavg', 'm')

#gwh_aggregated_mean = module_gwhead.aggregate_mean(spatial_unit, filter_array)


gwh_yr = module.convert_month_to_year_avg(gwh_aggregated)

gwh_world = np.mean(gwh_yr, axis = 0)

np.mean(np.gradient(gwh_world)[-10:])

#test ok
# grad_gwh = gwh_gradient[1]
# grad8506 = gwh_gradient[1][idlist == 8506]
# plt.plot(gwh_aggregated[idlist == 8506][0])
# yrgwh = module.convert_month_to_year_avg(gwh_aggregated)
# np.mean(np.gradient(yrgwh[idlist ==8506][0])[-10:])
# plt.plot(yrgwh[idlist == 8506][0])

#test ok
# gwh_var = gwh_variation[1]
# gwh_var_map = gwh_variation[0]
# gwh_var_map[filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]
# gwh_var[idlist == 8506]

#calculate total evapotranspiration outputs---------------------------------

ET_aggregated = module_evapotranspiration.aggregate(spatial_unit, filter_array)

ET_ff_int = module.FF_integral(consumption, ET_aggregated, filter_array, spatial_unit, 'FF_ET_avg_integral')


#ET_variation = module_evapotranspiration.calculate_variation(ET_aggregated, spatial_unit, filter_array)

#ET_gradient = module_evapotranspiration.calculate_grad_avg(ET_aggregated, spatial_unit, filter_array)

ET_world = np.sum(ET_aggregated, axis = 0)

np.mean(np.gradient(ET_world)[-10:])/1e9
#test ok
# var = ET_variation[1]
# var_map = ET_variation[0]
# var_map[filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]
# var[idlist == 8506]

#test ok
# grad = ET_gradient[1]
# ET_gradient[1][idlist == 8506]
# plt.plot(ET_aggregated[idlist == 8506][0])
# yrgwh = module.convert_month_to_year_avg(gwh_aggregated)
# np.mean(np.gradient(ET_aggregated[idlist ==8506][0])[-10:])
# plt.plot(ET_aggregated[idlist == 8506][0])


#calculate aggregated values groundwater storage-------------------------- ---
gws_aggregated = module_gwstorage.aggregate(spatial_unit, filter_array)

gws_yr = module.convert_month_to_year_avg(gws_aggregated)
gws_ff_lm = module.FF_lm(consumption, gws_yr, filter_array, spatial_unit, 'GWS')

gws_ff_integral = module.FF_integral(consumption, gws_aggregated/12, filter_array, spatial_unit, 'GWS')
gws_ff_int = np.trapz(gws_aggregated, axis = 1)/np.trapz(consumption, axis = 1)*(51/612)

gws_ff_grad = module.FF_grad(consumption, gws_yr, filter_array, spatial_unit, 'GWS')



#gws_aggregated_w = module_gwstorage.aggregate_weight(spatial_unit, filter_array)

#gws_variation = module_gwstorage.calculate_variation(gws_aggregated, spatial_unit, filter_array)

#gws_gradient = module_gwstorage.calculate_grad_avg(gws_aggregated, spatial_unit, filter_array)

#gws_variation_w = module.calculate_variation(gws_aggregated_w, spatial_unit, filter_array, latitude, longitude, 'groundwater storage variation wavg', 'm3')



gws_yr = module.convert_month_to_year_avg(gws_aggregated)

gws_world = np.sum(gws_yr, axis = 0)

np.mean(np.gradient(gws_world)[-10:])/1e9

# #test ok
# gws_var = gws_variation[1]
# gws_var_map = gws_variation[0]
# gws_var_map[filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]
# gws_var[idlist == 8506]

# grad_gws = gws_gradient[1]
# gws_gradient[1][idlist == 8506]
# plt.plot(gws_aggregated[idlist == 8506][0])
# yrgws = module.convert_month_to_year_avg(gws_aggregated)
# np.mean(np.gradient(yrgws[idlist ==8506][0])[-10:])
# plt.plot(yrgws[idlist == 8506][0])

#calculate aggregated values soil moisture-------------------------- ---
sm_aggregated = module_soil_moisture.aggregate(spatial_unit, filter_array)

#sm_variation = module_soil_moisture.calculate_variation(sm_aggregated, spatial_unit, filter_array)

#sm_gradient = module_soil_moisture.calculate_grad_avg(sm_aggregated, spatial_unit, filter_array)

sm_world = np.sum(sm_aggregated, axis = 0)



# #test map ok
# sm_var = sm_variation[1]
# sm_var_map = sm_variation[0]
# sm_var_map[filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]
# sm_var[idlist == 8506]

# grad_sm = sm_gradient[1]
# sm_gradient[1][idlist == 8506]
# np.mean(np.gradient(sm_aggregated[idlist ==8506][0])[-10:])
# plt.plot(sm_aggregated[idlist == 8506][0])


#calculate aggregated values discharge-------------------------- ---
q_aggregated = module_discharge.aggregate(spatial_unit, filter_array)

#q_variation = module_discharge.calculate_variation(q_aggregated, spatial_unit, filter_array)#not normalized

#q_gradient = module_discharge.calculate_grad_avg(q_aggregated, spatial_unit, filter_array)



world_q0 = np.sum(q_aggregated, axis = 0)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(world_q0, label='discharge')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('m3/s')  # Add a y-label to the axes.
ax.set_title("World discharge 1960/2010")  # Add a title to the axes.
ax.legend()  # Add a legend.

q_yr = module.convert_month_to_year_avg(q_aggregated)

world_q = np.sum(q_yr, axis = 0)
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(world_q, label='discharge')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('m3/s')  # Add a y-label to the axes.
ax.set_title("World discharge 1960/2010")  # Add a title to the axes.
ax.legend()  # Add a legend.

#overall gradient


q_world_m = np.sum(q_yr, axis = 0)

np.mean(np.gradient(q_world_m)[-10:])/1e9





#to betested
# # #test map ok
# q_var = q_variation[1]
# q_var_map = q_variation[0]
# q_var_map[filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]/1e9
# q_var[idlist == 8506]/1e9

# #test map  ok
# grad_q = q_gradient[1]
# q_gradient[1][idlist == 8506]
# q_gradient[0][filter_array[idlist == 8506][0][0], filter_array[idlist == 8506][0][1]]

# #test value ok
# yrq = module.convert_month_to_year_avg(q_aggregated)
# yrq_m = ma.masked_where(isnan(yrq) == 1 , yrq)
# np.mean(np.gradient(yrq[idlist ==8506][0])[-10:])
# plt.plot(yrq[idlist == 8506][0])






#amazon river 17918

n = 17918
m = "Amazone basin"

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(consumption[idlist == n][0]/1e9, label='consumption')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

consumption_variation[1][idlist == n]/1e9

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(q_aggregated[idlist == n][0]/1e9, label='discharge')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3/s')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

q_variation[1][idlist == n]/1e9

yrq = module.convert_month_to_year_avg(q_aggregated)

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(yrq[idlist == n][0]/1e9, label='discharge')  # Plot some data on the axes.
ax.set_xlabel('yr')  # Add an x-label to the axes.
ax.set_ylabel('km3/s')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()




fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gws_aggregated[idlist == n][0]/1e9, label='groundwater storage')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

gws_variation[1][idlist == n]/1e9



fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gwh_aggregated[idlist == n][0], label='groundwater head')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('m')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

gwh_variation[1][idlist == n]

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(sm_aggregated[idlist == n][0]/1e9, label='soil moisture')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

sm_variation[1][idlist == n]/1e9

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(ET_aggregated[idlist == n][0]/1e9, label='evapotranspiration')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('m3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

ET_variation[1][idlist == n]/1e9


#congo basin 18821

n = 18821
m = "Congo basin"

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(net_consumption[idlist == n][0]/1e9, label='consumption')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

consumption_variation[1][idlist == n]/1e9

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(q_aggregated[idlist == n][0]/1e9, label='discharge')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3/s')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

q_variation[1][idlist == n]/1e9


#inconsistent results
q = q_aggregated[idlist == n][0]
np.sum(np.diff(q))/1e9
plt.plot(np.diff(q)/1e9)


q1 = q_yr[idlist == n][0]
np.sum(np.diff(q1))/1e9

var1 = np.cumsum(np.diff(q))
plt.plot(var1/1e9)

plt.plot(np.diff(q)/1e9)

np.trapz(np.gradient(q))
plt.plot(np.gradient(q))


var2 = np.cumsum(np.diff(q1))
plt.plot(var2/1e9)

plt.plot(np.diff(q1)/1e9)
np.trapz(np.gradient(q1))
plt.plot(np.gradient(q1)/1e9)


#qgis check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gws_aggregated[idlist == n][0]/1e9, label='groundwater storage')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

gws_variation[1][idlist == n]/1e9

#qgos check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gwh_aggregated[idlist == n][0], label='groundwater head')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('m')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

gwh_variation[1][idlist == n]

#qgis check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(sm_aggregated[idlist == n][0]/1e9, label='soil moisture')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

sm_variation[1][idlist == n]/1e9

#Qgis ceck ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(ET_aggregated[idlist == n][0]/1e9, label='evapotranspiration')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

ET_variation[1][idlist == n]/1e9

#Qgis check ok


#danube basin



#Mississipi 12735

n = 12735
m = "Mississipi basin"

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(net_consumption[idlist == n][0]/1e9, label='consumption')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(np.gradient(net_consumption[idlist == n][0])/1e9, label='consumption gradient')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3/yr')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()


#test values is wrong
np.mean(np.gradient(net_consumption[idlist == n][0])/1e9)
np.mean(np.gradient(net_consumption[idlist == n][0][-10:]))/1e9
np.mean(np.gradient(net_consumption[idlist == n][0][-10:]))/1e9/np.mean(np.gradient(net_consumption[idlist == n][0])/1e9)
consumption_gradient[1][idlist ==n]/1e9#gradient recent

consumption_variation[1][idlist == n]/1e9




fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(q_aggregated[idlist == n][0], label='discharge')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('m3/s')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

q_variation[1][idlist == n]

#qgis check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gws_aggregated[idlist == n][0]/1e9, label='groundwater storage')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

gws_variation[1][idlist == n]/1e9

#qgos check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gwh_aggregated[idlist == n][0], label='groundwater head')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('m')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

gwh_variation[1][idlist == n]

#qgis check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(sm_aggregated[idlist == n][0]/1e9, label='soil moisture')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

sm_variation[1][idlist == n]/1e9

#Qgis ceck ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(ET_aggregated[idlist == n][0]/1e9, label='evapotranspiration')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

ET_variation[1][idlist == n]/1e9


#Chinese big basin

n = 12182
m = "Yellow river basin"

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(net_consumption[idlist == n][0]/1e9, label='consumption')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

consumption_variation[1][idlist == n]/1e9

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(q_aggregated[idlist == n][0]/1e9, label='discharge')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3/s')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

q_variation[1][idlist == n]/1e9

#qgis check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gws_aggregated[idlist == n][0]/1e9, label='groundwater storage')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title( m + " 1960/2010")  # Add a title to the axes.
ax.legend()

gws_variation[1][idlist == n]/1e9

#qgos check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(gwh_aggregated[idlist == n][0], label='groundwater head')  # Plot some data on the axes.
ax.set_xlabel('month')  # Add an x-label to the axes.
ax.set_ylabel('m')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

gwh_variation[1][idlist == n]

#qgis check ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(sm_aggregated[idlist == n][0]/1e9, label='soil moisture')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

sm_variation[1][idlist == n]/1e9

#Qgis ceck ok

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(ET_aggregated[idlist == n][0]/1e9, label='evapotranspiration')  # Plot some data on the axes.
ax.set_xlabel('year')  # Add an x-label to the axes.
ax.set_ylabel('km3')  # Add a y-label to the axes.
ax.set_title(m +" 1960/2010")  # Add a title to the axes.
ax.legend()

ET_variation[1][idlist == n]/1e9
