# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:09:37 2021
only inputs with all rata and intermediary results
@author: easpi
"""

# Please set the pcrglobwb output directory (outputDir) in an absolute path.
# - Please make sure that you have access to it. 
outputDir = 'C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/output'

#outputDir = 'C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/output_catchments_in_ecoregions'
# Set the input directory map in an absolute path. The input forcing and parameter directories and files will be relative to this.
# - The following is an example using files from the opendap server.
#inputDir    = 'C:/Users/easpi/Documents/PhD Water Footprint/Papers/2 FF modelling with GHM/calculations/data'
inputDir = 'D:/Fate/data'

#~ # - The following is an example using input files stored locally in your computing machine.



#important propeties----------------------------------------------------------
#surface of gridcell
fn_area_map = 'Global_CellArea_m2_05min.map'

#dem file
fn_dem = 'DEM_05min.map'

#aquifer yield
fn_aquifer_yield = 'Sy_globe.map'




# The spatial unit delineation of interest------------------------------------
# if the landmask is not in pcraster format (e.g. .tiff or numpy), please adapt the  code in subsequent modules.
fn_catchments    = 'catchments.map'
fn_catchments6 = 'sub_watersheds_level6.map'
fn_catchmentsinecoregions = 'Catchments_Extract_FEOW3.nc'#this is netcdf not .map
fn_catchmentsldd = 'Catchment_IDD.nc'
fn_catchmentsValerio_old = 'basins_5min_valerio.tif'
fn_catchmentsValerio = 'basin_5min_pcrglob_adjusted_resized.tif'

# var_name_catchmentsinecoregions  = 'Catchments_Extract_FEOW.tif'

#set name of the scale
name_scale = 'basinV'#this extension will be added to the name of output files

#import filter instead of calculating
fn_filter_catchmentsinecoregions = 'filter_array_catchmentsinecoregions_scale.npy'
fn_filter_catchmentsldd = 'filter_array_catchments_ldd_rev1.npy'
fn_filter_catchments_valerio_old = "filter_array_catchments_valerio.npy"
fn_filter_catchments_valerio = "filter_array_catchments_valerio_adjusted.npy"

#stressor human run-----------------------------------------------------------

name_timeperiod = '1960to2010'

#groundwater head netcdf
fn_ground_water_depth ='groundwaterDepthLayer1_monthEnd_1960to2010.nc'
var_groundwater_depth = 'groundwater_depth_for_layer_1'

fn_ground_water_depth_natural = 'D:/fate/data/groundwaterDepthLayer1_yearavg_natural_output_19602010.nc'
var_groundwater_depth_natural = 'groundwater_depth_for_layer_1'

#evapotranspiration netcdr
fn_ET ='totalEvaporation_annuaTot_output_1960to2010.nc'
var_ET = 'total_evaporation'

fn_ET_natural = 'totalEvaporation_monthTot_output_1960to2004.nc'

#soil moisture netcdf
fn_soil_low ='storLowTotal_annualAvg_out_1960to2010_b_human.nc'#0-50cm?
var_soil_low = 'lower_soil_storage'
fn_soil_upp ='storUppTotal_annualAvg_out_1960to2010_b_human.nc'#50-150cm?
var_soil_upp = 'upper_soil_storage'
fn_soil_low_nat ='storLowTotal_annualAvg_out_1960to2010_natural.nc'#0-50cm?
var_soil_low = 'lower_soil_storage'
fn_soil_upp_nat ='storUppTotal_annualAvg_out_1960to2010_natural.nc'#50-150cm?
var_soil_upp = 'upper_soil_storage'

#discharge netcdf
fn_discharge = 'discharge_monthAvg_1960to2010_human.nc'
fn_discharge_natural = 'discharge_monthAvg_output_1960_2010_natural.nc'




#consumption caluclation------------------------------------------------------

#water astraction
fn_water_abstraction = 'M:/Documents/data/totalAbstractions_annuaTot_1960to2010.nc'

#return flow caluculation data
fn_recharge = 'M:/Documents/data/gwRecharge_monthTot_output_1960to2010_human.nc'
fn_recharge_natural  ='M:/Documents/data/gwRecharge_monthTot_output_1960to2010_natural.nc'
fn_withdrawal_domestic = 'M:/Documents/data/domesticWithdrawal_annualTot_out_1960to2010_b.nc'
fn_withdrawal_industry = 'M:/Documents/data/industry_Withdrawal_annualTot_out_1960to2010_b.nc'

#return flows
# Water consumptions are calculated as withdrawal – return flow. Thus:
# Non-irrigation returnflow = (industrial withdrawal + domestic withdrawal) – non-irrigation water consumption
# Irrigation returnflow is the difference between groundwater recharge under natural conditions and under human impacted conditions.
# I did not report nonpaddy irrigation abstraction but you can calculate those by total abstraction – non-irr water withdrawal- paddy irrigation water withdrawal. 



#results saved

