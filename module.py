# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:09:30 2021

@author: easpi
"""

#import os
from netCDF4 import Dataset
from pcraster import readmap, pcr2numpy
import numpy as np
from numpy import isnan, ma, trapz
from matplotlib import pyplot as plt
from scipy import stats
import tifffile
import Input


def weight(pointer_array, idlist, spatial_unit):
    area = pcr2numpy(readmap(Input.inputDir +'/'+ Input.fn_area_map), mv = 1e20)
    #a = area[spatial_unit.mask == 0]
    n_spatial_unit = pointer_array.shape[0]
    out = np.full((n_spatial_unit), 1e20)
    for i in range(n_spatial_unit):#select catchment in the same order as the pointer array sot hat the idlist is still valid
        out[i] = np.sum(area[pointer_array[i][0], pointer_array[i][1]])
    out = ma.masked_where(isnan(out), out)

    return out

def exclude(s, mask):
    out = np.full(s.shape, 1e20)
    for  i in range(s.shape[0]):
        if mask[i]==0:
            out[i,:] = s[i,:]
    out = ma.masked_values(out, 1e20)
    return out

def index_filter(spatial_unit):
        """
        Parameters
        ----------
        spatial_unit : TYPE array (lat,lon)
            DESCRIPTION. array containing the ID of the catchments starting with ID = 0
    
        Returns
        -------
        index_filter : TYPE tuple ([spatial unit id, (lat list,lon list)], [id list])
        idlsit : TYPE array where the ID of the catchment is stored in the same position  as index_filter row number
            DESCRIPTION. the list contains for each catchmetn ID, the list of latitude index in position 0 and the list of longitude indexes for position 1
    
        """
        index_filter = []
    # select ID of spatial units present in the map
        idlist = np.unique(spatial_unit)
        for n in idlist:
            a = np.where(spatial_unit == n)
            index_filter.append(a)
        pointer_array = np.array(index_filter[0], dtype=list)
        return pointer_array, idlist
        
# def set_spatial_unit(filename, var_name):

#     dataset1 = Dataset(filename)
#     print(dataset1)
#     basin = spatial_unit(dataset1.variables[var_name][:])
#     pointer_array = basin.index_filter()
    
#     return basin,pointer_array

 
def convert_month_to_year_avg(s):
    """
    Parameters
    ----------
    s : TYPE scalar array of shape (spatial unit id, time)
        DESCRIPTION.it represents the scalar variable timseries at monthly timestep

    Returns
    -------
    out : TYPE scalar array (spatial unit id, time)
        DESCRIPTION. converted array to yearly timestep (average 12 months)

    """

    out = np.zeros( (s.shape[0], s.shape[1]//12) )
    for i in range(s.shape[0]):#for all catchments
        for j in range(out.shape[1]):#for all year timesteps
            out[i,j] = np.mean(s[i, j*12:j*12+12])
    #out_mask = ma.masked_where(isnan(s), out)
    out_mask = ma.masked_where(isnan(out) == 1, out)
    return out_mask
                   
def convert_month_to_year_sum(s):
    """
    Parameters
    ----------
    s : TYPE scalar array of shape (spatial unit id, time)
        DESCRIPTION.it represents the scalar variable timseries at monthly timestep

    Returns
    -------
    out : TYPE scalar array (spatial unit id, time)
        DESCRIPTION. converted array to yearly timestep (average 12 months)

    """

    out = np.zeros( (s.shape[0], s.shape[1]//12) )
    for i in range(s.shape[0]):#for all catchments
        for j in range(out.shape[1]):#for all year timesteps
            out[i,j] = np.sum(s[i, j*12:j*12+12])
    #out_mask = ma.masked_where(isnan(s), out)
    out_mask = ma.masked_where(isnan(out) == 1, out)
    return out_mask

def moving_average(a, n):
    #https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    cumsum = np.cumsum(a, axis = 1)
    out = np.full((a.shape[0], a.shape[1]-n+1),1e20)
    for i in range(a.shape[0]):
        ret = cumsum[i,:]
        ret[n:] =ret[n:]-ret[:-n]
        out[i,:] = ret[n-1:]/n
    out = ma.masked_values(out, 1e20)
    return out


def make_map(stressor_aggregated, index_filter, idlist, spatial_unit):#correct that taking into acount the correct location of the catchment 
    """array inputs! spatial unit with mask"""
    """indexfilter has to be calculated based on spatial_unit beforehand"""
    l1=idlist
    # l1 = l1[np.where(isnan(l1) == 0)]
    # l = ma.getdata(l1).tolist()
    map_temp = np.full(spatial_unit.shape, 1e20)
       
   #  for i in l1:#for all catchment id 
   #      map_temp[index_filter[l1==i][0][0], index_filter[l1==i][0][1]] = stressor_aggregated[l1==i]
   #      #fill the map with the value associated with each catchment.
    for i in l1:#for all catchment id 
        a = np.where(l1 == i)[0][0]#verify position of the catchment in the id list
        if stressor_aggregated.mask[a]==0:
            map_temp[index_filter[a][0], index_filter[a][1]] = stressor_aggregated[a]
        #fill the map with the value associated with each catchment.
   
    #map_out = ma.masked_where(isnan(map_temp) == 1, map_temp)
    #map_out = ma.masked_values(map_temp, 1e20)
    map_out = ma.masked_where(isnan(spatial_unit) == 1, map_temp)
    map_out = ma.masked_values(map_out, 1e20)
    return map_out

    
def new_map_netcdf(filename, map_out, name_scalar, unit_scalar, lat, lon):
    """
    Parameters
    ----------
   
    Returns
    -------
    None. Netcdf file is written based on the inputs. and the dataset is 
    returned to main program.

    """
    dataset_out = Dataset(filename +'.nc','w',format = 'NETCDF4')
    dataset_out.createDimension("latitude",lat.shape[0])
    dataset_out.createDimension("longitude", lon.shape[0])
    

    latitude = dataset_out.createVariable("latitude","f8",("latitude",), zlib = True)
    longitude = dataset_out.createVariable("longitude","f8",("longitude",), zlib = True)
    scalar = dataset_out.createVariable(name_scalar,"f8",("latitude","longitude"), zlib = True)
    scalar.units = unit_scalar

    #fill NETCDF with results
    latitude[:] = lat[:]
    longitude[:] = lon[:]
    scalar[:] = map_out[:]

    dataset_out.sync()#write into the  saved file
    print(dataset_out)
    dataset_out.close()
    return "netcdf created, check folder"

def new_stressor_out_netcdf(filename, stressor_out, ID, t, name_stressor, unit_time,unit_stressor):
    """
    Parameters
    ----------
   
    Returns
    -------
    None. Netcdf file is written based on the inputs. and the dataset is 
    returned to main program.

    """
    dataset_out = Dataset(filename +'.nc','w',format = 'NETCDF4')
    dataset_out.createDimension("spatial_unit_id",ID.shape[0])
    dataset_out.createDimension("time", t.shape[0])
    

    time = dataset_out.createVariable("time","f4",("time",), zlib = True)
    time.units = unit_time
    spatial_unit_id = dataset_out.createVariable("spatial_unit_id","f4",("spatial_unit_id",), zlib = True)
    stressor_aggregated_timeserie = dataset_out.createVariable(name_stressor,"f4",("spatial_unit_id","time"), zlib = True)
    stressor_aggregated_timeserie.units = unit_stressor

    #fill NETCDF with results
    time[:] = t[:]
    spatial_unit_id[:] = ID[:]
    stressor_aggregated_timeserie [:] = stressor_out[:]

    dataset_out.sync()#write into the  saved file
    print(dataset_out)
    dataset_out.close()
    return "netcdf created, check folder"

# def DFM(consumption, stressor):
    
#     c_sum = np.sum(consumption, axis = 1)
#     threshold = stats.scoreatpercentile(c_sum[c_sum>0], per=25)
#     c_sum = ma.masked_less_equal(c_sum, threshold)#percentil 10 of integral where consupmtio nintegral is >0
    
#     c_int = np.cumsum(consumption, axis = 1)
#     s_int = np.cumsum(stressor, axis = 1)
#     DF = s_int/c_int
    
#     DFM = np.full((stressor.shape), 1e20)
    
#     for i in range(DFM.shape[0]):
#         if c_sum.mask[i] == 0:
#             DFM = (stressor - DF*consumption)/c_int
     
#     DFM = ma.masked_values(DFM, 1e20)
   
#     return DFM

         
def FF_lm(consumption, stressor, pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude, idlist):
    '''
    

    Parameters
    ----------
    consumption : TYPE
        DESCRIPTION. mask consumption where consumption = 0
    stressor : TYPE
        DESCRIPTION. stressor = human - natural or human - natural taking into accountthe initial point

    Returns
    -------
    None.

    '''
    c_int = np.sum(consumption, axis = 1)
    threshold = stats.scoreatpercentile(c_int[c_int>0], per=25)
    integral1temp = ma.masked_less_equal(c_int, threshold)#percentil 10 of integral where consupmtio nintegral is >0

#def FF_lm(consumption, stressor,pointer_array, spatial_unit, stressor_name):

    FF = np.full((stressor.shape[0],5), 1e20)

    for i in range(stressor.shape[0]-1):
        if integral1temp.mask[i] == 0:
        #if np.all(consumption[i,:] != 0):#fill values  will remain where consumption is null
            if np.all(isnan(stressor[i,:]) == 0):#fill values remain if there is no stressor value
                slope, intercept, r, p, se = stats.linregress(consumption[i,:], stressor[i,:])
                FF[i,0] = slope
                FF[i,1] = se
                FF[i,2] = r**2
                FF[i,3] = p
                if slope != 0:
                    FF[i,4] = np.abs(se/slope)
    
    FF = ma.masked_where(FF == 1e20, FF)
    
    error = np.sqrt(np.square(FF[:,1]/FF[:,0]))
    
    map_errorff = make_map(error, pointer_array, idlist, spatial_unit)
    new_map_netcdf(Input.outputDir +'/'+ 'ff_slope_error_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_errorff, stressor_name, stressor_unit, latitude, longitude)
    
    map_ff = make_map(FF[:,0], pointer_array, idlist, spatial_unit)
    new_map_netcdf(Input.outputDir +'/'+ 'ff_slope_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
 

    r2 = FF[:,2]
    r2 = r2[r2.mask == 0]
    
    p = FF[:,3]
    p = p[p.mask == 0]
    
    print("amount of catchments with R2>0,5:", np.where(r2>0.5)[0].shape, "out of ", r2.shape, "catchments with valid data")
       
    print("amount of catchments with p<0,05:", np.where(p<0.05)[0].shape, "out of ", p.shape, "catchments with valid data")
   
    temp= FF[FF[:,3]<0.05]
    coeff = temp[:,0]
    coeff = coeff[coeff.mask == 0]
    #plt.boxplot(coeff)
    print("amount of catchments with negative slope:", np.where(coeff<0)[0].shape, "out of ", coeff.shape, "catchments with p<0.05")

    temp= FF[FF[:,3]<0.05]
    coeff = temp[:,2]
    coeff = coeff[coeff.mask == 0]
    #plt.boxplot(coeff)
    print("amount of catchments with r2>0.5:", np.where(coeff>0.5)[0].shape, "out of ", coeff.shape, "catchments with p<0.05")


    coeff = FF[:,0]
    coeff = coeff[coeff.mask == 0]
    #plt.boxplot(coeff)
    print("amount of catchments with negative slope:", np.where(coeff<0)[0].shape, "out of ", coeff.shape, "catchments with validdata")


    #map_ff = make_map(FF[:,0], pointer_array, spatial_unit)
    
       #plt.matshow(map_ff, vmin = stats.scoreatpercentile(FF[:,0],5), vmax = stats.scoreatpercentile(FF[:,0],95))#ok
     
    #new_map_netcdf(Input.outputDir +'/'+ stressor_name + 'normalized' + '_' + Input.name_timeperiod + '_' + Input.name_scale, map_change, stressor_name + 'normalized', stressor_unit, latitude, longitude)
    
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_regression_' + stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    
    return FF, map_ff, map_errorff, error


def calculate_dif(a, a_ref):
    temp = a.copy()
    for i in range(a.shape[0]):#catchments
        for j  in range(a.shape[1]):#time
            temp[i,j] = a[i,j] - a_ref[i]
    return temp

def derivate_time(stressor, n):
    temp = stressor
    for i in range(n-1):
        temp1 = np.gradient(temp, axis = 1)
        temp = temp1.copy()
    return temp

def FF_sum(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude,  idlist):

    integral = np.sum(stressor, axis = 1)#careful with the time conversion: stressor. time 
    integral1 = np.sum(consumption, axis = 1)#careful with the time conversion: consumption volume
    #integral1 = ma.masked_inside(integral1, -1, 1825)#100 L/day per human -> at least 1 human in the catchment
    #integral1 = ma.masked_values(integral1, 0)
    #integral1 = ma.masked_inside(integral1, -1, 1)#100 L/day per human -> at least 5 human in the catchment
    threshold = stats.scoreatpercentile(integral1[integral1>0], per=25)
    integral1temp = ma.masked_less_equal(integral1, threshold)#percentil 10 of integral where consupmtio nintegral is >0
    integraltemp = ma.masked_where(integral1.mask == 1, integral)
    ff = integraltemp/integral1temp
    ff = ma.masked_where(isnan(ff) == 1, ff)
    
    map_ff = make_map(ff, pointer_array, idlist, spatial_unit)
    plt.matshow(map_ff, vmin = stats.scoreatpercentile(ff,5), vmax = stats.scoreatpercentile(ff,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+'ff_sum_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
    
   
    plt.imshow(map_ff)
    
       
    return ff, map_ff

def FF_mean(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude,  idlist):#for gw head

    integral = np.mean(stressor, axis = 1)#careful with the time conversion: stressor. time 
    integral1 = np.sum(consumption, axis = 1)#careful with the time conversion: consumption volume
    #integral1 = ma.masked_inside(integral1, -1, 1825)#100 L/day per human -> at least 1 human in the catchment
    #integral1 = ma.masked_values(integral1, 0)
    #integral1 = ma.masked_inside(integral1, -1, 1)#100 L/day per human -> at least 5 human in the catchment
    threshold = stats.scoreatpercentile(integral1[integral1>0], per=25)
    integral1temp = ma.masked_less_equal(integral1, threshold)#percentil 10 of integral where consupmtio nintegral is >0
    integraltemp = ma.masked_where(integral1.mask == 1, integral)
    
    ff = integraltemp/integral1temp
    ff = ma.masked_where(isnan(ff) == 1, ff)
    
    map_ff = make_map(ff, pointer_array, idlist, spatial_unit)
    plt.matshow(map_ff, vmin = stats.scoreatpercentile(ff,5), vmax = stats.scoreatpercentile(ff,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+'ff_mean_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
    
   
    plt.imshow(map_ff)
    
       
    return ff, map_ff



# def FF_integral(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude,  idlist):
#     '''
   
#         DESCRIPTION.

#     '''
#     integral = np.trapz(stressor, axis = 1)#careful with the time conversion: stressor. time 
#     integral1 = np.trapz(consumption, axis = 1)#careful with the time conversion: consumption volume
#     integral1 = ma.masked_inside(integral1, -1, 1)
#     #integral1 = ma.masked_values(integral1, 0)
#     ff = integral/integral1
#     ff = ma.masked_where(isnan(ff) == 1, ff)
    
#     map_ff = make_map(ff, pointer_array, idlist, spatial_unit)
#     plt.matshow(map_ff, vmin = stats.scoreatpercentile(ff,5), vmax = stats.scoreatpercentile(ff,95))
#     #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
#     new_map_netcdf(Input.outputDir +'/'+'ff_average_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
    
#     d = derivate_time(stressor, 3)
#     error = np.abs(stressor.shape[1]*np.max(d, axis = 1))
#     d1 = derivate_time(consumption, 3)
#     error1 = np.abs(consumption.shape[1]*np.max(d1, axis = 1))
    
    
#     temp = np.square(error/integral) + np.square(error1/integral1)
#     errorff = np.sqrt(temp)
    
#     #ff_low = ff*(1-errorff)
#     # ff_low = (integral - error)/(integral1 + error1)
#     #map_ff_low = make_map(ff_low, pointer_array, spatial_unit)
#     #new_map_netcdf(Input.outputDir +'/'+'ff_integral_low_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
#     #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_low'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff_low)
#     #new_map_netcdf(Input.outputDir +'/'+'ff_average_low_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff_low, stressor_name, stressor_unit, latitude, longitude)
    
    
#     #ff_high = ff*(1+errorff)
#     # ff_high = (integral + error)/(integral1 - error1)
#     #map_ff_high = make_map(ff_high, pointer_array, spatial_unit)
#     #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_high'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff_high)
#     #new_map_netcdf(Input.outputDir +'/'+'ff_average_high_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff_high, stressor_name, stressor_unit, latitude, longitude)
    
#     plt.imshow(map_ff)
    
#     map_errorff = make_map(errorff, pointer_array, idlist, spatial_unit)
#     new_map_netcdf(Input.outputDir +'/'+ 'ff_average_error_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_errorff, stressor_name, stressor_unit, latitude, longitude)
    
    
    
#     return ff, map_ff, errorff, map_errorff




def FF_grad(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude, idlist):
    
    
#      c_sum = np.sum(consumption, axis = 1)
#      threshold = stats.scoreatpercentile(c_sum[c_sum>0], per=25)
#      c_sum = ma.masked_less_equal(c_sum, threshold)#percentil 10 of integral where consupmtio nintegral is >0
# # 
#     DFM = np.full((stressor.shape), 1e20)
    
#     for i in range(DFM.shape[0]):
    
    grad = np.gradient(stressor, axis  = 1)
    grad1 = np.gradient(consumption, axis = 1)
    #grad1 = ma.masked_values(grad1, 0)#mask catchments with consumption gradient 0 m3/yr
    #grad1 = ma.masked_values(grad1, 0)
    ff = grad[:,-1]/grad1[:,-1]
    ff = ma.masked_where(isnan(ff) == 1, ff)
    map_ff = make_map(ff, pointer_array, idlist, spatial_unit)
    plt.matshow(map_ff, vmin = stats.scoreatpercentile(ff,5), vmax = stats.scoreatpercentile(ff,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_grad_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)

    # error = np.std(np.gradient(stressor, axis  = 1), axis = 1)
    # error1 =np.std(np.gradient(consumption, axis  = 1),axis = 1)
    
    # temp = np.square(error/grad) + np.square(error1/grad1)
    # errorff = np.sqrt(temp)
    
    #ff_low = ff*(1-errorff)
    
    # ff_low = (grad - error)/(grad1 + error1)
    #map_ff_low = make_map(ff_low, pointer_array, spatial_unit)
    #new_map_netcdf(Input.outputDir +'/'+'ff_integral_low_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
    # #tifffile.imsave(Input.outputDir +'/'+ 'ff_grad_low'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff_low)
    #new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_low_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff_low, stressor_name, stressor_unit, latitude, longitude)
    
    
    #ff_high = ff*(1+errorff)
    # ff_high = (grad + error)/(grad1 - error1)
    #map_ff_high = make_map(ff_high, pointer_array, spatial_unit)
    # #tifffile.imsave(Input.outputDir +'/'+ 'ff_grad_high'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff_high)
    #new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_high_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff_high, stressor_name, stressor_unit, latitude, longitude)
    
    # map_errorff = make_map(errorff, pointer_array, idlist, spatial_unit)
    # new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_error_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_errorff, stressor_name, stressor_unit, latitude, longitude)
    
    
    # # plt.imshow(map_ff)
    # # # plt.imshow(map_ff_low)
    # # plt.imshow(map_ff_high)    
 
    return ff, map_ff


def FF_mean_mean(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude,  idlist):
    
    #integral1 = ma.masked_inside(integral1, -1, 1825)#100 L/day per human -> at least 1 human in the catchment
    #integral1 = ma.masked_values(integral1, 0)
    #integral1 = ma.masked_inside(integral1, -1, 1)#100 L/day per human -> at least 5 human in the catchment
    integral1 = np.sum(consumption, axis =1)
    threshold = stats.scoreatpercentile(integral1[integral1>0], per=25)
    integral1temp = ma.masked_less_equal(integral1, threshold)#percentil 10 of integral where consupmtio nintegral is >0
    mask = integral1temp.mask
    
    s = exclude(stressor, mask)
    c = exclude(consumption, mask)
    
    mean = np.mean(s, axis = 1)#careful with the time conversion: stressor. time 
    mean1 = np.mean(c, axis = 1)#careful with the time conversion: consumption volume

    
    ff = mean/mean1
    ff = ma.masked_where(isnan(ff) == 1, ff)
    
    map_ff = make_map(ff, pointer_array, idlist, spatial_unit)
    plt.matshow(map_ff, vmin = stats.scoreatpercentile(ff,5), vmax = stats.scoreatpercentile(ff,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+'ff_marginal_mean'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
    
    
    
    plt.imshow(map_ff)
    
       
    return ff, map_ff


def error(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude,  idlist):
    integral1 = np.sum(consumption, axis =1)
    threshold = stats.scoreatpercentile(integral1[integral1>0], per=25)
    integral1temp = ma.masked_less_equal(integral1, threshold)#percentil 10 of integral where consupmtio nintegral is >0
    mask = integral1temp.mask
    
    s = exclude(stressor, mask)
    c = exclude(consumption, mask)
    m= s/c
    sd = np.std(m, axis = 1)
    mean = np.mean(m, axis = 1)

    err = np.absolute(sd/mean)

    map_err = make_map(err, pointer_array, idlist, spatial_unit)
    plt.matshow(map_err, vmin = stats.scoreatpercentile(err,5), vmax = stats.scoreatpercentile(err,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+'error'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_err, stressor_name, stressor_unit, latitude, longitude)

    return err, map_err

def error_independent(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude,  idlist):
    integral1 = np.sum(consumption, axis =1)
    threshold = stats.scoreatpercentile(integral1[integral1>0], per=25)
    integral1temp = ma.masked_less_equal(integral1, threshold)#percentil 10 of integral where consupmtio nintegral is >0
    mask = integral1temp.mask
    
    s = exclude(stressor, mask)
    c = exclude(consumption, mask)

    sd = np.std(s, axis = 1)
    mean = np.mean(s, axis = 1)

    sd1 = np.std(c, axis = 1)
    mean1 = np.mean(c, axis = 1)

    err = np.sqrt(np.square(sd/mean)+np.square(sd1/mean1))

    map_err = make_map(err, pointer_array, idlist, spatial_unit)
    plt.matshow(map_err, vmin = stats.scoreatpercentile(err,5), vmax = stats.scoreatpercentile(err,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_integral_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+'error'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_err, stressor_name, stressor_unit, latitude, longitude)

    return err, map_err
    


def FF_grad_old(consumption, stressor,pointer_array, spatial_unit, stressor_name, stressor_unit, latitude, longitude, idlist):
    
    
#      c_sum = np.sum(consumption, axis = 1)
#      threshold = stats.scoreatpercentile(c_sum[c_sum>0], per=25)
#      c_sum = ma.masked_less_equal(c_sum, threshold)#percentil 10 of integral where consupmtio nintegral is >0
# # 
#     DFM = np.full((stressor.shape), 1e20)
    
#     for i in range(DFM.shape[0]):
    
    grad = np.mean(np.gradient(stressor, axis  = 1), axis=1)
    grad1 = np.mean(np.gradient(consumption, axis = 1), axis = 1)
    #grad1 = ma.masked_values(grad1, 0)#mask catchments with consumption gradient 0 m3/yr
    #grad1 = ma.masked_values(grad1, 0)
    grad1 = ma.masked_inside(grad1, -1, 1)
    ff = grad/grad1
    ff = ma.masked_where(isnan(ff) == 1, ff)
    map_ff = make_map(ff, pointer_array, idlist, spatial_unit)
    plt.matshow(map_ff, vmin = stats.scoreatpercentile(ff,5), vmax = stats.scoreatpercentile(ff,95))
    #tifffile.imsave(Input.outputDir +'/'+ 'ff_grad_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff)
    new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)

    error = np.std(np.gradient(stressor, axis  = 1), axis = 1)
    error1 =np.std(np.gradient(consumption, axis  = 1),axis = 1)
    
    temp = np.square(error/grad) + np.square(error1/grad1)
    errorff = np.sqrt(temp)
    
    #ff_low = ff*(1-errorff)
    
    # ff_low = (grad - error)/(grad1 + error1)
    #map_ff_low = make_map(ff_low, pointer_array, spatial_unit)
    #new_map_netcdf(Input.outputDir +'/'+'ff_integral_low_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff, stressor_name, stressor_unit, latitude, longitude)
    # #tifffile.imsave(Input.outputDir +'/'+ 'ff_grad_low'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff_low)
    #new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_low_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff_low, stressor_name, stressor_unit, latitude, longitude)
    
    
    #ff_high = ff*(1+errorff)
    # ff_high = (grad + error)/(grad1 - error1)
    #map_ff_high = make_map(ff_high, pointer_array, spatial_unit)
    # #tifffile.imsave(Input.outputDir +'/'+ 'ff_grad_high'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale + '.tiff', map_ff_high)
    #new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_high_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_ff_high, stressor_name, stressor_unit, latitude, longitude)
    
    map_errorff = make_map(errorff, pointer_array, idlist, spatial_unit)
    new_map_netcdf(Input.outputDir +'/'+ 'ff_marginal_error_'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_errorff, stressor_name, stressor_unit, latitude, longitude)
    
    
    # plt.imshow(map_ff)
    # # plt.imshow(map_ff_low)
    # # plt.imshow(map_ff_high)    
 
    return ff, map_ff, errorff, map_errorff






# def calculate_groundwater_head(s, dem, t):
#     '''
    
#     Parameters
#     ----------
#     s : TYPE netcdf variable  or array (time, lat, lon)
#         DESCRIPTION. groundwater depth 
#     dem : TYPE array (lat, lon)
#         DESCRIPTION. digital elevation model
#     t : TYPE integer
#         DESCRIPTION. timestep index in s

#     Returns
#     -------
#     out : TYPE array(lat,lon)
#         DESCRIPTION. groundwater head at the given timestep t

#     '''
#     s_t = s[t,:,:]
#     out = dem - s_t
#     return out



# class stressor_maps(stressor, spatial_unit):
 
#     def __init__(self, stressor_aggregated, index_filter):
        
#         self.variation = np.sum(np.diff(stressor_aggregated, axis = 1), axis = 1)
#         self.gradient = np.gradient(stressor_aggregated, axis = 1)
#         self.integral = np.trapz(self.gradient, axis = 1)
        
#     def make_map_variation(self, index_filter, spatial_unit):
              
#         l = ma.getdata(spatial_unit.idlist).tolist()
#         a = spatial_unit.map
#         map_temp = np.full(a.shape, 65535)
            
#         for i in spatial_unit.idlist:#for all catchment id 
#             map_temp[index_filter[l.index(i),0], index_filter[l.index(i),1]] = self.variation[l.index(i)]
                
#         map_out = ma.masked_where(spatial_unit.mask == 1, map_temp)
        
#         return map_out
    
#     def make_map_gradient(self, index_filter, spatial_unit):
              
#         l = ma.getdata(spatial_unit.idlist).tolist()
#         map_temp = np.full(spatial_unit.map.shape, 65535)
            
#         for i in spatial_unit.idlist:#for all catchment id 
#             map_temp[index_filter[l.index(i),0], index_filter[l.index(i),1]] = self.gradient[l.index(i)]
                
#         map_out = ma.masked_where(spatial_unit.mask == 1, map_temp)
        
#         return map_out  

# def aggregation_mean(stressor, index_filter):
#         """
#     Parameters
#     ----------
#     stressor : TYPE array (time, lat, lon)
#         DESCRIPTION. stressor we wont to aggregate spatially
#     index_filter : TYPE array(idlist, lat array, lon array)
#         DESCRIPTION. filter array to select the spatial units

#     Returns
#     -------
#     out : TYPE array(idlist, time)
#         DESCRIPTION.the stressor is aggrgated by MEAN over each spatial unit, generating timeseries

#     """
#         out = np.zeros((index_filter.shape[0], stressor.shape[0]))
#         for t in range(stressor.shape[0]):
#             s = stressor.value[t,:,:]
#             for k in range(index_filter.shape[0]):
#                 out[k,t] = np.mean(s[index_filter[k,0],  index_filter[k,1]])
#         return out
                    
# def aggregation_sum(stressor, index_filter):
#     """
#     Parameters
#     ----------
#     stressor : TYPE array (time, lat, lon)
#         DESCRIPTION. stressor we wont to aggregate spatially
#     index_filter : TYPE array(idlist, lat array, lon array)
#         DESCRIPTION. filter array to select the spatial units

#     Returns
#     -------
#     out : TYPE array(idlist, time)
#         DESCRIPTION.the stressor is aggrgated by sum over each spatial unit, generating timeseries

#     """
#     out = np.zeros((index_filter.shape[0], stressor.time.shape[0]))
#     for t in range(stressor.time.shape[0]):
#         s = stressor.value[t,:,:]
#         for k in range(index_filter.shape[0]):
#             out[k,t] = np.sum(s[index_filter[k,0],  index_filter[k,1]])
#     return out

# def make_map_gradient(stressor_aggregated, index_filter, spatial_unit):
#     """array inputs!"""
#     l1 = np.unique(spatial_unit)
#     l = ma.getdata(l1).tolist()
#     map_temp = np.full(spatial_unit.shape, 65535)
    
#     stressor_aggregated_yr = convert_month_to_year_avg(stressor_aggregated)
#     gradient = np.mean(np.gradient(stressor_aggregated_yr, axis = 1)[:,-10:], axis = 1)

        
#     for i in spatial_unit.idlist:#for all catchment id 
#         map_temp[index_filter[l.index(i),0], index_filter[l.index(i),1]] = gradient[l.index(i)]
                
#     map_out = ma.masked_where(spatial_unit.mask == 1, map_temp)
        
#     return map_out


# def calculate_variation(aggregated_stressor, spatial_unit, pointer_array, latitude, longitude, stressor_name, stressor_unit):
#     '''
    

#     Parameters
#     ----------
#     aggregated_stressor : TYPE ndarray
#         DESCRIPTION. ANNUAL VALUES
#     spatial_unit : TYPE ndarray masked
#         DESCRIPTION.
#     pointer_array : TYPE ndarray
#         DESCRIPTION.
#     latitude : TYPE ndarray
#         DESCRIPTION.
#     longitude : TYPE ndarray
#         DESCRIPTION.
#     stressor_name : TYPE string
#         DESCRIPTION.
#     stressor_unit : TYPE string
#         DESCRIPTION.

#     Returns map of the sum of the differences normlized by the mean according to the pointer array partition, 
#     values of sum of difference for each catchment (use catchment id list for equivalence between position in the array and ID number of the catchment)
#     -------
#     None.

#     '''


#     var = np.sum(np.diff(aggregated_stressor, axis = 1), axis = 1)
    
#     ref = np.mean(aggregated_stressor, axis = 1)
    
#     change = var/ref
    
#     map_var = make_map(var, pointer_array, spatial_unit)
    
#     map_change = make_map(change, pointer_array, spatial_unit)
    
#     #plt.matshow(map_var, vmin = stats.scoreatpercentile(change,5), vmax = stats.scoreatpercentile(change,95))#ok
    
#     new_map_netcdf(Input.outputDir +'/'+ stressor_name + 'normalized' + '_' + Input.name_timeperiod + '_' + Input.name_scale, map_change, stressor_name + 'normalized', stressor_unit, latitude, longitude)

#     new_map_netcdf(Input.outputDir +'/'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale, map_var, stressor_name, stressor_unit, latitude, longitude)

#     return map_var, var, map_change


    


# def calculate_grad_avg(aggregated_stressor, spatial_unit, pointer_array, latitude, longitude, stressor_name, stressor_unit):
#     '''
    

#     Parameters
#     ----------
#     aggregated_stressor : TYPE ndarray
#         DESCRIPTION. ANNUAL VALUES!!!! SELECT YEARS TO BE INCLUDED IN THE GRADIENT AVERAGE
#     spatial_unit : TYPE ndarray masked
#         DESCRIPTION.
#     pointer_array : TYPE ndarray
#         DESCRIPTION.
#     latitude : TYPE ndarray
#         DESCRIPTION.
#     longitude : TYPE ndarray
#         DESCRIPTION.
#     stressor_name : TYPE string
#         DESCRIPTION.
#     stressor_unit : TYPE string
#         DESCRIPTION.

#     Returns map of the gradient average over the last 10 years according to the pointer array partition, normlaized by mean value
#     values gradient average over the last 10 years for each catchment (use catchment id list for equivalence between position in the array and ID number of the catchment)
#     -------
#     None.

#     '''
#     gradient = np.gradient(aggregated_stressor, axis = 1)
     
#     grad = np.mean(gradient, axis = 1)
    
#     #grad_mean = np.mean(gradient, axis = 1)
    
#     #change = grad_recent/grad_mean
     
#     map_grad = make_map(grad, pointer_array, spatial_unit)
    
#     #map_change = make_map(change, pointer_array, spatial_unit)
    
#     plt.matshow(map_grad, vmin = stats.scoreatpercentile(grad,5), vmax = stats.scoreatpercentile(grad,95))#ok
#     #plt.imshow(s_map_grad/1e6, vmin = stats.scoreatpercentile(s_grad/1e6,5), vmax = stats.scoreatpercentile(s_grad/1e6,95))
#     #yellow is max, blue is min
    
#     #new_map_netcdf(Input.outputDir +'/'+ stressor_name + 'normalized' + '_' + Input.name_timeperiod + '_' + Input.name_scale, map_change, stressor_name + 'normalized', stressor_unit, latitude, longitude)
    
#     new_map_netcdf(Input.outputDir +'/'+ stressor_name + '_' + Input.name_timeperiod + '_' + Input.name_scale,  map_grad, stressor_name, stressor_unit, latitude, longitude)

#     return map_grad, grad