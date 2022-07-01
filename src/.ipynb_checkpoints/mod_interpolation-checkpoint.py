import pyinterp
import xarray as xr
import numpy as np
import logging

format_log = "%(levelname)-10s %(asctime)s %(funcName)s : %(message)s"
logging.basicConfig(format=format_log, level=logging.INFO,
                        datefmt="%H:%M:%S")



def interpolation_orca(source_param, target_param, interpolation_param):
    
    logging.info(f'Read source grid info')
    ds_source = xr.open_dataset(source_param['filename'])
    ds_meshmask = xr.open_dataset(source_param['meshmask'])
    mask_coords = (ds_meshmask[source_param['var_mask']].values.ravel() == 0)
    coords = np.vstack((ds_meshmask[source_param['lon_name']].values.ravel()[~mask_coords],
                        ds_meshmask[source_param['lat_name']].values.ravel()[~mask_coords])).T
    
    
    logging.info(f'Read target grid info')
    if target_param['filename'] != '':
        if target_param['filename'][:-3] == '.nc':
            ds_target = xr.open_dataset(target_param['filename'])
        else:
            ds_target = xr.open_zarr(target_param['filename'])
        lon_out = ds_target.coords[target_param['lon_name']].values
        lat_out = ds_target.coords[target_param['lat_name']].values
        my, mx = np.meshgrid(ds_target[target_param['lat_name']].values, 
                             ds_target[target_param['lon_name']].values)
    elif (target_param['filename'] == '') and (type(target_param['lat_vector'])==np.ndarray) and (type(target_param['lon_vector'])==np.ndarray):
        lon_out = target_param['lon_vector']
        lat_out = target_param['lat_vector']
        my, mx = np.meshgrid(target_param['lat_vector'], 
                             target_param['lon_vector'])
    else:
        logging.info('Unknown output grid format!!!!!')
        
    coords_ds_target = np.vstack((mx.ravel(), my.ravel())).T
    
    # Create output dataset
    ds_out = xr.Dataset()
    # loop over time variable
    for vtime in ds_source[source_param['time_name']].values:
        logging.info(f'processing time_counter {np.datetime_as_string(vtime)}')
        # create RTree
        mesh = pyinterp.RTree()
        mesh.packing(coords,
                     ds_source[source_param['var_name']].where(ds_source[source_param['time_name']] == vtime, drop=True).values.ravel()[~mask_coords]
                    )
        
        # interpolation
        res, _ = mesh.inverse_distance_weighting(coords_ds_target, within=not(interpolation_param['extrapolate']), radius=interpolation_param['radius_of_search'])
        res = np.expand_dims(res.reshape(my.shape).T, axis=0)   # refomate array result
        
        # save time slice to dataset
        ds_out_tmp = xr.Dataset({target_param['var_name'] : (('time', 'lat', 'lon'), res)},
                               coords={target_param['time_name']: [vtime],
                                       target_param['lon_name']: lon_out, 
                                       target_param['lat_name']: lat_out, 
                                       })
        
        # merge into final dataset
        ds_out = xr.merge([ds_out, ds_out_tmp])
        
        del ds_out_tmp
    
    ds_out.to_netcdf(interpolation_param['output_filename'])
    logging.info(f'Interpolated field save in : {interpolation_param["output_filename"]}')
    
    return None

