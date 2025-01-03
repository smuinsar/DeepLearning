import geopandas as gpd
from osgeo import gdal, osr
import requests
import os
import rasterio
from rasterio.merge import merge
from shapely.geometry import box
from shapely import wkt
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, transform_bounds
from scipy import ndimage
import numpy as np  
import itertools
import datetime as dt
import h5py
from scipy.ndimage import median_filter

def url_generator(corner: tuple, 
                  layer: str ='last'):
    
    assert(layer in ['first', 'last', 'treecover2001'])
    
    base_url = 'https://storage.googleapis.com/global-surface-water/downloads2021/occurrence'
    lon, lat = corner
    if lon < 0:
        lon_str = f'{abs(lon):d}W'
    else:
        lon_str = f'{abs(lon):d}E'
    if lat >= 0:
        lat_str = f'{abs(lat):d}N'
    else:
        lat_str = f'{abs(lat):d}S'
    return f'{base_url}/occurrence_{lon_str}_{lat_str}v1_4_2021.tif'

def get_global_surface_water(poly,epsg,dst_height,dst_width,dst_transform,maskdir='mask_files',maskfilename='mask.tif'):
    '''
    downloading global surface water from https://global-surface-water.appspot.com
    reproject surface water to desired coordinates
    modified from https://github.com/OPERA-Cal-Val/DSWx-Validation-Experiments/blob/387acb91d7cce3ec62fac47394c219c74b9dee82/marshak/auxiliary_water_masks/Peckel%20Water%20Mask.ipynb#L63
      
      poly: approximate coverage of inputs
      epsg: epsg code for destination
      dst_height/dst_width: destination height and width
      dst_transform: destination transform (rasterio)
    '''

    # box inputs are minx, miny, maxx, maxy
    geometries = [box(-180 + i * 10, 
                    80 - (j + 1) * 10, 
                    -180 + (i + 1) * 10,
                    80 - (j) * 10) for i in range(36) for j in range(14)]

    # Upper left corner
    ul_corners = [(-180 + (i) * 10, 80 - (j) * 10) for i in range(36) for j in range(14)]
    # data
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=CRS.from_epsg(4326))

    gdf['source_url'] = list(map(url_generator, ul_corners))
    gdf['year'] = 2021

    # checking which water polygon intersects
    source_urls = []

    for row in gdf.itertuples():
        _geom = row.geometry
        check_intersects = _geom.intersects(wkt.loads(poly))

        if check_intersects:
            source_urls.append(row.source_url)

    # downloading geotiff file containing water mask
    os.makedirs(f'{maskdir}', exist_ok='True')

    filelist = []
    for _url in source_urls:
        
        filename = _url.split('/')[-1]

        filelist.append(f'{maskdir}/{filename}')

        if not os.path.isfile(f'{maskdir}/{filename}'):
            filedata = requests.get(_url).content
            with open(f'{maskdir}/{filename}', "wb") as file:
                file.write(filedata)

    # when polygon covers multiple water masks, merged
    raster_to_mosaic = []
    if (len(source_urls) > 1):
        for p in filelist:
            raster = rasterio.open(p)
            raster_to_mosaic.append(raster)
        mosaic, output = merge(raster_to_mosaic,nodata=255)

        output_meta = raster.meta.copy()
        output_meta.update(
            {"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "dtype": np.uint8,
                "transform": output
            }
        )
        mergefile = f'merge_tmp_{maskfilename}'
        with rasterio.open(f'{maskdir}/{mergefile}', 'w', **output_meta) as m:
            m.write(mosaic)
        filename = mergefile

    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(epsg)

    _in = rasterio.open(f'{maskdir}/{filename}','r')
    source = _in.read(1)    
    src_transform = _in.transform

    kwargs = _in.meta
    kwargs['transform'] = dst_transform
    kwargs['width'] = dst_width
    kwargs['height'] = dst_height

    with rasterio.open(f'{maskdir}/{maskfilename}','w', **kwargs) as dst:
        dest = np.zeros((dst_height,dst_width), np.uint8)
        reproject(
            source,
            dest,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)
        dst.write(dest, indexes = 1)

    with rasterio.open(f'{maskdir}/{maskfilename}') as ds:
        water_mask = ds.read(1)
    
    return water_mask

def get_global_surface_water_from_hdr(hdr_content, maskdir='mask_files', maskfilename='mask.tif'):
    '''
    Download global surface water from https://global-surface-water.appspot.com
    and reproject surface water to desired coordinates based on HDR file info.
    
    hdr_content: string containing the content of the HDR file
    maskdir: directory to store mask files
    maskfilename: filename for the output mask
    '''
    
    # Parse HDR content
    hdr_lines = hdr_content.split('\n')
    hdr_info = {}
    for line in hdr_lines:
        if '=' in line:
            key, value = line.split('=', 1)
            hdr_info[key.strip()] = value.strip()
    
    # Extract relevant information
    samples = int(hdr_info['samples'])
    lines = int(hdr_info['lines'])
    map_info = hdr_info['map info'].strip('{}').split(', ')
    epsg = int(map_info[7])
    
    # Calculate bounding box in input CRS
    x_min = float(map_info[3])
    y_max = float(map_info[4])
    pixel_width = float(map_info[5])
    pixel_height = float(map_info[6])
    x_max = x_min + samples * pixel_width
    y_min = y_max - lines * pixel_height
    
    # Convert bounding box to EPSG:4326
    src_crs = CRS.from_epsg(epsg)
    dst_crs = CRS.from_epsg(4326)
    
    lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, dst_crs, x_min, y_min, x_max, y_max)
    
    # Create polygon in EPSG:4326
    poly = box(lon_min, lat_min, lon_max, lat_max)
    
    # Create destination transform (in original CRS)
    dst_transform = rasterio.transform.from_origin(x_min, y_max, pixel_width, pixel_height)
    
    # box inputs are minx, miny, maxx, maxy
    geometries = [box(-180 + i * 10, 
                    80 - (j + 1) * 10, 
                    -180 + (i + 1) * 10,
                    80 - (j) * 10) for i in range(36) for j in range(14)]

    # Upper left corner
    ul_corners = [(-180 + (i) * 10, 80 - (j) * 10) for i in range(36) for j in range(14)]
    
    # data
    gdf = gpd.GeoDataFrame(geometry=geometries, crs=CRS.from_epsg(4326))
    gdf['source_url'] = list(map(url_generator, ul_corners))
    gdf['year'] = 2021

    # checking which water polygon intersects
    source_urls = []
    for row in gdf.itertuples():
        _geom = row.geometry
        check_intersects = _geom.intersects(poly)
        if check_intersects:
            source_urls.append(row.source_url)

    # downloading geotiff file containing water mask
    os.makedirs(maskdir, exist_ok=True)

    filelist = []
    for _url in source_urls:
        filename = _url.split('/')[-1]
        filelist.append(f'{maskdir}/{filename}')
        if not os.path.isfile(f'{maskdir}/{filename}'):
            filedata = requests.get(_url).content
            with open(f'{maskdir}/{filename}', "wb") as file:
                file.write(filedata)

    # when polygon covers multiple water masks, merge them
    raster_to_mosaic = []
    if len(source_urls) > 1:
        for p in filelist:
            raster = rasterio.open(p)
            raster_to_mosaic.append(raster)
        mosaic, output = merge(raster_to_mosaic, nodata=255)

        output_meta = raster.meta.copy()
        output_meta.update(
            {"driver": "GTiff",
             "height": mosaic.shape[1],
             "width": mosaic.shape[2],
             "dtype": np.uint8,
             "transform": output
            }
        )
        mergefile = f'merge_tmp_{maskfilename}'
        with rasterio.open(f'{maskdir}/{mergefile}', 'w', **output_meta) as m:
            m.write(mosaic)
        filename = mergefile
    else:
        filename = filelist[0].split('/')[-1]

    _in = rasterio.open(f'{maskdir}/{filename}', 'r')
    source = _in.read(1)    
    src_transform = _in.transform

    kwargs = _in.meta
    kwargs['transform'] = dst_transform
    kwargs['width'] = samples
    kwargs['height'] = lines
    kwargs['crs'] = src_crs

    with rasterio.open(f'{maskdir}/{maskfilename}', 'w', **kwargs) as dst:
        dest = np.zeros((lines, samples), np.uint8)
        reproject(
            source,
            dest,
            src_transform=src_transform,
            src_crs=dst_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=Resampling.nearest)
        dst.write(dest, indexes=1)

    with rasterio.open(f'{maskdir}/{maskfilename}') as ds:
        water_mask = ds.read(1)
    
    return water_mask

def stream_cslc(s3f,pol):
    '''
    streaming OPERA CSLC in S3 bucket and retrieving CSLC and parameters from HDFs
    '''

    grid_path = f'data'
    metadata_path = f'metadata'
    burstmetadata_path = f'{metadata_path}/processing_information/input_burst_metadata'
    id_path = f'identification'

    with h5py.File(s3f,'r') as h5:
        cslc = h5[f'{grid_path}/{pol}'][:]
        azimuth_carrier_phase = h5[f'{grid_path}/azimuth_carrier_phase'][:]
        flattening_phase = h5[f'{grid_path}/flattening_phase'][:]
        xcoor = h5[f'{grid_path}/x_coordinates'][:]
        ycoor = h5[f'{grid_path}/y_coordinates'][:]
        dx = h5[f'{grid_path}/x_spacing'][()].astype(int)
        dy = h5[f'{grid_path}/y_spacing'][()].astype(int)
        epsg = h5[f'{grid_path}/projection'][()].astype(int)
        sensing_start = h5[f'{burstmetadata_path}/sensing_start'][()].astype(str)
        sensing_stop = h5[f'{burstmetadata_path}/sensing_stop'][()].astype(str)
        dims = h5[f'{burstmetadata_path}/shape'][:]
        bounding_polygon = h5[f'{id_path}/bounding_polygon'][()].astype(str) 
        orbit_direction = h5[f'{id_path}/orbit_pass_direction'][()].astype(str)
        center_lon, center_lat = h5[f'{burstmetadata_path}/center']
        wavelength = h5[f'{burstmetadata_path}/wavelength'][()].astype(str)
        cslc = cslc*np.conj(np.exp(1j*azimuth_carrier_phase))*np.conj(np.exp(1j*flattening_phase))
    
    return cslc, azimuth_carrier_phase, flattening_phase, xcoor, ycoor, dx, dy, epsg, sensing_start, sensing_stop, dims, bounding_polygon, orbit_direction, center_lon, center_lat, wavelength

def convert_to_slcvrt(xcoor, ycoor, dx, dy, epsg, slc, date, outdir):

     os.makedirs(outdir,exist_ok=True)

     height, width = slc.shape

     slc_file = outdir + '/' + date+'.slc'
     slc_vrt = slc_file+'.vrt'

     if not (os.path.exists(slc_file) and os.path.exists(slc_vrt)):

          outtype = '<f'  #little endian (float)
          dtype = gdal.GDT_CFloat32
          drvout = gdal.GetDriverByName('ENVI')
          raster_out = drvout.Create(slc_file, width,height, 1, dtype)
          raster_out.SetGeoTransform([xcoor[0],dx,0.0,ycoor[0],0.0,dy])

          srs = osr.SpatialReference()
          srs.ImportFromEPSG(int(epsg))
          raster_out.SetProjection(srs.ExportToWkt())

          band_out = raster_out.GetRasterBand(1)
          band_out.WriteArray(slc)
          band_out.FlushCache()
          del band_out

          command = 'gdal_translate ' + slc_file + ' ' + slc_vrt + f' > {outdir}/tmp.LOG'
          os.system(command)
     else: 
          print(f'Files ({slc_file}, {slc_vrt}) already exist')

def array2raster(outrasterfile,OriginX, OriginY, pixelWidth,pixelHeight,epsg,array):
    #generating geotiff file from 2D array

    cols = array.shape[1]
    rows = array.shape[0]
    originX = OriginX
    originY = OriginY

    driver = gdal.GetDriverByName('ENVI')
    outRaster = driver.Create(outrasterfile, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def rasterWrite(outtif,arr,transform,crs_from_epsg,dtype=[],nodata=np.nan):
    #writing geotiff using rasterio  
    new_dataset = rasterio.open(outtif, 'w', driver='GTiff',
                            height = arr.shape[0], width = arr.shape[1],
                            count=1, dtype=dtype,
                            crs=crs_from_epsg,
                            transform=transform,nodata=nodata)
    new_dataset.write(arr, 1)
    new_dataset.close() 

def median_filt(off,size=6):
    return median_filter(off, size=size)

def filter_large_image_with_overlap(image, chunk_size=1000, filter_size=3, overlap=50):
    height, width = image.shape
    filtered_image = np.full_like(image, np.nan)
    
    for i in range(0, height, chunk_size - overlap):
        for j in range(0, width, chunk_size - overlap):
            # Define chunk boundaries with overlap
            i_start = max(0, i - overlap)
            i_end = min(height, i + chunk_size + overlap)
            j_start = max(0, j - overlap)
            j_end = min(width, j + chunk_size + overlap)
            
            # Extract chunk with overlap
            chunk = image[i_start:i_end, j_start:j_end]
            
            # Apply median filter to chunk
            filtered_chunk = median_filter(chunk, size=filter_size, mode='constant', cval=np.nan)
            
            # Define the region to update in the output image
            out_i_start = max(0, i)
            out_i_end = min(height, i + chunk_size)
            out_j_start = max(0, j)
            out_j_end = min(width, j + chunk_size)
            
            # Update the output image
            filtered_image[out_i_start:out_i_end, out_j_start:out_j_end] = filtered_chunk[
                out_i_start - i_start:out_i_end - i_start,
                out_j_start - j_start:out_j_end - j_start
            ]
    
    return filtered_image

def outlier_removal(off,std_interval=2):
    # std_interval: 2 (95% confidence interval), 1 (68% confidence interval)
    thr = np.nanmean(off) + std_interval * np.nanstd(off)
    off[abs(off)>thr] = np.nan
    return off

def sort_pair(days,minDelta=5,maxDelta=90):
    '''
    days: list of days
    minDelta: minimum temporal baseline (days), maxDelta: maximum temporal baseline (days)
    
    determining pairs of reference and secondary dates
    given minimum and maximum temporal baseline
    '''
    date_pair = list(itertools.combinations(days,2))   #possible pair of InSAR dates

    refDates = []
    secDates = []
    deltas = []

    for refDate, secDate in date_pair:
        delta = dt.datetime.strptime(secDate, "%Y%m%d") - dt.datetime.strptime(refDate, "%Y%m%d")
        delta = int(delta.days)
        
        if (delta > minDelta) & (delta < maxDelta):
            refDates.append(refDate)
            secDates.append(secDate)
            deltas.append(delta)
    
    return refDates, secDates
