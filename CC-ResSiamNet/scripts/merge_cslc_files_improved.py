#!/usr/bin/env python3
import argparse
import os
import re
from osgeo import gdal, osr
import numpy as np
from glob import glob

import rasterio
from rasterio.merge import merge
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from pyproj import Transformer

def read_hdr_info(hdr_file):
    with open(hdr_file, 'r') as f:
        content = f.read()
    
    # Extract UTM zone and hemisphere
    map_info = re.search(r'map info = {([^}]*)}', content)
    if map_info:
        map_info = map_info.group(1).split(',')
        utm_zone = int(map_info[7])
        hemisphere = map_info[8].strip()
        xres = float(map_info[5])
        yres = float(map_info[6])
    
    # Extract coordinate system string
    coord_system = re.search(r'coordinate system string = {([^}]*)}', content)
    if coord_system:
        coord_system = coord_system.group(1)
    
    return utm_zone, hemisphere, coord_system, xres, yres

def get_epsg_from_utm(utm_zone, hemisphere):
    if hemisphere.lower() == 'north':
        return 32600 + utm_zone
    else:
        return 32700 + utm_zone

# def reproject_file(input_file, output_file, target_epsg, xres, yres):
#    gdal.Warp(output_file, input_file, dstSRS=f'EPSG:{target_epsg}', xRes=xres, yRes=yres)

def reproject_file(input_file, output_file, target_epsg, xres, yres):
    gdal.Warp(output_file, input_file, 
              dstSRS=f'EPSG:{target_epsg}', 
              xRes=xres, yRes=yres,
              resampleAlg=gdal.GRA_Bilinear,  # Use bilinear resampling
              format='GTiff',
              outputType=gdal.GDT_CFloat32)  # Ensure output is complex float

#def merge_files(input_files, output_file, xres, yres):
#    gdal.Warp(output_file, input_files, format='GTiff', xRes=xres, yRes=yres, dstNodata=np.nan, srcNodata=np.nan)

def merge_files(input_files, output_file, xres, yres):
    # Determine the extent of all input files
    min_x, max_x, min_y, max_y = None, None, None, None
    projection = None
    
    for file in input_files:
        ds = gdal.Open(file)
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        
        if projection is None:
            projection = proj
        
        x1 = gt[0]
        y1 = gt[3]
        x2 = gt[0] + gt[1] * ds.RasterXSize
        y2 = gt[3] + gt[5] * ds.RasterYSize
        
        min_x = min(x1, x2, min_x) if min_x is not None else min(x1, x2)
        max_x = max(x1, x2, max_x) if max_x is not None else max(x1, x2)
        min_y = min(y1, y2, min_y) if min_y is not None else min(y1, y2)
        max_y = max(y1, y2, max_y) if max_y is not None else max(y1, y2)
        
        ds = None
    
    # Calculate the dimensions of the output raster
    width = int((max_x - min_x) / xres)
    height = int((max_y - min_y) / abs(yres))
    
    # Create the output dataset
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, width, height, 1, gdal.GDT_Float32, 
                           options=['COMPRESS=LZW', 'PREDICTOR=2'])
    
    # Set the geotransform and projection
    out_ds.SetGeoTransform((min_x, xres, 0, max_y, 0, -abs(yres)))
    out_ds.SetProjection(projection)
    
    # Initialize the output array with NaNs
    out_data = np.full((height, width), np.nan, dtype=np.float32)
    
    # Process each input file
    for file in input_files:
        ds = gdal.Open(file)
        gt = ds.GetGeoTransform()
        
        # Calculate the position of this file in the output raster
        x_offset = int((gt[0] - min_x) / xres)
        y_offset = int((max_y - gt[3]) / abs(yres))
        
        # Read the data
        data = ds.ReadAsArray()
        
        # If the data is complex, take the magnitude
        if data.dtype == np.complex64 or data.dtype == np.complex128:
            data = np.abs(data)
        
        # Update the output array, keeping the first non-NaN value
        mask = ~np.isnan(data) & np.isnan(out_data[y_offset:y_offset+data.shape[0], 
                                                   x_offset:x_offset+data.shape[1]])
        out_data[y_offset:y_offset+data.shape[0], 
                 x_offset:x_offset+data.shape[1]][mask] = data[mask]
        
        ds = None
    
    # Write the output data
    out_ds.GetRasterBand(1).WriteArray(out_data)
    out_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    
    # Close the dataset
    out_ds = None
    
    print(f"Merged file created: {output_file}")

def update_hdr(input_hdr, output_hdr, merged_dataset):
    # Read the original HDR content
    with open(input_hdr, 'r') as f:
        content = f.read()
    
    # Update map info
    geotransform = merged_dataset.GetGeoTransform()
    _ = merged_dataset.GetProjection().split('"')[-2]
    new_map_info = f"UTM, 1, 1, {geotransform[0]}, {geotransform[3]}, {abs(geotransform[1])}, {abs(geotransform[5])}, {_}, North, WGS-84"
    # content = re.sub(r'map info = {[^}]*}', f'map info = {{{new_map_info}}}', content)
    content = re.sub(r'map info\s*=\s*{.*}', f'map info = {{{new_map_info}}}', content)
    
    # Update coordinate system string
    srs = osr.SpatialReference(wkt=merged_dataset.GetProjection())
    new_coord_system = srs.ExportToWkt()
    # content = re.sub(r'coordinate system string = {[^}]*}', f'coordinate system string = {{{new_coord_system}}}', content)
    content = re.sub(r'coordinate system string\s*=\s*{.*}', f'coordinate system string = {{{new_coord_system}}}', content)    

    # Update dimensions
    content = re.sub(r'samples\s*=\s*\d+', f'samples = {merged_dataset.RasterXSize}', content)
    # content = re.sub(r'samples = \d+', f'samples = {merged_dataset.RasterXSize}', content)
    # content = re.sub(r'lines = \d+', f'lines = {merged_dataset.RasterYSize}', content)
    content = re.sub(r'lines\s*=\s*\d+', f'lines = {merged_dataset.RasterYSize}', content)
    
    # Write updated content to new HDR file
    with open(output_hdr, 'w') as f:
        f.write(content)

def generate_amplitude_image(input_slc, output_amplitude):
    # Open the input SLC file
    slc_dataset = gdal.Open(input_slc, gdal.GA_ReadOnly)
    
    # Read the complex data
    slc_data = slc_dataset.ReadAsArray()
    
    # Calculate amplitude: 20 * log10(abs(complex value))
    amplitude_data = 20 * np.log10(np.abs(slc_data))
    
    # Create the output dataset
    driver = gdal.GetDriverByName('GTiff')
    amplitude_dataset = driver.Create(output_amplitude, slc_dataset.RasterXSize, slc_dataset.RasterYSize, 1, gdal.GDT_Float32)
    
    # Set the geotransform and projection
    amplitude_dataset.SetGeoTransform(slc_dataset.GetGeoTransform())
    amplitude_dataset.SetProjection(slc_dataset.GetProjection())
    
    # Write the amplitude data
    amplitude_dataset.GetRasterBand(1).WriteArray(amplitude_data)
    
    # Close the datasets
    slc_dataset = None
    amplitude_dataset = None
    
    print(f"Amplitude image generated: {output_amplitude}")

def merge_multiple_slc_files(input_files, output_prefix, generate_amplitude=False):
    # Read EPSG code from the first file
    first_hdr = input_files[0].replace('.slc', '.hdr')
    utm_zone, hemisphere, _, xres, yres = read_hdr_info(first_hdr)
    target_epsg = get_epsg_from_utm(utm_zone, hemisphere)
    
    print(f"Target EPSG: {target_epsg}")
    
    # Process all input files
    processed_files = []
    for file in input_files:
        base, ext = os.path.splitext(file)
        hdr_file = f"{base}.hdr"
        
        # Read EPSG code for current file
        utm_zone, hemisphere, _, _, _ = read_hdr_info(hdr_file)
        current_epsg = get_epsg_from_utm(utm_zone, hemisphere)
        
        if current_epsg == target_epsg:
            print(f"File {file} already in target EPSG. Skipping reprojection.")
            processed_files.append(file)
        else:
            print(f"Reprojecting {file} from EPSG:{current_epsg} to EPSG:{target_epsg}")
            reprojected_file = f"{base}_reprojected{ext}"
            reproject_file(file, reprojected_file, target_epsg, xres, yres)
            processed_files.append(reprojected_file)

    # Merge processed files
    merged_slc = f"{output_prefix}.slc"
    merge_files(processed_files, merged_slc, xres, yres)

    # Create VRT
    merged_vrt = f"{output_prefix}.slc.vrt"
    gdal.BuildVRT(merged_vrt, merged_slc)

    # Update HDR
    output_hdr = f"{output_prefix}.hdr"
    merged_dataset = gdal.Open(merged_slc)
    update_hdr(first_hdr, output_hdr, merged_dataset)

    # Generate amplitude image if requested
    if generate_amplitude:
        amplitude_file = f"{output_prefix}_amplitude.tif"
        generate_amplitude_image(merged_slc, amplitude_file)

    # Clean up temporary files
    for file in processed_files:
        if file.endswith('_reprojected.slc'):
            os.remove(file)

    print(f"Merged SLC: {merged_slc}")
    print(f"Merged VRT: {merged_vrt}")
    print(f"Updated HDR: {output_hdr}")

def display_geotiff(input_dir, input_file, cmap='jet', vmin=None, vmax=None):

    file_path = f'{input_dir}/{input_file}'
    png_path = f'{input_dir}/{input_file.replace("tif","png")}'

    # Open the GeoTIFF file
    with rasterio.open(file_path) as src:
        # Read the data
        image = src.read(1)  # Assuming single band image

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display the image
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

    # Remove axis ticks
    ax.axis('off')

    fig.savefig(png_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Figure saved to {png_path}")

def plot_geotiff_over_google_earth(input_dir, input_file, vmin=None, vmax=None, zoom_level=10, cmap='jet'):
    """
    Plot GeoTIFF data over a Google Earth image.

    Args:
    geotiff_path (str): Path to the GeoTIFF file.
    output_path (str, optional): Path to save the output image. If None, the plot will be displayed.
    zoom_level (int, optional): Zoom level for the Google Earth image. Default is 14.
    """
    geotiff_path = f'{input_dir}/{input_file}'
    png_path = f'{input_dir}/{input_file.replace("tif","png")}'

    # Open the GeoTIFF file
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)  # Read the first band
        bounds = src.bounds
        src_crs = src.crs

        # Create a transformer object
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

        # Transform bounds to WGS84
        left, bottom = transformer.transform(bounds.left, bounds.bottom)
        right, top = transformer.transform(bounds.right, bounds.top)
        bounds_wgs84 = (left, right, bottom, top)

    # Create a figure and axis with cartopy projection
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add Google Earth imagery (in grayscale)
    google_earth = cimgt.GoogleTiles(style='satellite')
    ax.add_image(google_earth, zoom_level, cmap='gray', alpha=0.1)

    # Plot the GeoTIFF data
    img = ax.imshow(data, extent=bounds_wgs84,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap, alpha=0.8)

    # Set extent to match the GeoTIFF bounds (in WGS84)
    ax.set_extent(bounds_wgs84, crs=ccrs.PlateCarree())

    # Add gridlines (only left and top)
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = True
    gl.right_labels = False
    gl.bottom_labels = False
    gl.left_labels = True

    # Save or display the plot
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {png_path}")

def createParser(iargs = None):
    '''Commandline input parser'''
    parser = argparse.ArgumentParser(description='Merging cslc files')
    parser.add_argument("--inputPattern",
                  required=True, type=str, help="File pattern to match for cslc files (e.g., 'T160*/cslc/2*.slc' should be in single or double quote)")
    parser.add_argument("--outputDir", 
                  required=True, type=str, help='directory for output geotiff file')
    parser.add_argument("--outputPrefix",
                  required=True, type=str, help="output prefix (e.g., '20180808' should be in single or double quote)")
    parser.add_argument("--generate_amplitude", action='store_true', default=False,
                  help="flag if generate amplitude (default: False)") 
    parser.add_argument("--plot_amplitude", action='store_true', default=False,
                  help="flag if plot generated amplitude (default: False)")
    return parser.parse_args(args=iargs)

def main(inps):
    
    inputPattern = inps.inputPattern
    print(f'input pattern: {inputPattern}')
    outputDir = inps.outputDir
    os.makedirs(outputDir, exist_ok='True')
    print(f'output directory: {outputDir}')
    outputPrefix = inps.outputPrefix
    print(f'output prefix: {outputPrefix}')

    output_prefix = f'{outputDir}/{outputPrefix}'
    
    input_files = sorted(glob(f'{inputPattern}'))

    if inps.generate_amplitude and inps.plot_amplitude:
        generate_amplitude, plot_amplitude = True, True
    elif inps.generate_amplitude and not inps.plot_amplitude:
        generate_amplitude, plot_amplitude = True, False
    else:
        generate_amplitude, plot_amplitude = False, False

    merge_multiple_slc_files(input_files, output_prefix, generate_amplitude=generate_amplitude)

    outputTIFF = f'{outputPrefix}_amplitude.tif'

    if plot_amplitude:
        display_geotiff(outputDir, outputTIFF, cmap='gray')
        # plot_geotiff_over_google_earth(outputDir, outputTIFF, cmap='gray')

if __name__ == '__main__':
    # load arguments from command line
    inps = createParser()

    print("==================================================================")
    print("                 Merging multiple cslc files")
    print("==================================================================")
    
    # Run the main function
    main(inps)    
