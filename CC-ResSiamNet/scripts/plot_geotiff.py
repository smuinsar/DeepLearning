#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np

def geotiff_to_png(input_file):
    # Open the GeoTIFF file
    dataset = gdal.Open(input_file)
    
    # Read the data as a numpy array
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    
    # Create a new figure
    plt.figure(figsize=(10, 8))
    
    # Plot the image
    im = plt.imshow(array, cmap='jet')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.4)
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    output_file = input_file.replace('.tif','.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Convert GeoTIFF to PNG with colorbar")
    parser.add_argument("input", help="Input GeoTIFF file name")
    args = parser.parse_args()

    output_file = geotiff_to_png(args.input)
    print(f"Converted {args.input} to {output_file}")

if __name__ == "__main__":
    main()
