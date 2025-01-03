#!/usr/bin/env python3
import argparse
import os
import os.path
import numpy as np
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
import concurrent.futures
import glob
import re
import h5py
import matplotlib.pyplot as plt

def run_convert_slcvrt_to_hdf(p):
    _cslc_dir = p[0]
    _cslc_file = p[1]
    _hdf_dir = p[2]
    _glacier_shp = p[3]
    _plot_cslc   = p[4]

    with rio.open(f'{_cslc_dir}/{_cslc_file}') as _dst:
        _cslc = _dst.read(1)
        _crs = _dst.crs
        _cslc_transform = _dst.transform
        print(f'crs: {_crs}, transform: {_cslc_transform}')

        if _glacier_shp:
            ak_glaciers = gpd.read_file(_glacier_shp)    # reading inventory shapefile
            ak_glaciers_utm = ak_glaciers.to_crs({'init': _crs})     # reproject shapefile
            
            _cslc_, _ = mask(_dst, ak_glaciers_utm.geometry, invert=False, nodata=np.nan)
            _cslc = _cslc_[0]

    if _plot_cslc:
        # Plot
        scale_factor = 1.0; exp_factor= 0.2
        fig, ax = plt.subplots(figsize=(15, 10))
        cax=ax.imshow(scale_factor*(np.abs(_cslc))**exp_factor, cmap='gray',interpolation=None, origin='upper')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f'{_hdf_dir}/{_cslc_file}.png',dpi=300,bbox_inches='tight')
        plt.close(fig) 

    f = h5py.File(f'{_hdf_dir}/{_cslc_file}.h5','w')     # saving to hdf file (range offset)
    f.create_dataset('cslc',data=_cslc, compression='gzip', compression_opts=4, shuffle=True)
    f.attrs.create('cslc_transform',_cslc_transform)
    f.attrs.create('crs',str(_crs))
    f.close()

    return f'{_cslc_dir}/{_cslc_file} is successfully succussfully converted to {_hdf_dir}/{_cslc_file}.h5'

def createParser(iargs = None):
    '''Commandline input parser'''
    parser = argparse.ArgumentParser(description='convert cslc slcvrt files to hdf')
    parser.add_argument("--inputDir", dest='inputDir',
                         required=True, type=str, help='input directory')
    parser.add_argument("--inputCSLC", dest='inputCSLC',
                         default='merged', type=str, help='input directory for cslc (default: merged)')
    parser.add_argument("--out_dir", dest="out_dir",
                         default='cslc_hdf', type=str, help='output directory for cslc hdf files (default: cslc_hdf)')
    parser.add_argument("--nprocs", dest="nprocs",
                         default=5, type=int, help='Number of processes to run (default: 5)')
    parser.add_argument("--glacier_apply", dest="glacier_apply",
            default=False, action=argparse.BooleanOptionalAction, help='glacier masking is applied (default: False)')
    parser.add_argument("--glacier_shp", dest="glacier_shp",
            default='', type=str, help='glacier shapefile (e.g., RGI)')
    parser.add_argument("--plot_cslc", dest="plot_cslc",
            default=True, action=argparse.BooleanOptionalAction, help='if plotting cslc (default: True)')
    
    return parser.parse_args(args=iargs)

def main(inps):
    inputDir = inps.inputDir
    nprocs = inps.nprocs
    inputCSLC = inps.inputCSLC
    cslc_dir = f'{inputDir}/{inputCSLC}'   #location of cslc slv vrt 

    glacier_apply = inps.glacier_apply
    if inps.glacier_shp:
        glacier_shp = inps.glacier_shp
    else:
        glacier_shp = None

    out_dir = inps.out_dir
    hdf_dir = f'{inputDir}/{out_dir}'    #location of slc hdffiles
    os.makedirs(f'{hdf_dir}', exist_ok='True')

    plot_cslc = inps.plot_cslc

    cslc_filelist = sorted(glob.glob(f'{cslc_dir}/*.slc'))         # list of cslc files
    cslc_files = [ x.split('/')[-1] for x in cslc_filelist]
    pattern = re.compile(r'\b\d{8}.slc\b')    # pattern with yyyymmdd.slc
    cslc_files = [x for x in cslc_files if pattern.match(x)]

    params = []

    for cslc_file in cslc_files:
        if not os.path.isfile(f'{hdf_dir}/{cslc_file}.h5'):
            params.append([cslc_dir,cslc_file,hdf_dir,glacier_shp,plot_cslc])

    with concurrent.futures.ProcessPoolExecutor(max_workers=nprocs) as executor:
        for result in executor.map(run_convert_slcvrt_to_hdf,params):
            print(result)

if __name__ == '__main__':
    # load arguments from command line
    inps = createParser()
    
    # Run workflow
    main(inps)
