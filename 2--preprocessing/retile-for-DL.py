#!/usr/bin/env python
# coding: utf-8

'''
NOTE 2021/02/06 - This is a "frozen" Python saved to document the processing in Ross Winans's masters thesis. 
This is the exact version used to retile the 2015 DAR Coastal Imagery for the State of Hawaii. This script is saved strictly for posterity/documentation. 
Future development and improvements will happen on the main retile-for-DL.py script 
(https://github.com/rosswin/mapping_marine_debris/blob/master/preprocessing/retile-for-DL.py). It is not recommended that future users start with
this verion of the script for any future work.

NOTE 20201/02/06 - I chased down bullet point 8 in late 2020, but never got around to writing the in final results. There was actually no issue to resolve. The
distortion came from the source imagery tiles, not this script. The version you see here was the actual script use to preprocess data.

NOTE 2020/01/01 - This script is finicky and only works on DAR 2015 imagery. The code needs to be reformatted to take generic data sets/annotation formats. 
Everythign here is still a rough draft. I'll try to keep a log of issues and incrementally improve over time.

DESCRIPTION:
This script does 4 things:
1) takes a set of Geotiffs and "chips" the images with user-specified image size (in pixels) and overlap (in pixels).
2) writes a "chip index" (cindex), which are the envelopes of each chip and its affine transformation matrix (used for converting lat/long to pixel coordinates)
3) compares the cindex against a set of annotation polygons, the chips that contain an annotation (positive chips) are written to jpeg images for use in deep learning
4) Reformats the input annotations with pixel coordinates for each chips, writes those to a CSV file for use in deep learning.

TABLE OF CONTENTS:
1. HELPER FUNCTIONS- smaller pieces of code that do things such as convert image coordinates to pixel coordinates, or determine which annotations intersect which image.
2. BACKBONE FUNCTION- this function wraps the smaller functions together, one after the other, from input to output.
3. PARSE ARGS & LOGGING- the code that handles the user Input Flags detailed below and the writing of critical info to a log file.
4. MULTIPROCESSING- code that handles multi-processing to speed computation.

INPUT FLAGS:
1) -f --filelist:    a .txt file of absolute paths to geotiff files to be chipped
2) -o --outdir:      a directory to store the cindex, jpegs, reformatted annotations, and a log file
3) -a --annotations: a geopackage that comprises all of our annotations. Right now this is finicky and only designed to work with 2015 DAR coastal data

OTHER FLAGS:
1) -c --chipsize: the desired output chip size in pixels. Default is 512x512px
2) -s --stride:   the desired overlap in pixels. Default is 256x256px (50% overlap if using default chipsize)

OUTPUT FILES:
1) cindex: a geopackage. Represents envelopes of all chips. Currently defaults to the input geotiff's name with "_cindex" appended.
2) jpegs: positive chip jpegs to be used as the imagery in deep learning.
3) final_annotations.csv: the reformatted annotations with all info you could possibly need for deep learning
4) log.txt: a log file. Hopefully you don't need it.

TODO:
1) Generalize the code to read/write more than geotiffs/jpegs
2) Generalize the code to take points and polygons as annotation input. Maybe write lists of pos/neg chips for point files?
3) allow more user control of things (like num of CPU cores, whether to write pos/neg csv files, control out annotation format, etc.)
4) maybe allow other formats- such as PASCAL VOC
5) currently only writes 3 band images.
6) currently requires input geotiffs's row/col counts to be exactly divisible by chip size
7) script kicks out wierd ERROR: 4. It is working though... it must be a gdal or rasterio thing. Research required. 
    NOTE: this may be resolved in rasterio v1.1. https://github.com/mapbox/rasterio/commit/eb6549ac626a46dc52584fa2ac7888f5c4ddbe3a
8 ) Fixed in late 2020.
9) There is a potential unaddressed edge case. If annotation boxes are vary large, and the chip size is very small, there may not be
    an image chip that fully contains the annotation. The script would currently skip that. It's not really an issue given our data,
    so I am skipping for now. See the NOTE in return_intersection() on how to get started repairing this issue.  
10) figure out how to make the logs work with multiprocessing. Currently a jumbled mess. Could also improve error trapping/reporting/handling.

'''

import os
import sys
import time
import math
import csv

import multiprocessing
import logging
from itertools import product, repeat
import optparse

import rasterio
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.mask import mask

import pandas as pd
import geopandas as gpd
import numpy as np
import fiona
from shapely import geometry

#########################
## 1. HELPER FUNCTIONS ##
#########################

def grid_calc(width, height, stride):
    #get all the upper left (ul) pixel values of our chips. Combine those into a list of all the chip's ul pixels. 
    #NOTE: THESE ARE THE OPPOSITE OF WHAT I THINK THEY SHOULD BE (HEIGHT/WIDTH SWAPPED)
    #I DON"T KNOW WHY IT WORKS DONT TOUCH THIS FUNCTION UNLESS YOURE GOING TO OWN IT.
    x_chips = height // stride
    x_ul = [stride * s for s in range(x_chips)]

    y_chips = width // stride
    y_ul = [stride * s1 for s1 in range(y_chips)]

    xy_ul = []
    for x in x_ul:
        for y in y_ul:
            xy_ul.append((x, y))

    return xy_ul

def build_window(in_src, in_xy_ul, in_chip_size, in_chip_stride):

    out_window = windows.Window(col_off = in_xy_ul[0],
                                row_off = in_xy_ul[1],
                                width=in_chip_size,
                                height=in_chip_size)
    
    out_win_transform = windows.transform(out_window, in_src.transform)
     
    col_id = in_xy_ul[1] // in_chip_stride
    row_id = in_xy_ul[0] // in_chip_stride
    out_win_id = f'{col_id}_{row_id}'
    
    out_win_bounds = windows.bounds(out_window, out_win_transform)
    
    return out_window, out_win_transform, out_win_bounds, out_win_id

def make_gdf(polygons, attr_dict, out_crs, out_gdf_path='none'):
    gs = gpd.GeoSeries(polygons)
    
    df = pd.DataFrame(data=attr_dict)
    
    gdf = gpd.GeoDataFrame(df, geometry=gs)
    gdf.crs=out_crs
    
    #optionally write a file if the path was provided. NOTE: COULD EXPAND THIS TO HANDLE FORMATS OTHER THAN GPKG.
    if out_gdf_path != 'none':
        if os.path.exists(os.path.dirname(os.path.abspath(out_gdf_path))):
            print(f"Writing: {out_gdf_path}")
            
            gdf.to_file(out_gdf_path, driver='GPKG')
    
    #regardless of writing output, return GDF
    return gdf

def write_jpeg(in_data, in_count, in_size, in_win_transform, in_src_crs, in_out_path):
    #building a custom jpeg profile for our chip due to some gdal/rasterio bugs in walking from input geotiff to output jpeg
    profile={'driver': 'JPEG',
        'count': in_count,
        'dtype': rasterio.ubyte,
        'height': in_size,
        'width': in_size,
        'transform': in_win_transform,
        'crs': in_src_crs}
        
    #write the chip
    with rasterio.open(in_out_path, 'w', **profile) as dst:
        dst.write(in_data)

def coords_2_pix(in_bounds, in_affine):
    xmin = in_bounds[0]
    ymin = in_bounds[1]
    xmax = in_bounds[2]
    ymax = in_bounds[3]
    
    xs = (xmin, xmax)
    ys = (ymin, ymax)
    
    pix_coords = rasterio.transform.rowcol(in_affine, xs, ys)
    
    pix_bounds = (pix_coords[0][1], pix_coords[1][1], pix_coords[0][0], pix_coords[1][0])
    
    return pix_bounds

def coords_2_pix_gdf(gdf):
    gdf['x_min'] = gdf.bounds.minx
    gdf['y_min'] = gdf.bounds.miny
    gdf['x_max'] = gdf.bounds.maxx
    gdf['y_max'] = gdf.bounds.maxy
    
    gdf['px_x_min'] = (gdf['x_min'] - gdf['c_x_min']) // gdf['a0']
    gdf['px_x_max'] = (gdf['x_max'] - gdf['c_x_min']) // gdf['a0']

    gdf['px_y_min'] = (gdf['y_min'] - gdf['c_y_min']) // abs(gdf['a4'])
    gdf['px_y_max'] = (gdf['y_max'] - gdf['c_y_min']) // abs(gdf['a4'])
    
    return gdf

def pix_2_xy(in_bounds, in_affine):
    xmin = in_bounds[0]
    ymin = in_bounds[1]
    xmax = in_bounds[2]
    ymax = in_bounds[3]
    
    xs = (xmin, xmax)
    ys = (ymin, ymax)
    
    pix_coords = rasterio.transform.xy(in_affine, xs, ys)
    
    pix_bounds = (pix_coords[0][0], pix_coords[1][1], pix_coords[0][1], pix_coords[1][0] )
    return pix_bounds

def return_intersection(in_tindex, in_annotations, unique_annotation_id):
    inter = gpd.overlay(in_tindex, in_annotations)
    inter['intersect_area'] = inter['geometry'].area
    #print(f"length of intersection: {len(inter)}")
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #print(inter[['filename', 'unique_pt_id', 'intersect_area']])

    #NOTE: this is where you would modify the code so that the largest intesection is chosen. Later you would need to make sure the
    #anno bounding box is clipped to the image's extent. For more info see the info in the header about edge cases involving anno
    #boxes too big to be 100% contained by an image chip.
    filter_partial_annotations = inter[(inter['intersect_area'] % 1.0) == 0]

    #the script requires a unique anno id to filter down where a single anno is contained in multiple images. This may not be
    #needed if we just take the max intersection values. Prob need to fix this if we're going to generalize to unseen data.
    remove_duplicates = filter_partial_annotations.drop_duplicates(subset=unique_annotation_id)

    return remove_duplicates

def create_cindex(in_file, in_size, in_stride, in_out_dir):
    basename = os.path.splitext(os.path.basename(in_file))[0]
    gdfs = []

    with rasterio.open(in_file, 'r') as src:
        #print(f"Initial Width/Height: {src.width}, {src.height}")
        #print(f"src height/width: {src.height}/{src.width}")
        #print(f"src bounds: {src.bounds}")
        #print(f"src transform: {src.transform}")
        
        upper_left_grid = grid_calc(src.width, src.height, in_stride)
        
        for ul in upper_left_grid:
            #note, we're currently working with slices because I can't make col_off, row_off work. Code needs to be reworked to naturally work with slices
            col_start = ul[0]
            col_stop = ul[0] + in_size
            row_start = ul[1]
            row_stop = ul[1] + in_size
            #slices = (col_start, row_start, col_stop, row_stop)
            colrow_bounds = (col_start, row_start, col_stop, row_stop)
                
            win, win_transform, win_bounds, win_id = build_window(src, ul, in_size, in_stride)

            #NOTE: I had to write my own affine lookup to get bounding boxs from windows (pix_2_coords). Rasterio's windows.bounds(win, win_transform) 
            #caused every overlpping tile to shift 256 pix in the x and y direction (removed overlap, doubled area covered by chip tindex)
            #therefore, the win_bounds variable above should not currently be used. I need to investigate further. I kind of like this better though,
            #because we store everything we need to convert between lat/longs and pixel coordinates in the tile index. It could make it easier to convert
            #our model's output bounding boxes back to lats/longs.

            ret = pix_2_xy(colrow_bounds, src.transform)
            #create and store the chip's geometry (the bounding box of the image chip)
            envelope = geometry.box(*ret)
            geometries=[]
            geometries.append(envelope)
            
            #store the image basename. No real reason, just comes in handy alot.
            attr_basename=[]
            attr_basename.append(in_file)
            
            #store chip name as an attribute in the cindex attribute table
            chip_name = f"{basename}_{win_id}"
            attr_filename = []
            attr_filename.append(chip_name)
            
            # store affine values as attributes in the cindex attribute table
            px_width = []
            row_rot = []
            col_off = []
            col_rot = []
            px_height = []
            row_off = []
            c_x_min = []
            c_y_min = []
            c_x_max = []
            c_y_max = []

            px_width.append(win_transform[0])
            row_rot.append(win_transform[1])
            col_off.append(win_transform[2])
            col_rot.append(win_transform[3])
            px_height.append(win_transform[4])
            row_off.append(win_transform[5])
            c_x_min.append(ret[0])
            c_y_min.append(ret[1])
            c_x_max.append(ret[2])
            c_y_max.append(ret[3])

            #create a single chip feature with attributes and all
            attr_dict = {}
            attr_dict['basename'] = attr_basename
            attr_dict['chip_name'] = attr_filename
            attr_dict['a0'] = px_width
            attr_dict['a1'] = row_rot
            attr_dict['a2'] = col_off
            attr_dict['a3'] = col_rot
            attr_dict['a4'] = px_height
            attr_dict['a5'] = row_off
            attr_dict['c_x_min'] = c_x_min
            attr_dict['c_y_min'] = c_y_min
            attr_dict['c_x_max'] = c_x_max
            attr_dict['c_y_max'] = c_y_max
        
            chip_gdf = make_gdf(geometries, attr_dict, src.crs)
            chip_gdf.head()

            #append our single chip feature to a list of all chips. Later we will merge all the single chips into a big chip index (cindex).
            gdfs.append(chip_gdf)
            
    #merge all those little chip features together into our master cindex for the input image 
    cindex_gdf_path = os.path.join(in_out_dir, f"{basename}_cindex.gpkg")
    cindex_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    cindex_gdf.crs = src.crs
    cindex_gdf.to_file(cindex_gdf_path, driver='GPKG')

    return cindex_gdf

def write_annotations(in_gdf, out_path='none'):
    coord_gdf = coords_2_pix_gdf(in_gdf)
    
    #set our columns/column order
    out_gdf = coord_gdf[['chip_name', 'x_min', 'y_min', 'x_max', 'y_max', 
                         'px_x_min', 'px_y_min', 'px_x_max', 'px_y_max', 
                         'label_name', 'label', 'label_int']]
    
    if out_path != 'none':
        out_gdf.to_csv(out_path, index=False)
    
    return out_gdf

def mask_raster(in_poly, src_raster, in_out_path, in_size):
    try:
        #Note: this is where the 3-band requirement is hard-coded. We could remove this if we were to feed the src.count into this function
        #with rasterio.open(src_raster, 'r', out_shape=(in_size, in_size, 3), resampling=Resampling.bilinear) as src:
            #out_data, out_transform = mask(src, [in_poly], crop=True)
        
        with rasterio.open(src_raster, 'r') as src:
            out_data, out_transform = mask(src, [in_poly], crop=True)

    except:
        print("ERROR 1 in mask_raster:")
        print("Could not read cropped data/transform.")
        sys.exit(0)

    write_jpeg(out_data, src.count, in_size, out_transform, src.crs, in_out_path)

##########################
## 2. BACKBONE FUNCTION ##
##########################

def backbone(args, in_f):
    try:
        #print(f"backbone: {in_f}")
        print(f"backbone: {in_f}")
        #unpack our args list.
        anno_path = args[0] 
        in_anno = gpd.read_file(anno_path)
        
        size = args[1]
        stride = args[2]
        out_dir = args[3]
        logging.info(f'size: {args[1]}, stride: {args[2]}, out_dir: {args[3]}')
        #print(f"{in_anno}, {size}, {stride}, {out_dir}")
    except:
        print("Error loading arguments into backbone. Check your args.")

    try:
        logging.info(f"cindex: {in_f}")
        #Chip out our image, return a cindex
        cindex = create_cindex(in_f, size, stride, out_dir)
    except:
        print("Error in cindex operation!")

    try:
        logging.info(f"intersect: {in_f}")
        #find all the annotations that intersect each chip. Filter chips with no annotations, filter annotations that are not fully contained within a chip.
        intersect = return_intersection(cindex, in_anno, 'unique_pt_id')
    except:
        print("Error in intersect operation!")

    try:
        logging.info(f"writing positive files: {in_f}")
        #generate a list of positive chips in the annotation database.
        pos_chips = intersect['chip_name'].unique().tolist()
        
        pos_chips_gdf = cindex[cindex['chip_name'].isin(pos_chips)]

        for i, row in pos_chips_gdf[['geometry', 'basename','chip_name']].iterrows():
            polygon = row["geometry"]
            src_raster = row['basename']
            out_raster_path = os.path.join(out_dir, f"{row['chip_name']}.jpg")

            #this also write our positive image chip to a jpeg located at out_raster_path
            mask_raster(polygon, src_raster, out_raster_path, size, )
    except:
        print("Error when writing images!")
        print(f"polygon: {polygon}")
        print(f"src_raster: {src_raster}")
        print(f"out_raster_path: {out_raster_path}")

    logging.info(f"backbone COMPLETE: {in_f}")

    #return our annotations to be bound into a island-wide annotation data set.
    return intersect


#################################
## 3. PARSE ARGS & LOGGING  ##
#################################

if __name__ == "__main__":
    #handle our input arguments
    parser = optparse.OptionParser()

    parser.add_option('-f', '--filelist',
        action="store", dest="file_list",
        type='string', help="A txt list of absolute file paths, one file per line. Must be tifs.")

    parser.add_option('-o', '--outdir',
        action='store', dest='usr_out_dir',
        type='string', help='An out directory to stash files. Images, tile indexes, logfiles, etc.')

    parser.add_option('-t', '--chipsize',
        action='store', dest='usr_size',
        type='int', default=512,
        help='Size of image chips in pixel (ie. 512 is 512x512px chips.)')

    parser.add_option('-s', '--stride',
        action='store', dest='usr_stride',
        type='int', default=256,
        help='Amount of image chip overlap in pixels (ie 256 is 50 percent overlap with a chipsize of 512)')

    parser.add_option('-a', '--annotations',
        action='store', dest='in_annotation',
        type='string',
        help='path to a geopackage containing marine debris annotation envelopes. Note: this probably wont work with all annotations. Use preapproved annos for now.')

    options, args = parser.parse_args()

    #setup logging
    log_name = r'log.txt'
    log_name = os.path.join(options.usr_out_dir, log_name)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", 
                        datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(log_name)])

    #do some checks to make sure we can find inputs and outputs.
    if os.path.exists(options.usr_out_dir):
        pass
    else:
        print('ERROR: Cannot find out directory. Abort.')
        logging.error('ERROR: Cannot find out directory. Abort.')
        sys.exit(0)

    if os.path.exists(options.file_list):
        pass
    else:
        print('ERROR: Cannot find input filelist. Abort.')
        logging.error('ERROR: Cannot find input filelist. Abort.')
        sys.exit(0)

    if os.path.exists(options.in_annotation):
        pass
    else:
        print('ERROR: Cannot find input annotations. Abort.')
        logging.error('ERROR: Cannot find input annotations. Abort.')
        sys.exit(0)

    #open our file list and arguments. Zip them up into a list of arguments and a input file that feeds into multiprocessing's starmap.
    with open(options.file_list, 'r') as f:
        in_paths = [line.strip() for line in f]

    args = [[options.in_annotation, options.usr_size, options.usr_stride, options.usr_out_dir]] * len(in_paths)
    zipped = zip(args, in_paths)

#########################
## 4. Multiprocessing  ##
#########################

    #start the pool. Each entry in results will contain a gdf of all the resulting chips.
    pool=multiprocessing.Pool(processes=8)
    map_results = pool.starmap_async(backbone, zipped, chunksize=1) #chunksize=1 is to make the while loop below display the correct info. I have no idea what it does. Risky.

    while not map_results.ready():
        print(f"retile_for_deeplearning_V2.py | {map_results._number_left} of {len(in_paths)} files remain.") #_number_left is prob wrong way to do this. https://stackoverflow.com/questions/49807345/multiprocessing-pool-mapresult-number-left-not-giving-result-i-would-expect
        time.sleep(5)

    pool.close()
    pool.join()

    results = map_results.get()
    print(f"Writing final annotations.")
    logging.info(f"Writing final annotations.")

    #merge all the pd.Dataframes, convert to gpd.GeoDataFrame
    results_df = pd.concat(results, ignore_index=True)
    results_gdf = gpd.GeoDataFrame(results_df, geometry='geometry')
    
    #write annotations to csv
    out_path = os.path.join(options.usr_out_dir, 'final_annotations.csv')
    write_annotations(results_gdf, out_path)

    print("SUCCESS!")
    logging.info(f"SUCCESS!")