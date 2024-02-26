
import os
os.environ['USE_PYGEOS'] = '0'
import trackintel as ti
import pandas as pd
import pyproj
import numpy as np
import warnings
from shapely.wkt import loads
import datetime
from shapely import wkt
from shapely.geometry import Point
import sys


input_pfs_folder = input_pfs_folder
input_sp_folder = input_sp_folder
output_folder = output_folder

for i in range(1, 7):
    pfs_name = f'pfs{i}.csv'
    pfs_full_path = os.path.join(input_pfs_folder, pfs_name)
    sp_name = f'sp_{i}.csv'
    sp_full_path = os.path.join(input_sp_folder, sp_name)
    
    #df = pd.read_csv(file_full_path, index_col=None)
    
    # ## Stop Inference
    pfs = ti.io.file.read_positionfixes_csv(pfs_full_path, 
                                            columns={'user_id':'user_id', 'latitude':'latitude', 'longitude':'longitude', 'time':'tracked_at'},
                                            tz= 'America/New_York',
                                            crs ='EPSG:4326',
                                            index_col=None)
    
    sp = ti.io.file.read_staypoints_csv(sp_full_path, 
                                            columns={'user_id':'user_id', 'started_at':'started_at', 'finished_at':'finished_at', 'geom':'geom','id':'id'},
                                            tz= 'America/New_York',
                                            crs ='EPSG:4326',
                                            index_col=None)
    # generate staypoints
    #warnings.filterwarnings("ignore", message="The shapely GEOS version .* is incompatible with the PyGEOS GEOS version .*")
    
    # pfs, sp = pfs.as_positionfixes.generate_staypoints(method='sliding', distance_metric='haversine', dist_threshold=100, time_threshold=5.0, gap_threshold=720.0)
    
    
    # ## Trip Inference
    
    warnings.filterwarnings("ignore", message="The positionfixes with ids .* lead to invalid tripleg geometries.*")
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method='between_staypoints', gap_threshold = 60)
    
    from shapely.errors import ShapelyError
    try:
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method='between_staypoints', gap_threshold = 60)
    except ShapelyError as e:
        print(f"Ignoring GEOSException: {e}")
        pass

    
    
    
    # # Write positionfixes to csv file.
    # ti.io.file.write_positionfixes_csv(pfs, 'pfs.csv')

    # # Write staypoints to csv file.
    # ti.io.file.write_staypoints_csv(sp, 'staypoints.csv')
    ti.io.file.write_positionfixes_csv(pfs, os.path.join(input_pfs_folder, f'pfs_{i}_test.csv'))
    # Write triplegs to csv file.
    ti.io.file.write_triplegs_csv(tpls, os.path.join(output_folder,'Jackson_tripleg', f'tp_{i}.csv'))
