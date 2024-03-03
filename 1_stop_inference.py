"""
adopting the trackintel package (Martin et al. 2023) 
extract staypoints, positionfix linked with the staypoint and triplegs from GPS.

Input: GPS(lat lon timestamp user_id)

Output fields: 
    staypoint: user_id	started_at	finished_at	geom id	
    positionfix: 	user_id	flag	tracked_at	geom	staypoint_id	tripleg_id ID	
    tripleg: user_id	started_at	finished_at	geom id	
"""


import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import trackintel as ti

# read GPS from csv file
pfs = ti.io.file.read_positionfixes_csv('GPS.csv', index_col = 'ID')

# generate staypoints and modified pfs
pfs, sp = pfs.as_positionfixes.generate_staypoints(method='sliding', distance_metric='haversine', dist_threshold=100, time_threshold=5.0, gap_threshold=720.0, n_jobs=-1)
print(len(sp['user_id'].unique()))

# generate triplegs and modified pfs
pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp, method='between_staypoints', gap_threshold = 60)

# Write positionfixes to csv file.
ti.io.file.write_positionfixes_csv(pfs, 'pfs.csv')

# Write staypoints to csv file.
ti.io.file.write_staypoints_csv(sp, 'staypoints.csv')

# Write triplegs to csv file.
ti.io.file.write_triplegs_csv(tpls, 'triplegs.csv')