"""
based on the extracted staypoints and food retailer location, 
1. extract food related stops based on distance threshold
    Input:
        stop points: id	user_id	started_at	finished_at	geom
        retail locations: ObjectId	Store_Name	Longitude	Latitude
        threshold: in meter
    Output:
        id	user_id	started_at	finished_at	lat	lon	retail_id	retail_lat	retail_lon
2. split the stops by users
3. filter the stops, limit 2h visit duration


"""



import os
os.environ['USE_PYGEOS'] = '0'
import geopandas
import pandas as pd
from shapely.geometry import Point
import warnings
from shapely.wkt import loads
import trackintel as ti
from tqdm import tqdm
import numpy as np

# read the input
stop_point_df = pd.read_csv('staypoints.csv')
retail_pd = pd.read_csv('retail.csv')


# function for selecting food-related stops within certain radius
def is_close(dist, latA1, lonA2, latB1, lonB2):
    # Create shapely Point objects from the locations
    point1 = Point(lonA2, latA1)
    point2 = Point(lonB2, latB1)
    # Calculate distance between points in meters
    distance = point1.distance(point2) * 111319.9
    # Return True if distance is less than or equal to dist meters
    return distance <= dist

radius = 200
# iterate over the points, call the function
count = 0
warnings.simplefilter(action='ignore', category=Warning)
data_to_append_list = []
for sp in tqdm(stop_point_df.itertuples(), total=len(stop_point_df)):
    p = loads(sp[5])
    lat = p.y
    lon = p.x
    # iterate over retailers
    for row in retail_pd.itertuples():
        if is_close(radius, float(row[12]), float(row[11]), lat, lon):
            count += 1
            data_to_append = {'id': sp[1], 'user_id': sp[2], 'started_at': sp[3], 'finished_at': sp[4], 'lat': lat, 'lon': lon, 'retail_id': row[1], 
                                'retail_lat': float(row[12]), 'retail_lon': float(row[11])}
            data_to_append_list.append(data_to_append)
            
            break

new_df = pd.DataFrame.from_records(data_to_append_list)
new_df.to_csv(f'retail_{radius}_food_sp.csv')





# split the data by users, for later trip extraction
retail_folder = 'retail_folder'
food_sp_df = pd.read_csv(f'retail_{radius}.csv')
for user_id in food_sp_df['user_id'].unique():
    user_data = food_sp_df[food_sp_df['user_id'] == user_id]
    user_data.to_csv(os.path.join(retail_folder, radius, f'user_{user_id}_food_sp_{radius}.csv'), index=False)   
    



# filter, keep trips duration < 2 hour
from datetime import timedelta
# get the file list 
files = os.listdir(os.path.join(retail_folder, radius))
# Iterate through files of each user
for file in tqdm(files, desc='Processing Files', unit='file'):
    # read the file
    user_id = int(file.split('_')[1])
    food_sp_file = f'user_{user_id}_food_sp.csv'
    food_sp_df_path = os.path.join(retail_folder, radius, food_sp_file)
    food_sp_df = pd.read_csv(food_sp_df_path)
    # format the file
    food_sp_df['started_at_tm'] = pd.to_datetime(food_sp_df['started_at']).dt.tz_localize(None) + timedelta(hours=-4)
    food_sp_df['finished_at_tm'] = pd.to_datetime(food_sp_df['finished_at']).dt.tz_localize(None) + timedelta(hours=-4)
    # filter the duration criteria
    food_sp_df['duration_h'] = (food_sp_df['finished_at_tm']-food_sp_df['started_at_tm']) / np.timedelta64(1, 'h')
    food_sp_df = food_sp_df[(food_sp_df['duration_h']<=2)]
    
    if len(food_sp_df)>0:
        # each user one file
        output_file_path = os.path.join(retail_folder, radius, f'user_{user_id}_food_sp_{radius}_limit2h.csv')
        food_sp_df.to_csv(output_file_path, index=False)
        # combined file
        output_file_path_merge = os.path.join(retail_folder, radius, f'food_sp_limit2h_{radius}_merged.csv')
        food_sp_df.to_csv(output_file_path_merge, mode='a', header=not os.path.exists(output_file_path_merge), index=False)

