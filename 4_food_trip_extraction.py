"""
based on 
    1. the extracted triplegs from stop_inference.py, and 
    2. the food stop extracted from food_stop_extraction.py
generate food trips

method: sliding window. 
    we set the staying thresholds to a space of 100-meter radius, and a duration of at least 5 minutes and at most 720 minutes

Input: triplegs, food stops, time window threshold
Output: food trips
    user_id	tripleg_ID	trip_started_at	trip_finished_at	trip 	retail_id	retail_lat	retail_lon

"""


import pandas as pd 
import datetime
from shapely import wkt, Point
from tqdm import tqdm
import os

retail_folder = 'retail_folder'

radius = 200
files = os.listdir(os.path.join(retail_folder, radius))

# Iterate for each user
for file in tqdm(files, desc='Processing Files', unit='file'):
    
    user_id = int(file.split('_')[1])
    
    # read the food stop file
    food_sp_file = f'user_{user_id}_food_sp_{radius}_limit2h.csv'
    food_sp_df_path = os.path.join(retail_folder, radius, food_sp_file)
    food_sp_df = pd.read_csv(food_sp_df_path)
    food_sp_df_path['started_at'] = pd.to_datetime(food_sp_df_path['started_at'])
    
    # read the triplegs
    tripleg = pd.read_csv('triplegs.csv')
    
    # users with food visit
    ID_list=food_sp_df['user_id'].unique()
    len(ID_list)
    
    # trip forming, using sliding window method to connect stop points
    time_window = datetime.timedelta(minutes=15)
    miss = 0
    check_len = 0
    data_to_append_list = []
    
    # iterate over users
    for id in tqdm(ID_list):
        # extract food stop of the user
        stop_by_user = food_sp_df[food_sp_df['user_id'] == id]
        # extract tripleg of the user
        related_tripleg_by_user = tripleg[tripleg['user_id'] == id]
        if len(related_tripleg_by_user) == 0: 
            miss += 1
            continue
        # slide the window
        for stop_row in stop_by_user.itertuples():
            tripleg_found = related_tripleg_by_user[
                (related_tripleg_by_user['finished_at'] > stop_row[4] - time_window) &
                (related_tripleg_by_user['finished_at'] <= stop_row[4])
            ]
            #did not found related trip
            if len(tripleg_found) == 0: continue
            #check if there if repeat data
            if len(tripleg_found) > 1: 
                check_len += 1
                tripleg_found = tripleg_found.nlargest(1, 'finished_at')
            
            '''
                The end location and time is start time of next stop point.
            '''
            # prepare the data for output
            data_to_append = {'user_id': id, 'tripleg_ID': tripleg_found.iloc[0]['id'], 
                              'trip_started_at': tripleg_found.iloc[0]['started_at'],
                              'trip_finished_at': tripleg_found.iloc[0]['finished_at'],
                              'trip':tripleg_found.iloc[0]['geom'],
                              'retail_id':tripleg_found.iloc[0]['retail_id']}
            
            data_to_append_list.append(data_to_append)
             
    # write to file
    food_trip_df = pd.DataFrame(data_to_append_list, columns=['user_id', 'tripleg_ID', 'trip_started_at', 'trip_finished_at', 'trip','retail_id'])
    # each user one file
    output_file_path = os.path.join(retail_folder, radius, f'user_{user_id}_food_trip_{radius}.csv')
    food_trip_df.to_csv(output_file_path, index=False)
    # merged file
    output_file_path_merge = os.path.join(retail_folder, radius, f'food_trip_{radius}_merged.csv')
    food_trip_df.to_csv(output_file_path_merge, mode='a', header=not os.path.exists(output_file_path_merge), index=False)





