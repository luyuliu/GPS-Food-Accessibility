'''
based on the extracted food-related stops and trips, calculate individuals food beahvior metrics

Input: 
    1. retail info
    2. home location info
    3. food-related stops
    4. food-related trips
    
Output: 
    line plots (curves) of
    1. daily number of stops
    2. day of week pattern
    3. time of day pattern
    
'''

import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os



rad = 200

# types of stores
type_1 = 'Large Groceries'
type_2 = 'Big Box Stores'
type_3 = 'Small Healthy Outlets'
type_4 = 'Processed Food Outlets'
the_index = [type_1,type_2,type_3,type_4]


# read retailer dataset
df_store = geopandas.read_file('retail.shp')

# read food stops
retail_folder = 'retail_folder'
food_sp_df_path = os.path.join(retail_folder, rad, f'food_sp_{rad}_limit2h_merged.csv')
food_related_stop_df = pd.read_csv(food_sp_df_path)
food_related_stop_df['started_at'] = pd.to_datetime(food_related_stop_df['started_at']).dt.tz_localize(None)

# read user home
user_table_df = pd.read_csv('home_location.csv')
user_table_df = user_table_df[['user_id','LAT-4326','LON-4326','GEOID20','LAT3857','LON3857']]
user_num = user_table_df.shape[0]


# Perform the stop-store-user join based on user_id, store_id
# join store data, only keep the key columns
merged_df = pd.merge(food_related_stop_df, df_store, left_on='retail_id',right_on='ObjectId')
merged_df = merged_df[['user_id','lat','lon','retail_id','retail_lat','retail_lon','TYPE','SUBTYPE','started_at','finished_at']]
# join user data
merged_df["user_id"] = merged_df["user_id"].astype(str) 
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_df = pd.merge(merged_df, user_table_df, left_on='user_id',right_on='user_id', how='inner') 
# format the merged df
merged_df['started_at_tm'] = pd.to_datetime(merged_df['started_at'])
merged_df['started_at_hour'] = merged_df['started_at_tm'].dt.hour
merged_df['started_at_date'] = merged_df['started_at_tm'].dt.date


# distance calculation function. Distance in metres
import math
R = 6371e3
def calculate_bearing(homeLatitude, homeLongitude, destinationLatitude, destinationLongitude):
    rlat1   =   homeLatitude * (math.pi/180) 
    rlat2   =   destinationLatitude * (math.pi/180) 

    dlat    =   (destinationLatitude - homeLatitude) * (math.pi/180)
    dlon    =   (destinationLongitude - homeLongitude) * (math.pi/180)

    a = (math.sin(dlat/2) * math.sin(dlat/2)) + (math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon/2) * math.sin(dlon/2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c 

    return distance

# calculate home to store distance
merged_df['home_retail_dis'] = merged_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['retail_lat'], row['retail_lon']), axis=1)

# store the calculated data
merged_df.to_csv('food_sp_{rad}_limit2h_with_store_user.csv', index=False)






# group by user, and calculate individual metrics
grouped_data = merged_df.groupby('user_id')  #"user_id"

# initialize the result df
new_df = pd.DataFrame()


# iterate over users, for each metrics


# metric 1: number of visits
for user_id, group in grouped_data:
    user_data = {
        'user_id': user_id,
        'total_count': group['id'].count(),
        'type1_count': group[group['TYPE'] == 1]['id'].count(),
        'type2_count': group[group['TYPE'] == 2]['id'].count(),
        'type3_count': group[group['TYPE'] == 3]['id'].count(),
        'type4_count': group[group['TYPE'] == 4]['id'].count()
    }
    
    new_df = new_df.append(user_data, ignore_index=True)

new_df.to_csv('food_sp_{rad}_count_by_user.csv', index=False)
# calculate population metrics
# all visit
total = sum(new_df['total_count'])
ave = total/user_num
# visit by type
stop_by_type = new_df.iloc[:, -4:].sum()
stop_by_type.index = the_index
stop_ave_by_type = stop_by_type/user_num



# metric 2: number of unique visits
for user_id, group in grouped_data:
    user_data = {
        'user_id': user_id,
        'total_uni_count': group['retail_id'].unique().shape[0],
        'type1_uni_count': group[group['TYPE'] == 1]['retail_id'].unique().shape[0],
        'type2_uni_count': group[group['TYPE'] == 2]['retail_id'].unique().shape[0],
        'type3_uni_count': group[group['TYPE'] == 3]['retail_id'].unique().shape[0],
        'type4_uni_count': group[group['TYPE'] == 4]['retail_id'].unique().shape[0]
    }
    
    new_df = new_df.append(user_data, ignore_index=True)

new_df.to_csv('food_sp_{rad}_unique_by_user.csv', index=False)
# calculate population metrics
# all visit
total = sum(new_df['total_uni_count'])
ave = total/user_num
# visit by type
unique_sum_type = new_df.iloc[:, -4:].sum()
unique_sum_type.index = the_index
unique_ave_type = unique_sum_type/user_num




# metric 3: home to store distance
for user_id, group in grouped_data:
    user_data = {
        'user_id': user_id,
        'total_dis_mean': group['home_retail_dis'].mean(),
        'type1_dis_mean': group[group['TYPE'] == 1]['home_retail_dis'].mean(),
        'type2_dis_mean': group[group['TYPE'] == 2]['home_retail_dis'].mean(),
        'type3_dis_mean': group[group['TYPE'] == 3]['home_retail_dis'].mean(),
        'type4_dis_mean': group[group['TYPE'] == 4]['home_retail_dis'].mean()
    }
    
    new_df = new_df.append(user_data, ignore_index=True)
    
new_df.to_csv('food_sp_{rad}_dis_by_user.csv', index=False)
# calculate population metrics
# all visit
total = sum(new_df['total_dis_mean'])
ave = total/user_num
# visit by type
dis_sum_type = new_df.iloc[:, -4:].sum()
dis_sum_type.index = the_index
dis_ave_type = unique_sum_type/user_num


# calculate nearest store distance for each user
import geopandas as gpd

# input
user_gdf = user_table_df
store_gdf = df_store

#initialize the result
nearest_store = []
result_df = gpd.GeoDataFrame(columns=['user_id'])

for index, user in user_gdf.iterrows():
    result_row = {'user_id': user['user_id']}
    # calculate distance for each type
    for store_type in [1,2,3,4]:
        filtered_stores = store_gdf[store_gdf['TYPE'] == store_type]
        nearest_store_id = filtered_stores.distance(user['geometry']).idxmin()
        nearest_store_distance = user['geometry'].distance(filtered_stores.loc[nearest_store_id, 'geometry'])/1000
        # write down the nearest store id and distance        
        result_row[f'ObjectId_{store_type}'] = filtered_stores.loc[nearest_store_id, 'ObjectId']
        result_row[f'distance_{store_type}'] = round(nearest_store_distance, 3)
        
    result_df = result_df.append(result_row, ignore_index=True)

result_df.to_csv('nearest_store_by_user.csv', index=False)

nn_dis_1 = result_df['distance_1'] 
nn_dis_2 = result_df['distance_2'] 
nn_dis_3 = result_df['distance_3'] 
nn_dis_4 = result_df['distance_4'] 





# metric 4: home-based trip proportion

from shapely import wkt

# read food trips and extract start lat and lon
merged_trip_df = pd.read_csv(os.path.join(retail_folder, rad, f'food_trip_{rad}_merged.csv'))
# function for extract trip start location from wkt string
def extract_coordinates(wkt_string):
    point = wkt.loads(wkt_string)
    return point.x, point.y
merged_trip_df[['start_lon', 'start_lat']] = merged_trip_df['trip_start_location'].apply(extract_coordinates).apply(pd.Series)

# merge retailer info
merged_trip_df = pd.merge(merged_trip_df, df_store, left_on='retail_id',right_on='ObjectId')
# merge user info
merged_trip_df["user_id"] = merged_trip_df["user_id"].astype(str)
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_trip_df = pd.merge(merged_trip_df, user_table_df, left_on='user_id',right_on='user_id', how='inner')

# calculate distance between home and trip start point
merged_trip_df['home_start_dis'] = merged_trip_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['start_lat'], row['start_lon']), axis=1)
merged_trip_df["home_start_dis"] = merged_trip_df["home_start_dis"]/1000
merged_trip_df.to_csv(os.path.join('food_trip_{rad}_merged_joined_home_start.csv'), index=False)

# group by user and calculate the metric
grouped_tp_data = merged_trip_df.groupby('user_id')
new_tp_df = pd.DataFrame()
for user_id, group in grouped_tp_data:
    user_data = {
        'user_id': user_id,
        'total_count' : group['user_id'].count(),
        'total1_count' : group[group['TYPE'] == 1]['user_id'].count(),
        'total2_count' : group[group['TYPE'] == 2]['user_id'].count(),
        'total3_count' : group[group['TYPE'] == 3]['user_id'].count(),
        'total4_count' : group[group['TYPE'] == 4]['user_id'].count(),
        'total_home':  sum(group['home_start_dis'].apply(lambda x: (x < 1))),
        'type1_home': sum(group[group['TYPE'] == 1]['home_start_dis'].apply(lambda x: (x < 0.2))),
        'type2_home': sum(group[group['TYPE'] == 2]['home_start_dis'].apply(lambda x: (x < 0.2))),
        'type3_home': sum(group[group['TYPE'] == 3]['home_start_dis'].apply(lambda x: (x < 0.2))),
        'type4_home': sum(group[group['TYPE'] == 4]['home_start_dis'].apply(lambda x: (x < 0.2)))
    }
    
    new_tp_df = new_tp_df.append(user_data, ignore_index=True)

new_tp_df.to_csv(root + 'food_trip_{rad}_pct_by_user.csv', index=False)

# calculate population metrics
# all visit
new_tp_df['tt_pct'] = new_tp_df['total_home']/new_tp_df['total_count']
new_tp_df['tt_pct'].mean()*100
# visit by type
new_tp_df['type1_pct'] = new_tp_df['type1_home']/new_tp_df['total1_count']
new_tp_df['type2_pct'] = new_tp_df['type2_home']/new_tp_df['total2_count']
new_tp_df['type3_pct'] = new_tp_df['type3_home']/new_tp_df['total3_count']
new_tp_df['type4_pct'] = new_tp_df['type4_home']/new_tp_df['total4_count']
new_tp_df['type1_pct'].mean()*100
new_tp_df['type2_pct'].mean()*100
new_tp_df['type3_pct'].mean()*100
new_tp_df['type4_pct'].mean()*100



