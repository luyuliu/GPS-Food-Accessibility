

import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt

root = "D:/food_related_trips/results/"



type_1 = 'Large Groceries'
type_2 = 'Big Box Stores'
type_3 = 'Small Healthy Outlets'
type_4 = 'Processed Food Outlets'


rad = "50" #50,100,150,200

food_related_stop_df_all = pd.read_csv(root + 'retail_' + rad + '_merged.csv') #total stops
user_table_df = df_user_home_wm.copy()
# subset_columns = ['user_id','LAT-4326','LON-4326','GEOID20','LAT3857','LON3857']
# user_table_df = df_user_home_wm.loc[:, subset_columns]

food_related_stop_df_all['started_at'] = pd.to_datetime(food_related_stop_df_all['started_at']).dt.tz_localize(None)

# start_date = pd.to_datetime('2022-09-01 00:00:00')
# end_date = pd.to_datetime('2022-10-15 23:59:59')
#food_related_stop_df_all = food_related_stop_df_all[(food_related_stop_df_all['started_at'] >= start_date) & (food_related_stop_df_all['started_at'] <= end_date)]


# Perform the stop-store-user join based on ID
merged_df = pd.merge(food_related_stop_df_all, df_store_wm, left_on='retail_id',right_on='ObjectId')

merged_df = merged_df[['user_id','lat','lon','retail_id','retail_lat','retail_lon','TYPE','SUBTYPE','started_at','finished_at']]
user_table_df = user_table_df[['user_id','LAT-4326','LON-4326','GEOID20','LAT3857','LON3857']]
merged_df["user_id"] = merged_df["user_id"].astype(str) 
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_df = pd.merge(merged_df, user_table_df, left_on='user_id',right_on='user_id', how='inner') 
#merged_df.to_csv(root + 'retail_' + rad + '_stop_joined_raw.csv', index=False)





import math
R = 6371e3
def calculate_bearing(homeLatitude, homeLongitude, destinationLatitude, destinationLongitude):
    rlat1   =   homeLatitude * (math.pi/180) 
    rlat2   =   destinationLatitude * (math.pi/180) 
    #rlon1   =   homeLongitude * (math.pi/180) 
    #rlon2   =   destinationLongitude * (math.pi/180)
    dlat    =   (destinationLatitude - homeLatitude) * (math.pi/180)
    dlon    =   (destinationLongitude - homeLongitude) * (math.pi/180)
    # Formula for bearing
    #y = math.sin(rlon2 - rlon1) * math.cos(rlat2)
    #x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(rlon2 - rlon1)
    # Haversine formula to find distance
    a = (math.sin(dlat/2) * math.sin(dlat/2)) + (math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon/2) * math.sin(dlon/2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Distance in metres
    distance = R * c 
    # Bearing in radians
    #bearing = math.atan2(y, x)
    #bearingDegrees = bearing * (180/math.pi)

    return distance





rad = "50" #50,100,150,200
merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_raw.csv')

merged_df["user_id"] = merged_df["user_id"].astype(str)
merged_df['id'] = range(1, len(merged_df) + 1)
merged_df['started_at_tm'] = pd.to_datetime(merged_df['started_at'])
merged_df['started_at_hour'] = merged_df['started_at_tm'].dt.hour
merged_df['started_at_date'] = merged_df['started_at_tm'].dt.date
merged_df['home_retail_dis'] = merged_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['retail_lat'], row['retail_lon']), axis=1)

# merged_df.to_csv(root + 'retail_' + rad + '_stop_joined_with_dis.csv', index=False)









rad = "50"

#merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_raw.csv')
merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_with_dis.csv')
new_type_short = df_store_wm[['ObjectId','TYPE']]
merged_df = pd.merge(merged_df, df_store_wm, left_on='retail_id',right_on='ObjectId')
#merged_df.to_csv(root + 'retail_' + rad + '_stop_joined_with_dis_with_type.csv', index=False)

merged_df["user_id"] = merged_df["user_id"].astype(str)
merged_df['id'] = range(1, len(merged_df) + 1)
merged_df['started_at_tm'] = pd.to_datetime(merged_df['started_at'])
merged_df['started_at_hour'] = merged_df['started_at_tm'].dt.hour
merged_df['started_at_date'] = merged_df['started_at_tm'].dt.date


# user_in["user_id"] = user_in["user_id"].astype(str)
# user_out["user_id"] = user_out["user_id"].astype(str)

# trips_by_desert = pd.merge(merged_df, user_in, left_on='user_id',right_on='user_id', how='inner') #375631 = + 
# trips_by_undesert = pd.merge(merged_df, user_out, left_on='user_id',right_on='user_id', how='inner') #299281 = + 

# merged_df_1 = merged_df[merged_df['new_type'] == 1]
# merged_df_2 = merged_df[merged_df['new_type'] == 2]


#df_store_wm_filtered = df_store_wm#[df_store_wm['new_type'] == 2]

food_related_stop_df = merged_df.copy()

food_related_stop_df = food_related_stop_df#[food_related_stop_df['new_type'] == 2]

grouped_data = food_related_stop_df.groupby('user_id')  #"user_id"

new_df = pd.DataFrame()





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

new_df.to_csv(root + 'retail_' + rad + '_stop_count_by_user_new.csv', index=False)






rad = "50"

by_user_all_50 = pd.read_csv(root + 'retail_' + rad + '_stop_count_by_user_1007.csv')
total_50 = sum(by_user_all_50['total_count'])
by_user_all_50["user_id"] = by_user_all_50["user_id"].astype(str)

# by_user_in_50 = pd.merge(user_in_short, by_user_all_50, on='user_id', how='left')
# by_user_in_50.fillna(0, inplace=True)
# by_user_out_50 = pd.merge(user_out_short, by_user_all_50, on='user_id', how='left')
# by_user_out_50.fillna(0, inplace=True)

# total_by_out_50 = sum(by_user_out_50['total_count'])
# total_by_in_50 = sum(by_user_in_50['total_count'])

ave_50 = total_50/user_num
# ave_50_in = total_by_in_50/user_in_num
# ave_50_out = total_by_out_50/user_out_num

stop_sum_50 = by_user_all_50.iloc[:, -5:].sum()
stop_sum_50.index = the_index
stop_ave_50 = stop_sum_50/user_num
stop_ave_50

# stop_sum_50_in = by_user_in_50.iloc[:, -5:].sum()
# stop_sum_50_in.index = the_index
# stop_ave_50_in = stop_sum_50_in/user_in_num
# stop_ave_50_in

# stop_sum_50_out = by_user_out_50.iloc[:, -5:].sum()
# stop_sum_50_out.index = the_index
# stop_ave_50_out = stop_sum_50_out/user_out_num
# stop_ave_50_out

























rad = "50" #50,100,150,200

merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_with_dis_with_type.csv')
# merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_raw.csv')
# merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_with_dis.csv')
# new_type_short = df_store_wm[['ObjectId','TYPE']]
# merged_df = pd.merge(merged_df, df_store_wm, left_on='retail_id',right_on='ObjectId')


merged_df["user_id"] = merged_df["user_id"].astype(str)
merged_df['id'] = range(1, len(merged_df) + 1)
merged_df['started_at_tm'] = pd.to_datetime(merged_df['started_at'])
merged_df['started_at_hour'] = merged_df['started_at_tm'].dt.hour
merged_df['started_at_date'] = merged_df['started_at_tm'].dt.date

food_related_stop_df = merged_df.copy()

grouped_data = food_related_stop_df.groupby('user_id')  #"user_id"


new_df = pd.DataFrame()
for user_id, group in grouped_data:
    user_data = {
        'user_id': user_id,
        'total_count': group['retail_id'].unique().shape[0],
        'type1_count': group[group['TYPE'] == 1]['retail_id'].unique().shape[0],
        'type2_count': group[group['TYPE'] == 2]['retail_id'].unique().shape[0],
        'type3_count': group[group['TYPE'] == 3]['retail_id'].unique().shape[0],
        'type4_count': group[group['TYPE'] == 4]['retail_id'].unique().shape[0]
    }
    
    new_df = new_df.append(user_data, ignore_index=True)

new_df.to_csv(root + 'retail_' + rad + '_unique_by_user_new.csv', index=False)





rad = "50"

by_user_all_50 = pd.read_csv(root + 'retail_' + rad + '_unique_by_user_1007.csv')
by_user_all_50["user_id"] = by_user_all_50["user_id"].astype(str)

# by_user_in_50 = pd.merge(user_in_short, by_user_all_50, on='user_id', how='left')
# by_user_in_50.fillna(0, inplace=True)
# by_user_out_50 = pd.merge(user_out_short, by_user_all_50, on='user_id', how='left')
# by_user_out_50.fillna(0, inplace=True)


unique_sum_50 = by_user_all_50.iloc[:, -5:].sum()
unique_sum_50.index = the_index
unique_ave_50 = unique_sum_50/user_num
unique_ave_50

# unique_sum_50_in = by_user_in_50.iloc[:, -5:].sum()
# unique_sum_50_in.index = the_index
# unique_ave_50_in = unique_sum_50_in/user_in_num
# unique_ave_50_in

# unique_sum_50_out = by_user_out_50.iloc[:, -5:].sum()
# unique_sum_50_out.index = the_index
# unique_ave_50_out = unique_sum_50_out/user_out_num
# unique_ave_50_out










rad = "50" #50,100,150,200

merged_df = pd.read_csv(root + 'retail_' + rad + '_stop_joined_with_dis_with_type.csv')

food_related_stop_df = merged_df.copy()

trip_store_dis_df = food_related_stop_df.copy()
trip_store_dis_df["home_retail_dis"] = trip_store_dis_df["home_retail_dis"]/1000
grouped_data = trip_store_dis_df.groupby('user_id')

new_df = pd.DataFrame()
for user_id, group in grouped_data:
    user_data = {
        'user_id': user_id,
        'total_mean': group['home_retail_dis'].mean(),
        'type1_mean': group[group['TYPE'] == 1]['home_retail_dis'].mean(),
        'type2_mean': group[group['TYPE'] == 2]['home_retail_dis'].mean(),
        'type3_mean': group[group['TYPE'] == 3]['home_retail_dis'].mean(),
        'type4_mean': group[group['TYPE'] == 4]['home_retail_dis'].mean()
    }
    
    new_df = new_df.append(user_data, ignore_index=True)

new_df.to_csv(root + 'retail_' + rad + '_dist_by_user_new.csv', index=False)










rad = "50"

by_user_all_50 = pd.read_csv(root + 'retail_' + rad + '_dist_by_user_1007.csv')
by_user_all_50["user_id"] = by_user_all_50["user_id"].astype(float).astype(int).astype(str)

# by_user_in_50 = pd.merge(user_in_short, by_user_all_50, on='user_id', how='left')
# #by_user_in_50.fillna(0, inplace=True)
# by_user_out_50 = pd.merge(user_out_short, by_user_all_50, on='user_id', how='left')
# #by_user_out_50.fillna(0, inplace=True)


dis_mean_50 = by_user_all_50.iloc[:, -5:].mean()
dis_mean_50

# dis_mean_50_in = by_user_in_50.iloc[:, -5:].mean()
# dis_mean_50_in

# dis_mean_50_out = by_user_out_50.iloc[:, -5:].mean()
# dis_mean_50_out















# nearest distance

import geopandas as gpd

user_gdf = user_in_duval
store_gdf = df_store_wm

nearest_store = []
result_df = gpd.GeoDataFrame(columns=['user_id'])

for index, user in user_gdf.iterrows():
    
    result_row = {'user_id': user['user_id']}
    
    for store_type in [1,2,3,4]:
        filtered_stores = store_gdf[store_gdf['TYPE'] == store_type]
        nearest_store_id = filtered_stores.distance(user['geometry']).idxmin()
        nearest_store_distance = user['geometry'].distance(filtered_stores.loc[nearest_store_id, 'geometry'])/1000
                
        result_row[f'ObjectId_{store_type}'] = filtered_stores.loc[nearest_store_id, 'ObjectId']
        result_row[f'distance_{store_type}'] = round(nearest_store_distance, 3)
        
    result_df = result_df.append(result_row, ignore_index=True)

#result_df.to_csv(root + 'nearest_store_results_new_type.csv', index=False)



result_df = pd.read_csv(root + 'nearest_store_results_new_type_1007.csv')


his_1 = result_df['distance_1'] 
his_2 = result_df['distance_2'] 
his_3 = result_df['distance_3'] 
his_4 = result_df['distance_4'] 




plt.rcParams["figure.figsize"] = (6,5)
import seaborn as sns
sns.set_style('darkgrid')
sns.distplot(his_1,label='type1',hist=False)
sns.distplot(his_2,label='type2',hist=False)
sns.distplot(his_3,label='type3',hist=False)
sns.distplot(his_4,label='type4',hist=False)


plt.title('histogram of home to nearest-store distance (km)', fontsize=15)
plt.xlabel("distance (km)")
plt.xlim(0, 30)
plt.legend()
plt.show()





# nearest dist and store dist
hts_50_dis = pd.read_csv(root + 'retail_50_dist_by_user_1007.csv')
hts_100_dis = pd.read_csv(root + 'retail_100_dist_by_user_1007.csv')
hts_150_dis = pd.read_csv(root + 'retail_150_dist_by_user_1007.csv')
hts_200_dis = pd.read_csv(root + 'retail_200_dist_by_user_1007.csv')

his_50_1 = hts_50_dis['type1_mean'] 
his_100_1 = hts_100_dis['type1_mean'] 
his_150_1 = hts_150_dis['type1_mean'] 
his_200_1 = hts_200_dis['type1_mean'] 

his_50_2 = hts_50_dis['type2_mean'] 
his_100_2 = hts_100_dis['type2_mean'] 
his_150_2 = hts_150_dis['type2_mean'] 
his_200_2 = hts_200_dis['type2_mean'] 

his_50_3 = hts_50_dis['type3_mean'] 
his_100_3 = hts_100_dis['type3_mean'] 
his_150_3 = hts_150_dis['type3_mean'] 
his_200_3 = hts_200_dis['type3_mean'] 

his_50_4 = hts_50_dis['type4_mean']   
his_100_4 = hts_100_dis['type4_mean'] 
his_150_4 = hts_150_dis['type4_mean'] 
his_200_4 = hts_200_dis['type4_mean'] 




import seaborn as sns
sns.set_style('darkgrid')

plt.rcParams["figure.figsize"] = (6,5)

sns.distplot(his_150_1,label='visited_type1',hist=True)
sns.distplot(his_200_2,label='visited_type2',hist=True)
sns.distplot(his_50_3,label='visited_type3',hist=True)
sns.distplot(his_50_4,label='visited_type4',hist=True)

plt.title('histogram of home to visited-store distance (km)', fontsize=15)

plt.xlabel("distance (km)")

plt.xlim(0, 30)
plt.ylim(0, 0.12)
plt.legend()
plt.show()











import seaborn as sns
sns.set_style('darkgrid')

sns.distplot(his_200_1,label='visited_type1',hist=False,color='blue')
sns.distplot(his_1,label='nearest_type1',hist=False,color='blue', kde_kws={'linestyle':'--'})

plt.title('nearest v.s. visited home-store distance. type1 (km)', fontsize=15)

plt.xlim(0, 30)
plt.ylim(0, 0.4)
plt.legend()
plt.show()







# home-based





import os

def is_file_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        return not content
    

folder_path = 'D:/Dropbox/Dropbox (UFL)/GPS data travel survey/Data&Results/Jacksonville/Jackson_tour/by_user_limit2h/retail_100'
result_path = 'D:/food_related_trips/results'

files = os.listdir(os.path.join(folder_path))

merged_data = pd.DataFrame()

for file_name in files:
    file_path = os.path.join(folder_path, file_name)

    # Check if the file is not empty
    if os.path.isfile(file_path) and not is_file_empty(file_path):
        # Read the CSV file
        current_data = pd.read_csv(file_path)

        # Merge with the existing data
        merged_data = pd.concat([merged_data, current_data], ignore_index=True)

# Save the merged data to a new CSV file
merged_data.to_csv(os.path.join(result_path,'food_tour_100_merged_limit2h.csv'), index=False)






csv_trip_all = pd.read_csv(os.path.join(result_path,'food_trip_4_type_merged_limit2h.csv'))
csv_trip_all['id'] = range(1, len(csv_trip_all) + 1)

from shapely import wkt
csv_trip_all['trip_start_geom'] = csv_trip_all['trip_start_location'].apply(wkt.loads)
food_related_trip_df_all = geopandas.GeoDataFrame(csv_trip_all, geometry=csv_trip_all['trip_start_geom'])
food_related_trip_df_all.crs = 'EPSG:4326'


food_related_trip_df_all['trip_start_timestamp'] = pd.to_datetime(food_related_trip_df_all['trip_start_timestamp']).dt.tz_localize(None)

start_date = pd.to_datetime('2022-09-01 00:00:00')
end_date = pd.to_datetime('2022-10-15 23:59:59')
food_related_trip_df_all = food_related_trip_df_all[(food_related_trip_df_all['trip_start_timestamp'] >= start_date) & (food_related_trip_df_all['trip_start_timestamp'] <= end_date)]



food_related_trip_df_all['id'].size # 1062286
food_related_trip_df_all['user_id'].unique().size # 144197
#food_related_trip_df_all['user_id'] = food_related_trip_df_all['deviceID']

food_related_trip_df_all['start_lon'] = food_related_trip_df_all['geometry'].x
food_related_trip_df_all['start_lat'] = food_related_trip_df_all['geometry'].y

food_related_trip_df_all_wm = food_related_trip_df_all.to_crs(epsg=3857)





df_store_wm_filtered = df_store_wm#[df_store_wm['TYPE'] == 2]
user_table_df = df_user_home_all_wm


# Perform the home_store join based on ID
merged_trip_df = pd.merge(food_related_trip_df_all_wm, df_store_wm_filtered, left_on='retail_id',right_on='ObjectId')
merged_trip_df["user_id"] = merged_trip_df["user_id"].astype(str)
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_trip_df = pd.merge(merged_trip_df, user_table_df, left_on='user_id',right_on='user_id', how='inner')
#merged_trip_df = merged_trip_df[['id','user_id','start_lon','start_lat','retail_id','Longitude','Latitude','TYPE','SUBTYPE','trip_start_timestamp','trip_end_timestamp','LAT-4326','LON-4326']]

merged_trip_df['home_start_dis'] = merged_trip_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['start_lat'], row['start_lon']), axis=1)

#merged_trip_df.to_csv(os.path.join(result_path,'food_trip_4_type_merged_limit2h_joined_home_start.csv'), index=False)


import seaborn as sns
merged_trip_df_10k = merged_trip_df[merged_trip_df['home_start_dis']<10000]
sns.distplot(merged_trip_df_10k['home_start_dis'],hist=False)



merged_trip_df['trip_start_timestamp'] = pd.to_datetime(merged_trip_df['trip_start_timestamp'])
merged_trip_df['started_at_hour'] = merged_trip_df['trip_start_timestamp'].dt.hour
merged_trip_df['started_at_date'] = merged_trip_df['trip_start_timestamp'].dt.date
#merged_trip_df['duration_h'] = (merged_trip_df['finished_at_tm']-merged_trip_df['started_at_tm']) / np.timedelta64(1, 'h')
#merged_trip_df = merged_trip_df[(merged_trip_df['duration_h']<=2)]

# tour_by_desert = pd.merge(merged_trip_df, user_in, left_on='user_id',right_on='user_id', how='inner')    #13788 = 1461 + 12327
# tour_by_undesert = pd.merge(merged_trip_df, user_out, left_on='user_id',right_on='user_id', how='inner') #10219 = 1465 + 8754
# #tour_by_desert[tour_by_desert['TYPE'] == 1]


# merged_trip_df_1 = merged_trip_df[merged_trip_df['TYPE'] == 1]
# grouped_data_1 = merged_trip_df_1[['trip_start_timestamp','id_x']].groupby(pd.Grouper(key='trip_start_timestamp', freq='D')).count()
# grouped_data_1.index = grouped_data_1.index.date

# merged_trip_df_2 = merged_trip_df[merged_trip_df['TYPE'] == 2]
# grouped_data_2 = merged_trip_df_2[['trip_start_timestamp','id_x']].groupby(pd.Grouper(key='trip_start_timestamp', freq='D')).count()
# grouped_data_2.index = grouped_data_2.index.date

# merged_trip_df_3 = merged_trip_df[merged_trip_df['TYPE'] == 3]
# grouped_data_3 = merged_trip_df_3[['trip_start_timestamp','id_x']].groupby(pd.Grouper(key='trip_start_timestamp', freq='D')).count()
# grouped_data_3.index = grouped_data_3.index.date

# merged_trip_df_4 = merged_trip_df[merged_trip_df['TYPE'] == 4]
# grouped_data_4 = merged_trip_df_4[['trip_start_timestamp','id_x']].groupby(pd.Grouper(key='trip_start_timestamp', freq='D')).count()
# grouped_data_4.index = grouped_data_4.index.date



merged_trip_df_1 = merged_trip_df[merged_trip_df['TYPE_x'] == 1]
merged_trip_df_2 = merged_trip_df[merged_trip_df['TYPE_x'] == 2]
merged_trip_df_3 = merged_trip_df[merged_trip_df['TYPE_x'] == 3]
merged_trip_df_4 = merged_trip_df[merged_trip_df['TYPE_x'] == 4]

grouped_df = merged_trip_df.groupby('user_id')
total_count = grouped_df.size()
criteria_count = grouped_df['home_start_dis'].apply(lambda x: (x < 1000).sum())
# percentage = (criteria_count / total_count) * 100
percentage = sum(criteria_count) / sum(total_count)* 100
average_pct = np.mean(percentage)

grouped_df = merged_trip_df_1.groupby('user_id')
total_count = grouped_df.size()
criteria_count = grouped_df['home_start_dis'].apply(lambda x: (x < 1000).sum())
# percentage = (criteria_count / total_count) * 100
percentage = sum(criteria_count) / sum(total_count)* 100
average_pct_1 = np.mean(percentage)

grouped_df = merged_trip_df_2.groupby('user_id')
total_count = grouped_df.size()
criteria_count = grouped_df['home_start_dis'].apply(lambda x: (x < 1000).sum())
# percentage = (criteria_count / total_count) * 100
percentage = sum(criteria_count) / sum(total_count)* 100
average_pct_2 = np.mean(percentage)

grouped_df = merged_trip_df_3.groupby('user_id')
total_count = grouped_df.size()
criteria_count = grouped_df['home_start_dis'].apply(lambda x: (x < 1000).sum())
# percentage = (criteria_count / total_count) * 100
percentage = sum(criteria_count) / sum(total_count)* 100
average_pct_3 = np.mean(percentage)

grouped_df = merged_trip_df_4.groupby('user_id')
total_count = grouped_df.size()
criteria_count = grouped_df['home_start_dis'].apply(lambda x: (x < 1000).sum())
# percentage = (criteria_count / total_count) * 100
percentage = sum(criteria_count) / sum(total_count)* 100
average_pct_4 = np.mean(percentage)






















new_df = percentage.to_frame()
new_df["user_id"] = new_df.index
new_df = new_df.reset_index(drop=True)

# user join by home location
df_user_home_wm["user_id"] = df_user_home_wm["user_id"].astype(str)
new_df["user_id"] = new_df["user_id"].astype(str)
user_percent_home = df_user_home_wm.merge(new_df, left_on='user_id',right_on='user_id')


# join and aggreg by tract
points = user_percent_home
polygons = df_acs_duval_wm

points_in_polygons = geopandas.sjoin(points, polygons, op='within')

trip_store_ave = points_in_polygons.groupby('GEOID20_left')[0].mean().reset_index(name='percent_median')

polygons_with_ave = polygons.merge(trip_store_ave, left_on='GEOID20', right_on='GEOID20_left', how='left')



fig, ax = plt.subplots(figsize=(16, 8))

ax = polygons_with_ave.plot(ax=ax, column='percent_median', cmap='Greens', linewidth=0, edgecolor='black', legend=True)
ax = df_undesert_wm.plot(ax=ax,facecolor='none',edgecolor='k', linewidth=0.2,legend=True,label='food_undesert')
ax = df_desert_wm.plot(ax=ax,facecolor='none', edgecolor="r", linewidth=0.5,legend=True,label='food_desert')

plt.title('homebased_percent_median (%)', fontsize=15)

ax.axis('off')
# Show the plot
plt.show()




stat_all_0 = new_df[0].mean()


user_in["user_id"] = user_in["user_id"].astype(str)
user_out["user_id"] = user_out["user_id"].astype(str)
trips_by_desert = pd.merge(new_df, user_in, left_on='user_id',right_on='user_id')
trips_by_undesert = pd.merge(new_df, user_out, left_on='user_id',right_on='user_id')

stat_in_0 = trips_by_desert[0].mean()
stat_out_0 = trips_by_undesert[0].mean()
