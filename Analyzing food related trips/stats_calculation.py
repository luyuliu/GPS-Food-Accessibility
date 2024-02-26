


from shapely import wkt

def extract_coordinates(wkt_string):
    point = wkt.loads(wkt_string)
    return point.x, point.y

result_path = result_path

merged_trip_50_df = pd.read_csv(os.path.join(result_path,'food_tour_50_merged_limit2h.csv'))
merged_trip_50_df[['start_lon', 'start_lat']] = merged_trip_50_df['trip_start_location'].apply(extract_coordinates).apply(pd.Series)

merged_trip_50_df = pd.merge(merged_trip_50_df, df_store_wm_filtered, left_on='retail_id',right_on='ObjectId')
merged_trip_50_df["user_id"] = merged_trip_50_df["user_id"].astype(str)
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_trip_50_df = pd.merge(merged_trip_50_df, user_table_df, left_on='user_id',right_on='user_id', how='inner')

merged_trip_50_df['home_start_dis'] = merged_trip_50_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['start_lat'], row['start_lon']), axis=1)

merged_trip_50_df.to_csv(os.path.join(result_path,'food_tour_50_merged_limit2h_joined_home_start.csv'), index=False)


merged_trip_50_df['trip_start_timestamp'] = pd.to_datetime(merged_trip_50_df['trip_start_timestamp'])
merged_trip_50_df['started_at_hour'] = merged_trip_50_df['trip_start_timestamp'].dt.hour
merged_trip_50_df['started_at_date'] = merged_trip_50_df['trip_start_timestamp'].dt.date

merged_trip_50_df["home_start_dis"] = merged_trip_50_df["home_start_dis"]/1000

import seaborn as sns
sns.distplot(merged_trip_50_df['home_start_dis'],hist=False)



grouped_tp_data = merged_trip_50_df.groupby('user_id')

new_tp_50_df = pd.DataFrame()
for user_id, group in grouped_tp_data:
    user_data = {
        'user_id': user_id,
        'total1_count' : group[group['TYPE'] == 1]['user_id'].count(),
        'total2_count' : group[group['TYPE'] == 2]['user_id'].count(),
        'total3_count' : group[group['TYPE'] == 3]['user_id'].count(),
        'total4_count' : group[group['TYPE'] == 4]['user_id'].count(),
        'total_home':  sum(group['home_start_dis'].apply(lambda x: (x < 1))),
        'type1_home': sum(group[group['TYPE'] == 1]['home_start_dis'].apply(lambda x: (x < 1))),
        'type2_home': sum(group[group['TYPE'] == 2]['home_start_dis'].apply(lambda x: (x < 1))),
        'type3_home': sum(group[group['TYPE'] == 3]['home_start_dis'].apply(lambda x: (x < 1))),
        'type4_home': sum(group[group['TYPE'] == 4]['home_start_dis'].apply(lambda x: (x < 1)))
    }
    
    new_tp_50_df = new_tp_50_df.append(user_data, ignore_index=True)

new_tp_50_df.to_csv(root + 'food_tour_50_stat_by_user_tour_2h.csv', index=False)


new_tp_50_df['type1_pct'] = new_tp_50_df['type1_home']/new_tp_50_df['total1_count']
new_tp_50_df['type2_pct'] = new_tp_50_df['type2_home']/new_tp_50_df['total2_count']
new_tp_50_df['type3_pct'] = new_tp_50_df['type3_home']/new_tp_50_df['total3_count']
new_tp_50_df['type4_pct'] = new_tp_50_df['type4_home']/new_tp_50_df['total4_count']
new_tp_50_df['type1_pct'].mean()*100
new_tp_50_df['type2_pct'].mean()*100
new_tp_50_df['type3_pct'].mean()*100
new_tp_50_df['type4_pct'].mean()*100


new_tp_50_df['type1_home'].sum()/new_tp_50_df['total1_count'].sum()*100
new_tp_50_df['type2_home'].sum()/new_tp_50_df['total2_count'].sum()*100
new_tp_50_df['type3_home'].sum()/new_tp_50_df['total3_count'].sum()*100
new_tp_50_df['type4_home'].sum()/new_tp_50_df['total4_count'].sum()*100







merged_trip_100_df = pd.read_csv(os.path.join(result_path,'food_tour_100_merged_limit2h.csv'))

merged_trip_100_df[['start_lon', 'start_lat']] = merged_trip_100_df['trip_start_location'].apply(extract_coordinates).apply(pd.Series)


merged_trip_100_df = pd.merge(merged_trip_100_df, df_store_wm_filtered, left_on='retail_id',right_on='ObjectId')
merged_trip_100_df["user_id"] = merged_trip_100_df["user_id"].astype(str)
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_trip_100_df = pd.merge(merged_trip_100_df, user_table_df, left_on='user_id',right_on='user_id', how='inner')

merged_trip_100_df['home_start_dis'] = merged_trip_100_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['start_lat'], row['start_lon']), axis=1)

merged_trip_100_df.to_csv(os.path.join(result_path,'food_tour_100_merged_limit2h_joined_home_start.csv'), index=False)


merged_trip_100_df['trip_start_timestamp'] = pd.to_datetime(merged_trip_100_df['trip_start_timestamp'])
merged_trip_100_df['started_at_hour'] = merged_trip_100_df['trip_start_timestamp'].dt.hour
merged_trip_100_df['started_at_date'] = merged_trip_100_df['trip_start_timestamp'].dt.date

merged_trip_100_df["home_start_dis"] = merged_trip_100_df["home_start_dis"]/1000

import seaborn as sns
sns.distplot(merged_trip_100_df['home_start_dis'],hist=False)



grouped_tp_data = merged_trip_100_df.groupby('user_id')

new_tp_100_df = pd.DataFrame()
for user_id, group in grouped_tp_data:
    user_data = {
        'user_id': user_id,
        'total1_count' : group[group['TYPE'] == 1]['user_id'].count(),
        'total2_count' : group[group['TYPE'] == 2]['user_id'].count(),
        'total3_count' : group[group['TYPE'] == 3]['user_id'].count(),
        'total4_count' : group[group['TYPE'] == 4]['user_id'].count(),
        'total_home':  sum(group['home_start_dis'].apply(lambda x: (x < 1))),
        'type1_home': sum(group[group['TYPE'] == 1]['home_start_dis'].apply(lambda x: (x < 1))),
        'type2_home': sum(group[group['TYPE'] == 2]['home_start_dis'].apply(lambda x: (x < 1))),
        'type3_home': sum(group[group['TYPE'] == 3]['home_start_dis'].apply(lambda x: (x < 1))),
        'type4_home': sum(group[group['TYPE'] == 4]['home_start_dis'].apply(lambda x: (x < 1)))
    }
    
    new_tp_100_df = new_tp_100_df.append(user_data, ignore_index=True)

new_tp_100_df.to_csv(root + 'food_tour_100_stat_by_user_tour_2h.csv', index=False)


new_tp_100_df['type1_pct'] = new_tp_100_df['type1_home']/new_tp_100_df['total1_count']
new_tp_100_df['type2_pct'] = new_tp_100_df['type2_home']/new_tp_100_df['total2_count']
new_tp_100_df['type3_pct'] = new_tp_100_df['type3_home']/new_tp_100_df['total3_count']
new_tp_100_df['type4_pct'] = new_tp_100_df['type4_home']/new_tp_100_df['total4_count']
new_tp_100_df['type1_pct'].mean()*100
new_tp_100_df['type2_pct'].mean()*100
new_tp_100_df['type3_pct'].mean()*100
new_tp_100_df['type4_pct'].mean()*100


new_tp_100_df['type1_home'].sum()/new_tp_100_df['total1_count'].sum()*100
new_tp_100_df['type2_home'].sum()/new_tp_100_df['total2_count'].sum()*100
new_tp_100_df['type3_home'].sum()/new_tp_100_df['total3_count'].sum()*100
new_tp_100_df['type4_home'].sum()/new_tp_100_df['total4_count'].sum()*100








merged_trip_150_df = pd.read_csv(os.path.join(result_path,'food_tour_150_merged_limit2h.csv'))

merged_trip_150_df[['start_lon', 'start_lat']] = merged_trip_150_df['trip_start_location'].apply(extract_coordinates).apply(pd.Series)


merged_trip_150_df = pd.merge(merged_trip_150_df, df_store_wm_filtered, left_on='retail_id',right_on='ObjectId')
merged_trip_150_df["user_id"] = merged_trip_150_df["user_id"].astype(str)
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_trip_150_df = pd.merge(merged_trip_150_df, user_table_df, left_on='user_id',right_on='user_id', how='inner')

merged_trip_150_df['home_start_dis'] = merged_trip_150_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['start_lat'], row['start_lon']), axis=1)

merged_trip_150_df.to_csv(os.path.join(result_path,'food_tour_150_merged_limit2h_joined_home_start.csv'), index=False)


merged_trip_150_df['trip_start_timestamp'] = pd.to_datetime(merged_trip_150_df['trip_start_timestamp'])
merged_trip_150_df['started_at_hour'] = merged_trip_150_df['trip_start_timestamp'].dt.hour
merged_trip_150_df['started_at_date'] = merged_trip_150_df['trip_start_timestamp'].dt.date

merged_trip_150_df["home_start_dis"] = merged_trip_150_df["home_start_dis"]/1000

import seaborn as sns
sns.distplot(merged_trip_150_df['home_start_dis'],hist=False)



grouped_tp_data = merged_trip_150_df.groupby('user_id')

new_tp_150_df = pd.DataFrame()
for user_id, group in grouped_tp_data:
    user_data = {
        'user_id': user_id,
        'total1_count' : group[group['TYPE'] == 1]['user_id'].count(),
        'total2_count' : group[group['TYPE'] == 2]['user_id'].count(),
        'total3_count' : group[group['TYPE'] == 3]['user_id'].count(),
        'total4_count' : group[group['TYPE'] == 4]['user_id'].count(),
        'total_home':  sum(group['home_start_dis'].apply(lambda x: (x < 1))),
        'type1_home': sum(group[group['TYPE'] == 1]['home_start_dis'].apply(lambda x: (x < 1))),
        'type2_home': sum(group[group['TYPE'] == 2]['home_start_dis'].apply(lambda x: (x < 1))),
        'type3_home': sum(group[group['TYPE'] == 3]['home_start_dis'].apply(lambda x: (x < 1))),
        'type4_home': sum(group[group['TYPE'] == 4]['home_start_dis'].apply(lambda x: (x < 1)))
    }
    
    new_tp_150_df = new_tp_150_df.append(user_data, ignore_index=True)

new_tp_150_df.to_csv(root + 'food_tour_150_stat_by_user_tour_2h.csv', index=False)


new_tp_150_df['type1_pct'] = new_tp_150_df['type1_home']/new_tp_150_df['total1_count']
new_tp_150_df['type2_pct'] = new_tp_150_df['type2_home']/new_tp_150_df['total2_count']
new_tp_150_df['type3_pct'] = new_tp_150_df['type3_home']/new_tp_150_df['total3_count']
new_tp_150_df['type4_pct'] = new_tp_150_df['type4_home']/new_tp_150_df['total4_count']
new_tp_150_df['type1_pct'].mean()*100
new_tp_150_df['type2_pct'].mean()*100
new_tp_150_df['type3_pct'].mean()*100
new_tp_150_df['type4_pct'].mean()*100


new_tp_150_df['type1_home'].sum()/new_tp_150_df['total1_count'].sum()*100
new_tp_150_df['type2_home'].sum()/new_tp_150_df['total2_count'].sum()*100
new_tp_150_df['type3_home'].sum()/new_tp_150_df['total3_count'].sum()*100
new_tp_150_df['type4_home'].sum()/new_tp_150_df['total4_count'].sum()*100





merged_trip_200_df = pd.read_csv(os.path.join(result_path,'food_tour_200_merged_limit2h.csv'))

merged_trip_200_df[['start_lon', 'start_lat']] = merged_trip_200_df['trip_start_location'].apply(extract_coordinates).apply(pd.Series)


merged_trip_200_df = pd.merge(merged_trip_200_df, df_store_wm_filtered, left_on='retail_id',right_on='ObjectId')
merged_trip_200_df["user_id"] = merged_trip_200_df["user_id"].astype(str)
user_table_df["user_id"] = user_table_df["user_id"].astype(str)
merged_trip_200_df = pd.merge(merged_trip_200_df, user_table_df, left_on='user_id',right_on='user_id', how='inner')

merged_trip_200_df['home_start_dis'] = merged_trip_200_df.apply(lambda row: calculate_bearing(row['LAT-4326'], row['LON-4326'],row['start_lat'], row['start_lon']), axis=1)

merged_trip_200_df.to_csv(os.path.join(result_path,'food_tour_200_merged_limit2h_joined_home_start.csv'), index=False)


merged_trip_200_df['trip_start_timestamp'] = pd.to_datetime(merged_trip_200_df['trip_start_timestamp'])
merged_trip_200_df['started_at_hour'] = merged_trip_200_df['trip_start_timestamp'].dt.hour
merged_trip_200_df['started_at_date'] = merged_trip_200_df['trip_start_timestamp'].dt.date

merged_trip_200_df["home_start_dis"] = merged_trip_200_df["home_start_dis"]/1000

import seaborn as sns
sns.distplot(merged_trip_200_df['home_start_dis'],hist=False)



grouped_tp_data = merged_trip_200_df.groupby('user_id')

new_tp_200_df = pd.DataFrame()
for user_id, group in grouped_tp_data:
    user_data = {
        'user_id': user_id,
        'total1_count' : group[group['TYPE'] == 1]['user_id'].count(),
        'total2_count' : group[group['TYPE'] == 2]['user_id'].count(),
        'total3_count' : group[group['TYPE'] == 3]['user_id'].count(),
        'total4_count' : group[group['TYPE'] == 4]['user_id'].count(),
        'total_home':  sum(group['home_start_dis'].apply(lambda x: (x < 1))),
        'type1_home': sum(group[group['TYPE'] == 1]['home_start_dis'].apply(lambda x: (x < 1))),
        'type2_home': sum(group[group['TYPE'] == 2]['home_start_dis'].apply(lambda x: (x < 1))),
        'type3_home': sum(group[group['TYPE'] == 3]['home_start_dis'].apply(lambda x: (x < 1))),
        'type4_home': sum(group[group['TYPE'] == 4]['home_start_dis'].apply(lambda x: (x < 1)))
    }
    
    new_tp_200_df = new_tp_200_df.append(user_data, ignore_index=True)

new_tp_200_df.to_csv(root + 'food_tour_200_stat_by_user_tour_2h.csv', index=False)


new_tp_200_df['type1_pct'] = new_tp_200_df['type1_home']/new_tp_200_df['total1_count']
new_tp_200_df['type2_pct'] = new_tp_200_df['type2_home']/new_tp_200_df['total2_count']
new_tp_200_df['type3_pct'] = new_tp_200_df['type3_home']/new_tp_200_df['total3_count']
new_tp_200_df['type4_pct'] = new_tp_200_df['type4_home']/new_tp_200_df['total4_count']
new_tp_200_df['type1_pct'].mean()*100
new_tp_200_df['type2_pct'].mean()*100
new_tp_200_df['type3_pct'].mean()*100
new_tp_200_df['type4_pct'].mean()*100


new_tp_200_df['type1_home'].sum()/new_tp_200_df['total1_count'].sum()*100
new_tp_200_df['type2_home'].sum()/new_tp_200_df['total2_count'].sum()*100
new_tp_200_df['type3_home'].sum()/new_tp_200_df['total3_count'].sum()*100
new_tp_200_df['type4_home'].sum()/new_tp_200_df['total4_count'].sum()*100
