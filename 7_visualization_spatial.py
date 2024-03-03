"""
based on the extracted individual metrics, calculate the tract average, and plot
 
Input: 
    1. calculated individual metrics
    2. home location info
    3. study area shapefile

Output: 
    1. maps of tract-level metrics
    2. other figures (histograms, scatter plots, etc.)
"""



import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import seaborn as sns

# read related dataset
# study area
df_acs_duval = geopandas.read_file('shapes/boundary_duval.shp')
# home location
df_user_home = pd.read_csv('home_location.shp')
df_user_home["user_id"] = df_user_home["user_id"].astype(str)
df_user_home = geopandas.sjoin(df_user_home,df_acs_duval)
df_user_home = df_user_home[["user_id",'GEOID20']]
# retailer data
df_store_all = geopandas.read_file('shapes/retail.shp')


# types of stores
type_1 = 'Large Groceries'
type_2 = 'Big Box Stores'
type_3 = 'Small Healthy Outlets'
type_4 = 'Processed Food Outlets'
the_index = [type_1,type_2,type_3,type_4]

# radius
rad = 200



# metric 1: number of visits
# read calculated metrics
count_by_user = pd.read_csv('food_sp_{rad}_count_by_user.csv')
count_by_user["user_id"] = count_by_user["user_id"].astype(str)
count_by_user = pd.merge(count_by_user, df_user_home, left_on='user_id',right_on='user_id', how='inner')

# by type
count_by_user["trip_cnt_ave"] = count_by_user["type1_count"]
# join and aggreg by tract
trip_cnt_ave = count_by_user.groupby('GEOID20')['trip_cnt_ave'].mean().reset_index(name='trip_cnt_ave')
# spatial join
polygons = df_acs_duval
polygons_with_ave = polygons.merge(trip_cnt_ave)
# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_cnt_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(f'number of visits {type_1}', fontsize=15)
ax.axis('off')
plt.show()





# metric 2: number of unique visits
# read calculated metrics
unique_by_user = pd.read_csv('food_sp_{rad}_unique_by_user.csv')
unique_by_user["user_id"] = unique_by_user["user_id"].astype(str)
unique_by_user = pd.merge(unique_by_user, df_user_home, left_on='user_id',right_on='user_id', how='inner')

# by type
unique_by_user["trip_uni_ave"] = unique_by_user["type1_uni_count"]
# join and aggreg by tract
trip_uni_ave = unique_by_user.groupby('GEOID20')['trip_uni_ave'].mean().reset_index(name='trip_uni_ave')
# spatial join
polygons = df_acs_duval
polygons_with_ave = polygons.merge(trip_uni_ave)
# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_uni_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(f'number of unique stores visited {type_1}', fontsize=15)
ax.axis('off')
plt.show()






# metric 3: home to store distance

# read calculated metrics
dis_by_user = pd.read_csv('food_sp_{rad}_dis_by_user.csv')
dis_by_user["user_id"] = dis_by_user["user_id"].astype(str)
dis_by_user = pd.merge(dis_by_user, df_user_home, left_on='user_id',right_on='user_id', how='inner')

nearest_by_user = pd.read_csv('nearest_store_by_user.csv')
nearest_by_user["user_id"] = nearest_by_user["user_id"].astype(str)
nearest_by_user = pd.merge(nearest_by_user, df_user_home, left_on='user_id',right_on='user_id', how='inner')

# by type
# visited distance
dis_by_user["trip_dis_ave"] = dis_by_user["type1_mean"]
# join and aggreg by tract
trip_dis_ave = dis_by_user.groupby('GEOID20')['trip_dis_ave'].mean().reset_index(name='trip_dis_ave')
# spatial join
polygons = df_acs_duval
polygons_with_ave = polygons.merge(trip_dis_ave)
# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(f'average home-store distance (km) {type_1}', fontsize=15)
ax.axis('off')
plt.show()
# histograms
his_1 = dis_by_user['type1_mean'] 
his_2 = dis_by_user['type2_mean'] 
his_3 = dis_by_user['type3_mean'] 
his_4 = dis_by_user['type4_mean'] 
plt.rcParams["figure.figsize"] = (6,5)
sns.set_style('darkgrid')
sns.distplot(his_1,label=type_1,hist=False)
sns.distplot(his_2,label=type_2,hist=False)
sns.distplot(his_3,label=type_3,hist=False)
sns.distplot(his_4,label=type_4,hist=False)
plt.title('histogram of home to visited-store distance (km)', fontsize=15)
plt.xlabel("distance (km)")
plt.xlim(0, 30)
plt.legend()
plt.show()


# nearest distance
nearest_by_user["nearest_dis_ave"] = nearest_by_user["distance_1"]
# join and aggreg by tract
nearest_dis_ave = nearest_by_user.groupby('GEOID20')['nearest_dis_ave'].mean().reset_index(name='nearest_dis_ave')
# spatial join
polygons = df_acs_duval
polygons_with_ave = polygons.merge(nearest_dis_ave)
# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='nearest_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(f'nearest home-store distance (km) {type_1}', fontsize=15)
ax.axis('off')
plt.show()
# histograms
his_1 = nearest_by_user['distance_1'] 
his_2 = nearest_by_user['distance_2'] 
his_3 = nearest_by_user['distance_3'] 
his_4 = nearest_by_user['distance_4'] 
plt.rcParams["figure.figsize"] = (6,5)
sns.set_style('darkgrid')
sns.distplot(his_1,label=type_1,hist=False)
sns.distplot(his_2,label=type_2,hist=False)
sns.distplot(his_3,label=type_3,hist=False)
sns.distplot(his_4,label=type_4,hist=False)
plt.title('histogram of home to nearest-store distance (km)', fontsize=15)
plt.xlabel("distance (km)")
plt.xlim(0, 30)
plt.legend()
plt.show()


# compare nearest and visited
df_merged = pd.merge(dis_by_user, nearest_by_user, on='user_id', how='left')
data = {'X': df_merged['type1_mean'],
        'Y': df_merged['distance_1']}
df = pd.DataFrame(data)
plt.figure(figsize=(6, 6))
sns.kdeplot(x='X', y='Y', data=df, fill=True, cmap='Blues', levels=5,clip=((0,100),(0,100)))
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title(type_1)
plt.show()





# metric 4: home-based trip proportion
# read calculated metrics
pct_by_user = pd.read_csv('food_trip_{rad}_pct_by_user.csv')
pct_by_user["user_id"] = pct_by_user["user_id"].astype(str)
pct_by_user = pd.merge(pct_by_user, df_user_home, left_on='user_id',right_on='user_id', how='inner')

# by type
pct_by_user["percent_median"] = pct_by_user["type1_pct"]
# join and aggreg by tract
percent_median = pct_by_user.groupby('GEOID20')['percent_median'].mean().reset_index(name='percent_median')
# spatial join
polygons = df_acs_duval
polygons_with_ave = polygons.merge(percent_median)
# plot
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='percent_median', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(f'proportion of home-based visit (%) {type_1}', fontsize=15)
ax.axis('off')
plt.show()