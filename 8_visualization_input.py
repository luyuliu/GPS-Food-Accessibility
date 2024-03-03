
"""
plot the non-metrics-related datasets in maps
 
Input: 
    1. study area shapefile and ACS data
    2. inferred user home location
    3. food-related stops
    
Output: 
    1. sociodemographic characteristics maps
    2. sampling rate map
"""

import pandas as pd
import geopandas
import matplotlib.pyplot as plt



# read inferred user home location csv and conver to geopandas geodf
csv_user_home = pd.read_csv('home_location.csv')
df_user_home = geopandas.GeoDataFrame(csv_user_home, geometry=geopandas.points_from_xy(csv_user_home['LON-4326'], csv_user_home['LAT-4326']))

# read study are shapefile
df_acs_duval = geopandas.read_file('shapes/boundary_duval.shp')

# read retailer data
df_store_all = geopandas.read_file('shapes/retail.shp')

# read food stops
rad = 200 
food_stops = pd.read_csv('food_sp_{rad}_limit2h_with_store_user')
df_user_with_stops = df_user_home[df_user_home['user_id'].isin(food_stops['user_id'])]




# plot socio demographic data
# Percent population of 18-39 years
df_acs_duval_cal = df_acs_duval.copy()
df_acs_duval_cal['POP_18_39_DEN'] = (df_acs_duval_cal['AGE_18_21']+df_acs_duval_cal['AGE_22_29']+df_acs_duval_cal['AGE_30_39'])/df_acs_duval_cal['TOTALPOP']*100
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal.plot(column='POP_18_39_DEN', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent population of 18-39 years', fontsize=15)
plt.axis('off')
plt.show()
# Population per acre
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal.plot(column='DEN_POP', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Population per acre', fontsize=15)
plt.axis('off')
plt.show()
# Percent population of Persons White alone
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal['WHITE_DEN'] = (df_acs_duval_cal['WHITE']/df_acs_duval_cal['TOTALPOP'])*100
df_acs_duval_cal.plot(column='WHITE_DEN', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent population of Persons White alone', fontsize=15)
plt.axis('off')
plt.show()
# Median age for population
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal.plot(column='MED_AGE', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Median age for population', fontsize=15)
plt.axis('off')
plt.show()
# Percent population with income below poverty level
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal.plot(column='PCT_POV', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent population with income below poverty level ', fontsize=15)
plt.axis('off')
plt.show()
# Percent housing units with no vehicle
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal['VEHICLE_0_PCT'] = df_acs_duval_cal['VEHICLE_0']/df_acs_duval_cal['HSE_UNITS'] *100
df_acs_duval_cal.plot(column='VEHICLE_0_PCT', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent housing units with no vehicle', fontsize=15)
plt.axis('off')
plt.show()
# Food desert tracts
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_cal.plot(column='LA1and10', linewidth=0.2, cmap='Blues', edgecolor='black', ax=ax, legend=False)
plt.title('Food desert tracts', fontsize=15)
plt.axis('off')
plt.show()






# plot sampling rate
points = df_user_with_stops
polygons = df_acs_duval
# perform spatial join, count point in polygon
points_in_polygons = geopandas.sjoin(points, polygons, op='within')
point_counts = points_in_polygons.groupby('GEOID20_left').size().reset_index(name='point_count')
polygons_with_count = polygons.merge(point_counts, left_on='GEOID20', right_on='GEOID20_left', how='left')
polygons_with_count[['point_count']] = polygons_with_count[['point_count']].fillna(0)
# calculate tract sampling rate
polygons_with_count['sampling_rate'] = polygons_with_count['point_count']/polygons_with_count['TOTALPOP']*100
# spatial map
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
polygons_with_count.plot(column='sampling_rate', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True, vmin=0, vmax=30, ax=ax)
cbar = ax.get_figure().get_axes()[1]
cbar.set_ylabel('Proportion of Users with Food-Related Stops', rotation=270, labelpad=15)
plt.title('Proportion of Tract Population with Food-Related Stops (%)', fontsize=15)
plt.axis('off')
plt.show()
# hist
import seaborn as sns
sns.histplot(polygons_with_count['sampling_rate'])
plt.title('Distribution of Sampling Rates for Food-Related Stops Across Tracts')
plt.xlabel("Proportion of Tract Population with Food-Related Stops (%)")
plt.ylabel("Number of Tracts")
plt.show()



