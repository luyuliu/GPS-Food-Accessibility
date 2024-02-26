
import pandas as pd
import geopandas
import matplotlib.pyplot as plt



csv_user_home_all = pd.read_csv('home_location_merged.csv')
csv_user_home_all = csv_user_home_all.rename(columns={'Unnamed: 0': 'id'})
#csv_user_home_all = csv_user_home_all.reset_index().rename(columns={'Unnamed: 0': 'id'})
#csv_user_home_all['id'] = csv_user_home_all.index + 1
df_user_home_all = geopandas.GeoDataFrame(csv_user_home_all, geometry=geopandas.points_from_xy(csv_user_home_all['LON-4326'], csv_user_home_all['LAT-4326']))
df_user_home_all.crs = 'EPSG:4326'
df_user_home_all_wm = df_user_home_all.to_crs(epsg=3857)




df_bound = geopandas.read_file('./shapes/boundary_duval.shp')
df_bound.crs
df_bound_wm = df_bound.to_crs(epsg=3857)

# df_all_tracts = geopandas.read_file('./shapes/census_tracts_2010.shp')
# df_all_tracts.crs
# df_all_tracts_wm = df_all_tracts.to_crs(epsg=3857)

# df_tracts = geopandas.read_file('./shapes/duval_acs.shp')
# df_tracts = df_tracts[df_tracts['GEOID20'] != '12031990000']
# df_tracts.crs
# df_tracts_wm = df_tracts.to_crs(epsg=3857)

df_acs_duval = geopandas.read_file('./shapes/duval_acs_with_flag.shp')
df_acs_duval = df_acs_duval[df_acs_duval['GEOID20'] != '12031990000']
df_acs_duval.crs
df_acs_duval_wm = df_acs_duval.to_crs(epsg=3857)
#df_acs_duval_3086 = df_acs_duval.to_crs(epsg=3086)





df_acs_duval_wm_cal = df_acs_duval_wm.copy()
df_acs_duval_wm_cal['POP_18_39_DEN'] = (df_acs_duval_wm_cal['AGE_18_21']+df_acs_duval_wm_cal['AGE_22_29']+df_acs_duval_wm_cal['AGE_30_39'])/df_acs_duval_wm_cal['TOTALPOP']*100
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal.plot(column='POP_18_39_DEN', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent population of 18-39 years', fontsize=15)
plt.axis('off')
plt.show()


fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal.plot(column='DEN_POP', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Population per acre', fontsize=15)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal['WHITE_DEN'] = (df_acs_duval_wm_cal['WHITE']/df_acs_duval_wm_cal['TOTALPOP'])*100
df_acs_duval_wm_cal.plot(column='WHITE_DEN', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent population of Persons White alone', fontsize=15)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal.plot(column='MED_AGE', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Median age for population', fontsize=15)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal.plot(column='PCT_POV', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent population with income below poverty level ', fontsize=15)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal['VEHICLE_0_PCT'] = df_acs_duval_wm_cal['VEHICLE_0']/df_acs_duval_wm_cal['HSE_UNITS'] *100
df_acs_duval_wm_cal.plot(column='VEHICLE_0_PCT', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True,ax=ax)
plt.title('Percent housing units with no vehicle', fontsize=15)
plt.axis('off')
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
df_acs_duval_wm_cal.plot(column='LA1and10', linewidth=0.2, cmap='Blues', edgecolor='black', ax=ax, legend=False)
plt.title('Food desert tracts', fontsize=15)
plt.axis('off')
plt.show()



df_desert_wm = df_acs_duval_wm[df_acs_duval_wm['LA1and10'] == 1]
df_undesert_wm = df_acs_duval_wm[df_acs_duval_wm['LA1and10'] == 0]





df_store_all = geopandas.read_file('./shapes/food_retail_joined_new.shp')
df_store_all.crs
df_store_all_wm = df_store_all.to_crs(epsg=3857)

df_store_wm = df_store_all_wm
store_in = geopandas.sjoin(df_store_wm,df_desert_wm)
store_out = geopandas.sjoin(df_store_wm,df_undesert_wm)


df_store_old = geopandas.read_file('./shapes/food_retail_joined_duval.shp')
df_store_old.crs
df_store_old_wm = df_store_old.to_crs(epsg=3857)




df_user_home_wm = geopandas.sjoin(df_user_home_all_wm,df_acs_duval_wm_cal)

df_desert_wm = df_desert_wm.rename(columns={'index_right': 'index_area_right'})
df_desert_wm = df_desert_wm.rename(columns={'index_left': 'index_area_left'})
df_user_home_wm = df_user_home_wm.rename(columns={'index_right': 'index_point_right'})
df_user_home_wm = df_user_home_wm.rename(columns={'index_left': 'index_point_left'})
user_in = geopandas.sjoin(df_user_home_wm,df_desert_wm)
user_in_duval = geopandas.sjoin(df_user_home_wm,df_acs_duval_wm_cal)
#user_out = geopandas.overlay(df_user_home_wm,df_desert_wm,"difference")
df_undesert_wm = df_undesert_wm.rename(columns={'index_right': 'index_area_right'})
df_undesert_wm = df_undesert_wm.rename(columns={'index_left': 'index_area_left'})
user_out = geopandas.sjoin(df_user_home_wm,df_undesert_wm)


root = "D:/food_related_trips/results/"

user_with_stops = pd.read_csv(root + 'retail_stat_by_user_before_cp_2h.csv')
df_user_home_wm_with_stops = df_user_home_wm[df_user_home_wm['user_id'].isin(user_with_stops['user_id'])]


points = df_user_home_wm_with_stops
polygons = df_acs_duval_wm_cal


points_in_polygons = geopandas.sjoin(points, polygons, op='within')
point_counts = points_in_polygons.groupby('GEOID20_left').size().reset_index(name='point_count')
polygons_with_count = polygons.merge(point_counts, left_on='GEOID20', right_on='GEOID20_left', how='left')
polygons_with_count[['point_count']] = polygons_with_count[['point_count']].fillna(0)

polygons_with_count['sampling_rate'] = polygons_with_count['point_count']/polygons_with_count['TOTALPOP']*100

# Plotting the choropleth map with a labeled colorbar
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
polygons_with_count.plot(column='sampling_rate', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True, vmin=0, vmax=30, ax=ax)

# Add colorbar with a labeled title
cbar = ax.get_figure().get_axes()[1]
cbar.set_ylabel('Proportion of Users with Food-Related Stops', rotation=270, labelpad=15)

plt.title('Proportion of Tract Population with Food-Related Stops (%)', fontsize=15)
plt.axis('off')
plt.show()



import os
polygons_with_count_lim = polygons_with_count[['geometry','POP_18_39_DEN','DEN_POP','WHITE_DEN','MED_AGE','PCT_POV','VEHICLE_0_PCT','LA1and10','point_count','sampling_rate']]
polygons_with_count_lim.to_csv(os.path.join(root,'sample_rate_withstop.csv'))



import numpy as np
from scipy.stats import poisson, chi2


observed_rate = polygons_with_count['sampling_rate']
mean_count = np.mean(observed_rate)
variance_count = np.var(observed_rate)

# Calculate the variance-to-mean ratio
vm_ratio = variance_count / mean_count

# Perform a chi-squared test for overdispersion
degrees_of_freedom = len(observed_rate) - 1
chi2_stat = degrees_of_freedom * vm_ratio
p_value = 1 - chi2.cdf(chi2_stat, degrees_of_freedom)

# Output results
print(f"Mean: {mean_count}")
print(f"Variance: {variance_count}")
print(f"Variance-to-Mean Ratio: {vm_ratio}")
print(f"Chi-squared Statistic: {chi2_stat}")
print(f"P-value: {p_value}")

# Perform a significance test (e.g., at a 5% significance level)
if p_value < 0.05:
    print("The data shows evidence of overdispersion.")
else:
    print("The data does not show evidence of overdispersion.")




import seaborn as sns
correlation_matrix = polygons_with_count_lim[['POP_18_39_DEN','DEN_POP','WHITE_DEN','MED_AGE','PCT_POV','VEHICLE_0_PCT','LA1and10','sampling_rate']].corr()
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()






points = df_user_home_wm_with_stops
polygons = df_acs_duval_wm

points_in_polygons = geopandas.sjoin(points, polygons, op='within')
point_counts = points_in_polygons.groupby('GEOID20_left').size().reset_index(name='point_count')
polygons_with_count = polygons.merge(point_counts, left_on='GEOID20', right_on='GEOID20_left', how='left')
polygons_with_count[['point_count']] = polygons_with_count[['point_count']].fillna(0)

polygons_with_count['sampling_rate'] = polygons_with_count['point_count']/polygons_with_count['TOTALPOP']*100

# Plotting the choropleth map with a labeled colorbar
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
polygons_with_count.plot(column='sampling_rate', cmap='viridis', linewidth=0.2, edgecolor='black', legend=True, vmin=0, vmax=30, ax=ax)

# Add colorbar with a labeled title
cbar = ax.get_figure().get_axes()[1]
cbar.set_ylabel('Proportion of Users with Food-Related Stops', rotation=270, labelpad=15)

plt.title('Proportion of Tract Population with Food-Related Stops (%)', fontsize=15)
plt.axis('off')
plt.show()


import seaborn as sns
polygons_with_count_his = polygons_with_count.copy()
polygons_with_count_his.loc[polygons_with_count_his["sampling_rate"]>30,"sampling_rate"]=30
sns.histplot(polygons_with_count_his['sampling_rate'])
plt.title('Distribution of Sampling Rates for Food-Related Stops Across Tracts')
plt.xlabel("Proportion of Tract Population with Food-Related Stops (%)")
plt.ylabel("Number of Tracts")
plt.xlim(0, 30)
plt.show()








