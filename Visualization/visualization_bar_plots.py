import matplotlib.pyplot as plt
import numpy as np










root = root


# Number of stops
retail_types = ['\nTotal', '\nType1', '\nType2', '\nType3', '\nType4']

radius = ['50', '100', '150', '200']


stop_50 = stop_ave_50
stop_100 = stop_ave_100
stop_150 = stop_ave_150
stop_200 = stop_ave_200

# Define the x-axis positions for each day
x = np.arange(len(retail_types))

bar_width = 0.1

plt.rcParams["figure.figsize"] = (8,4)

plt.text(-0.24,-0.5,'50     200             50     200              50     200              50     200              50     200', fontsize=10)

plt.bar(x - 1.75*bar_width, stop_50, bar_width, color='deepskyblue')
plt.bar(x - 0.5*bar_width, stop_100, bar_width, color='deepskyblue')
plt.bar(x + 0.75*bar_width, stop_150, bar_width, color='deepskyblue')
plt.bar(x + 2*bar_width, stop_200, bar_width, color='deepskyblue')

plt.xlabel('Store type')
plt.ylabel('Average number of trips')
#plt.title('Number of Food Trips')

plt.xticks(x, retail_types)

#plt.legend()
# plt.legend(ncol=2)

plt.show()



















# unique stores

radius = ['50', '100', '150', '200']

stop_50 = unique_ave_50
stop_100 = unique_ave_100
stop_150 = unique_ave_150
stop_200 = unique_ave_200


# Define the x-axis positions for each day
x = np.arange(len(retail_types))

bar_width = 0.1

plt.rcParams["figure.figsize"] = (8,4)

plt.bar(x - 1.75*bar_width, stop_50, bar_width, color='deepskyblue')
plt.bar(x - 0.5*bar_width, stop_100, bar_width, color='deepskyblue')
plt.bar(x + 0.75*bar_width, stop_150, bar_width, color='deepskyblue')
plt.bar(x + 2*bar_width, stop_200, bar_width, color='deepskyblue')

plt.text(-0.24,-0.15,'50     200             50     200              50     200              50     200              50     200', fontsize=10)

plt.xlabel('Store type')
plt.ylabel('Average number of unique stores visited')
#plt.title('Number of Food Trips')

plt.xticks(x, retail_types)

#plt.legend()
# plt.legend(ncol=2)

plt.show()

























# home-store distance
radius = ['50', '100', '150', '200']

stop_50 = dis_mean_50
stop_100 = dis_mean_100
stop_150 = dis_mean_150
stop_200 = dis_mean_200


# Define the x-axis positions for each day
x = np.arange(len(retail_types))

bar_width = 0.1

plt.bar(x - 1.75*bar_width, stop_50, bar_width, color='deepskyblue')
plt.bar(x - 0.5*bar_width, stop_100, bar_width, color='deepskyblue')
plt.bar(x + 0.75*bar_width, stop_150, bar_width, color='deepskyblue')
plt.bar(x + 2*bar_width, stop_200, bar_width, color='deepskyblue')

plt.text(-0.24,-0.6,'50     200             50     200              50     200              50     200              50     200', fontsize=10)

plt.ylim(0, 11) 

plt.xlabel('Store type')
plt.ylabel('Average home-to-store distance (km)')
#plt.title('Number of Food Trips')

plt.xticks(x, retail_types)

#plt.legend(loc='upper left')
# plt.legend(ncol=2)

plt.show()

















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


#result_df.to_csv(root + 'nearest_store_results_new_type_1007_back.csv', index=False)




result_df = 
result_df["user_id"] = result_df["user_id"].astype(float).astype(int).astype(str)

nn_user_in = pd.merge(user_in_short, result_df, on='user_id', how='left')
nn_user_out = pd.merge(user_out_short, result_df, on='user_id', how='left')

his_1_in = nn_user_in['distance_1'] 
his_2_in = nn_user_in['distance_2'] 
his_3_in = nn_user_in['distance_3'] 
his_4_in = nn_user_in['distance_4'] 

his_1_out = nn_user_out['distance_1'] 
his_2_out = nn_user_out['distance_2'] 
his_3_out = nn_user_out['distance_3'] 
his_4_out = nn_user_out['distance_4'] 

# nearest












# travelled dist and store dist
hts_50_dis = pd.read_csv(root + 'retail_50_dist_by_user_new.csv')
hts_100_dis = pd.read_csv(root + 'retail_100_dist_by_user_new.csv')
hts_150_dis = pd.read_csv(root + 'retail_150_dist_by_user_new.csv')
hts_200_dis = pd.read_csv(root + 'retail_200_dist_by_user_new.csv')

hts_50_dis["user_id"] = hts_50_dis["user_id"].astype(float).astype(int).astype(str)
hts_100_dis["user_id"] = hts_100_dis["user_id"].astype(float).astype(int).astype(str)
hts_150_dis["user_id"] = hts_150_dis["user_id"].astype(float).astype(int).astype(str)
hts_200_dis["user_id"] = hts_200_dis["user_id"].astype(float).astype(int).astype(str)

hts_50_in = pd.merge(user_in_short, hts_50_dis, on='user_id', how='left')
hts_50_out = pd.merge(user_out_short, hts_50_dis, on='user_id', how='left')
hts_100_in = pd.merge(user_in_short, hts_100_dis, on='user_id', how='left')
hts_100_out = pd.merge(user_out_short, hts_100_dis, on='user_id', how='left')
hts_150_in = pd.merge(user_in_short, hts_150_dis, on='user_id', how='left')
hts_150_out = pd.merge(user_out_short, hts_150_dis, on='user_id', how='left')
hts_200_in = pd.merge(user_in_short, hts_200_dis, on='user_id', how='left')
hts_200_out = pd.merge(user_out_short, hts_200_dis, on='user_id', how='left')

his_50_1_in = hts_50_in['type1_mean'] 
his_100_1_in = hts_100_in['type1_mean'] 
his_150_1_in = hts_150_in['type1_mean'] 
his_200_1_in = hts_200_in['type1_mean'] 

his_50_1_out = hts_50_out['type1_mean'] 
his_100_1_out = hts_100_out['type1_mean'] 
his_150_1_out = hts_150_out['type1_mean'] 
his_200_1_out = hts_200_out['type1_mean'] 

his_50_2_in = hts_50_in['type2_mean'] 
his_100_2_in = hts_100_in['type2_mean'] 
his_150_2_in = hts_150_in['type2_mean'] 
his_200_2_in = hts_200_in['type2_mean'] 

his_50_2_out = hts_50_out['type2_mean'] 
his_100_2_out = hts_100_out['type2_mean'] 
his_150_2_out = hts_150_out['type2_mean'] 
his_200_2_out = hts_200_out['type2_mean'] 


his_50_3_in = hts_50_in['type3_mean'] 
his_100_3_in = hts_100_in['type3_mean'] 
his_150_3_in = hts_150_in['type3_mean'] 
his_200_3_in = hts_200_in['type3_mean'] 

his_50_3_out = hts_50_out['type3_mean'] 
his_100_3_out = hts_100_out['type3_mean'] 
his_150_3_out = hts_150_out['type3_mean'] 
his_200_3_out = hts_200_out['type3_mean'] 

his_50_4_in = hts_50_in['type4_mean'] 
his_100_4_in = hts_100_in['type4_mean'] 
his_150_4_in = hts_150_in['type4_mean'] 
his_200_4_in = hts_200_in['type4_mean'] 

his_50_4_out = hts_50_out['type4_mean'] 
his_100_4_out = hts_100_out['type4_mean'] 
his_150_4_out = hts_150_out['type4_mean'] 
his_200_4_out = hts_200_out['type4_mean'] 



import seaborn as sns
sns.set_style('darkgrid')

sns.distplot(his_200_1_in,label='visited_type1_desert',hist=False,color='red')
sns.distplot(his_200_1_out,label='visited_type1_nondesert',hist=False,color='red', kde_kws={'linestyle':'--'})
sns.distplot(his_1_in,label='nearest_type1_desert',hist=False,color='black')
sns.distplot(his_1_out,label='nearest_type1_nondesert',hist=False,color='black', kde_kws={'linestyle':'--'})


plt.title('hist of average home-store distance (km) type1, 200m', fontsize=15)

plt.xlim(0, 30)
plt.ylim(0, 1)
plt.legend()
plt.show()



sns.set_style('darkgrid')
sns.distplot(his_200_2_in,label='visited_type2_desert',hist=False,color='red')
sns.distplot(his_200_2_out,label='visited_type2_nondesert',hist=False,color='red', kde_kws={'linestyle':'--'})
sns.distplot(his_2_in,label='nearest_type2_desert',hist=False,color='black')
sns.distplot(his_2_out,label='nearest_type2_nondesert',hist=False,color='black', kde_kws={'linestyle':'--'})

plt.title('hist of average home-store distance (km) type2, 200m', fontsize=15)

plt.xlim(0, 30)
plt.ylim(0, 1)
plt.legend()
plt.show()




import seaborn as sns
sns.set_style('darkgrid')

sns.distplot(his_50_3_in,label='visited_type3_desert',hist=False,color='red')
sns.distplot(his_50_3_out,label='visited_type3_nondesert',hist=False,color='red', kde_kws={'linestyle':'--'})
sns.distplot(his_3_in,label='nearest_type3_desert',hist=False,color='black')
sns.distplot(his_3_out,label='nearest_type3_nondesert',hist=False,color='black', kde_kws={'linestyle':'--'})


plt.title('hist of average home-store distance (km) type3, 50m', fontsize=15)

plt.xlim(0, 30)
plt.ylim(0, 1)
plt.legend()
plt.show()



import seaborn as sns
sns.set_style('darkgrid')

sns.distplot(his_50_4_in,label='visited_type4_desert',hist=False,color='red')
sns.distplot(his_50_4_out,label='visited_type4_nondesert',hist=False,color='red', kde_kws={'linestyle':'--'})
sns.distplot(his_4_in,label='nearest_type4_desert',hist=False,color='black')
sns.distplot(his_4_out,label='nearest_type4_nondesert',hist=False,color='black', kde_kws={'linestyle':'--'})


plt.title('hist of average home-store distance (km) type4, 50m', fontsize=15)

plt.xlim(0, 30)
plt.ylim(0, 1)
plt.legend()
plt.show()












# sns.distplot(his_50_1_in,label='type1_50_desert',hist=False,color='red')
# sns.distplot(his_100_1_in,label='type1_100_desert',hist=False,color='orange')
# sns.distplot(his_150_1_in,label='type1_150_desert',hist=False,color='green')
# sns.distplot(his_200_1_in,label='type1_200_desert',hist=False,color='blue')
# sns.distplot(his_50_1_out,label='type1_50_non_desert',hist=False,color='red', kde_kws={'linestyle':'--'})
# sns.distplot(his_100_1_out,label='type1_100_non_desert',hist=False,color='orange', kde_kws={'linestyle':'--'})
# sns.distplot(his_150_1_out,label='type1_150_non_desert',hist=False,color='green', kde_kws={'linestyle':'--'})
# sns.distplot(his_200_1_out,label='type1_200_non_desert',hist=False,color='blue', kde_kws={'linestyle':'--'})

