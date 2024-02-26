














nearest=pd.read_csv(root + 'nearest_store_results.csv')

nearest['user_id'] = nearest['user_id'].round().astype(int)

his_1_nn = nearest['distance_1'] 
his_2_nn = nearest['distance_2'] 
his_3_nn = nearest['distance_3'] 
his_4_nn = nearest['distance_4'] 

import seaborn as sns
sns.set_style('darkgrid')

plt.rcParams["figure.figsize"] = (6,5)

sns.distplot(his_1_nn,label='Large Groceries',hist=False)
sns.distplot(his_2_nn,label='Big Box Stores',hist=False)
sns.distplot(his_3_nn,label='Small Healthy Outlets',hist=False)
sns.distplot(his_4_nn,label='Processed Food Outlets',hist=False)

plt.title('Histogram of home to nearest-store distance (km)', fontsize=15)

plt.xlabel("distance (km)")

plt.xlim(0, 30)
#plt.ylim(0, 0.2)
plt.legend()
plt.show()




visited = pd.read_csv(root + 'retail_stat_by_user_before_cp_2h.csv')

his_1 = visited['type1_mean'] 
his_2 = visited['type2_mean'] 
his_3 = visited['type3_mean'] 
his_4 = visited['type4_mean'] 


import seaborn as sns
sns.set_style('darkgrid')

plt.rcParams["figure.figsize"] = (6,5)

sns.distplot(his_1,label='Large Groceries',hist=False)
sns.distplot(his_2,label='Big Box Stores',hist=False)
sns.distplot(his_3,label='Small Healthy Outlets',hist=False)
sns.distplot(his_4,label='Processed Food Outlets',hist=False)

plt.title('Histogram of home to visited-store distance (km)', fontsize=15)

plt.xlabel("distance (km)")

plt.xlim(0, 30)
plt.ylim(0, 0.12)
plt.legend()
plt.show()



df_merged = pd.merge(visited, nearest, on='user_id', how='left')


y_lim = max(df_merged['type1_mean'])

# ,clip=(-10,35)

# 1
data = {'X': df_merged['type1_mean'],
        'Y': df_merged['distance_1']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.kdeplot(x='X', y='Y', data=df, fill=True, cmap='Blues', levels=5,clip=((0,100),(0,100)))
plt.ylim(0,43)
plt.xlim(0,40)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Large Groceries')
plt.show()


# 2
data = {'X': df_merged['type2_mean'],
        'Y': df_merged['distance_2']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.kdeplot(x='X', y='Y', data=df, fill=True, cmap='Blues', levels=5,clip=((0,60),(0,60)))
plt.ylim(0,43)
plt.xlim(0,40)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Big Box Stores')
plt.show()


# 3
data = {'X': df_merged['type3_mean'],
        'Y': df_merged['distance_3']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.kdeplot(x='X', y='Y', data=df, fill=True, cmap='Blues', levels=5,clip=((0,100),(0,100)))
plt.ylim(0,43)
plt.xlim(0,40)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Small Healthy Outlets')
plt.show()


# 4
data = {'X': df_merged['type4_mean'],
        'Y': df_merged['distance_4']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.kdeplot(x='X', y='Y', data=df, fill=True, cmap='Blues', levels=5,clip=((0,60),(0,60)))
plt.ylim(0,43)
plt.xlim(0,40)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Processed Food Outlets')
plt.show()






types = ['Large Groceries','Big Box Stores', 'Small Healthy Outlets','Processed Food Outlets']





df_merged = pd.merge(visited, nearest, on='user_id', how='left')


y_lim = max(df_merged['type1_mean'])

# ,clip=(-10,35)

# 1
data = {'X': df_merged['type1_mean'],
        'Y': df_merged['distance_1']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.scatterplot(x='type1_mean', y='distance_1', data=df_merged,s=1)
plt.ylim(0,30)
plt.xlim(0,30)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Large Groceries')
plt.show()


# 2
data = {'X': df_merged['type2_mean'],
        'Y': df_merged['distance_2']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.scatterplot(x='type2_mean', y='distance_2', data=df_merged,s=1,alpha=0.5)
plt.ylim(0,30)
plt.xlim(0,30)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Big Box Stores')
plt.show()


# 3
data = {'X': df_merged['type3_mean'],
        'Y': df_merged['distance_3']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.scatterplot(x='type3_mean', y='distance_3', data=df_merged,s=1)
plt.ylim(0,30)
plt.xlim(0,30)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Small Healthy Outlets')
plt.show()


# 4
data = {'X': df_merged['type4_mean'],
        'Y': df_merged['distance_4']}
df = pd.DataFrame(data)

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
sns.scatterplot(x='type4_mean', y='distance_4', data=df_merged,s=1)
plt.ylim(0,30)
plt.xlim(0,30)
plt.xlabel('visited')
plt.ylabel('nearest')
plt.title('Processed Food Outlets')
plt.show()











df_merged = pd.merge(visited, nearest, on='user_id', how='left')


y_lim = max(df_merged['type1_mean'])

ax = sns.scatterplot(x='type1_mean', y='distance_1', data=df_merged)
ax.set_aspect('auto')
ax.set_ylim=(0,50)
plt.show()

plt.title('Large Groceries')

















nearest=pd.read_csv(root + 'nearest_store_results_new_type_1007.csv')
visited = pd.read_csv(root + 'retail_stat_by_user_before_cp_2h.csv')

food_related_stop_df =  df_merged.copy()





food_related_stop_df["home_retail_dis_vs"] = food_related_stop_df["type1_mean"]
food_related_stop_df["home_retail_dis_nn"] = food_related_stop_df["distance_1"]

df_user_home_wm_short = df_user_home_wm[["user_id",'GEOID20']]
polygons = df_acs_duval_wm

trip_store_dis_df = food_related_stop_df.merge(df_user_home_wm_short, left_on='user_id', right_on='user_id', how='left')
trip_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_vs'].mean().reset_index(name='trip_dis_ave')
nnst_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_nn'].mean().reset_index(name='nnst_dis_ave')

polygons_with_ave = polygons.merge(trip_dis_ave)
polygons_with_ave[['trip_dis_ave']] = polygons_with_ave[['trip_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[0]+' - visited', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave)
polygons_with_ave[['nnst_dis_ave']] = polygons_with_ave[['nnst_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='nnst_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[0]+' - nearest', fontsize=15)
ax.axis('off')
plt.show()


polygons_with_ave = polygons.merge(nnst_dis_ave).merge(trip_dis_ave)
polygons_with_ave['diff_dis_ave'] = polygons_with_ave['trip_dis_ave'] - polygons_with_ave['nnst_dis_ave']
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='diff_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(types[0]+' - difference', fontsize=15)
ax.axis('off')
plt.show()








food_related_stop_df["home_retail_dis_vs"] = food_related_stop_df["type2_mean"]
food_related_stop_df["home_retail_dis_nn"] = food_related_stop_df["distance_2"]

df_user_home_wm_short = df_user_home_wm[["user_id",'GEOID20']]
polygons = df_acs_duval_wm

trip_store_dis_df = food_related_stop_df.merge(df_user_home_wm_short, left_on='user_id', right_on='user_id', how='left')
trip_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_vs'].mean().reset_index(name='trip_dis_ave')
nnst_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_nn'].mean().reset_index(name='nnst_dis_ave')

polygons_with_ave = polygons.merge(trip_dis_ave)
polygons_with_ave[['trip_dis_ave']] = polygons_with_ave[['trip_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[1]+' - visited', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave)
polygons_with_ave[['nnst_dis_ave']] = polygons_with_ave[['nnst_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='nnst_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[1]+' - nearest', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave).merge(trip_dis_ave)
polygons_with_ave['diff_dis_ave'] = polygons_with_ave['trip_dis_ave'] - polygons_with_ave['nnst_dis_ave']
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='diff_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(types[1]+' - difference', fontsize=15)
ax.axis('off')
plt.show()








food_related_stop_df["home_retail_dis_vs"] = food_related_stop_df["type3_mean"]
food_related_stop_df["home_retail_dis_nn"] = food_related_stop_df["distance_3"]

df_user_home_wm_short = df_user_home_wm[["user_id",'GEOID20']]
polygons = df_acs_duval_wm

trip_store_dis_df = food_related_stop_df.merge(df_user_home_wm_short, left_on='user_id', right_on='user_id', how='left')
trip_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_vs'].mean().reset_index(name='trip_dis_ave')
nnst_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_nn'].mean().reset_index(name='nnst_dis_ave')

polygons_with_ave = polygons.merge(trip_dis_ave)
polygons_with_ave[['trip_dis_ave']] = polygons_with_ave[['trip_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[2]+' - visited', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave)
polygons_with_ave[['nnst_dis_ave']] = polygons_with_ave[['nnst_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='nnst_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[2]+' - nearest', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave).merge(trip_dis_ave)
polygons_with_ave['diff_dis_ave'] = polygons_with_ave['trip_dis_ave'] - polygons_with_ave['nnst_dis_ave']
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='diff_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(types[2]+' - difference', fontsize=15)
ax.axis('off')
plt.show()










food_related_stop_df["home_retail_dis_vs"] = food_related_stop_df["type4_mean"]
food_related_stop_df["home_retail_dis_nn"] = food_related_stop_df["distance_4"]

df_user_home_wm_short = df_user_home_wm[["user_id",'GEOID20']]
polygons = df_acs_duval_wm

trip_store_dis_df = food_related_stop_df.merge(df_user_home_wm_short, left_on='user_id', right_on='user_id', how='left')
trip_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_vs'].mean().reset_index(name='trip_dis_ave')
nnst_dis_ave = trip_store_dis_df.groupby('GEOID20')['home_retail_dis_nn'].mean().reset_index(name='nnst_dis_ave')

polygons_with_ave = polygons.merge(trip_dis_ave)
polygons_with_ave[['trip_dis_ave']] = polygons_with_ave[['trip_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='trip_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[3]+' - visited', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave)
polygons_with_ave[['nnst_dis_ave']] = polygons_with_ave[['nnst_dis_ave']].fillna(0)
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='nnst_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True, vmin=0, vmax=30)
plt.title(types[3]+' - nearest', fontsize=15)
ax.axis('off')
plt.show()

polygons_with_ave = polygons.merge(nnst_dis_ave).merge(trip_dis_ave)
polygons_with_ave['diff_dis_ave'] = polygons_with_ave['trip_dis_ave'] - polygons_with_ave['nnst_dis_ave']
fig, ax = plt.subplots(figsize=(8, 6))
ax = polygons_with_ave.plot(ax=ax, column='diff_dis_ave', cmap='bone_r', linewidth=0, edgecolor='black', legend=True)
plt.title(types[3]+' - difference', fontsize=15)
ax.axis('off')
plt.show()

















from scipy.stats import ks_2samp
import numpy as np

data1 = his_1_nn
data2 = his_2_nn
data3 = his_3_nn
data4 = his_4_nn

# Perform pairwise KS tests
datasets = [data1, data2, data3, data4]

for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        ks_statistic, ks_p_value = ks_2samp(datasets[i], datasets[j])
        print(f"KS Statistic between data{i + 1} and data{j + 1}: {ks_statistic}")
        print(f"P-value: {ks_p_value}")
        print()

# Interpret the results
alpha = 0.1
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        if ks_p_value < alpha:
            print(f"The distributions between nn distance of type{i + 1} and type{j + 1} are different (reject the null hypothesis)")
        else:
            print(f"The distributions between nn distance of type{i + 1} and type{j + 1} are not significantly different (fail to reject the null hypothesis)")



data1 = his_1
data2 = his_2
data3 = his_3
data4 = his_4

# Perform pairwise KS tests
datasets = [data1, data2, data3, data4]

for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        ks_statistic, ks_p_value = ks_2samp(datasets[i], datasets[j])
        print(f"KS Statistic between data{i + 1} and data{j + 1}: {ks_statistic}")
        print(f"P-value: {ks_p_value}")
        print()

# Interpret the results
alpha = 0.1
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        if ks_p_value < alpha:
            print(f"vt distance of type{i + 1} and type{j + 1} are different (reject the null hypothesis)")
        else:
            print(f"vt distance of type{i + 1} and type{j + 1} are not significantly different (fail to reject the null hypothesis)")






from scipy.stats import anderson_ksamp

data1 = his_1
data2 = his_2
data3 = his_3
data4 = his_4

# Perform pairwise Anderson-Darling tests
datasets = [data1, data2, data3, data4]

for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        ad_statistic, critical_values, significance_level = anderson_ksamp([datasets[i], datasets[j]])
        print(f"AD Statistic between data{i + 1} and data{j + 1}: {ad_statistic}")
        print(f"Critical Values: {critical_values}")
        print(f"Significance Level: {significance_level}")
        print()

# Interpret the results
alpha = 0.1
for i in range(len(datasets)):
    for j in range(i + 1, len(datasets)):
        if ad_statistic > critical_values[2]:  # Compare with the critical value at 5% significance level
            print(f"The distributions between data{i + 1} and data{j + 1} are different (reject the null hypothesis)")
        else:
            print(f"The distributions between data{i + 1} and data{j + 1} are not significantly different (fail to reject the null hypothesis)")
            
            
            


import numpy as np

print( np.mean(his_1))
print( np.var(his_1))

print( np.mean(his_2))
print( np.var(his_2))

print( np.mean(his_3))
print( np.var(his_3))

print( np.mean(his_4))
print( np.var(his_4))


print( np.mean(his_1_nn))
print( np.var(his_1_nn))

print( np.mean(his_2_nn))
print( np.var(his_2_nn))

print( np.mean(his_3_nn))
print( np.var(his_3_nn))

print( np.mean(his_4_nn))
print( np.var(his_4_nn))





data1 = his_1
data2 = his_2
data3 = his_3
data4 = his_4

data1_n = his_1_nn
data2_n = his_2_nn
data3_n = his_3_nn
data4_n = his_4_nn

# Perform pairwise Anderson-Darling tests
datasets = [data1,data1_n, data2, data2_n, data3, data3_n, data4, data4_n]
colors = ['blue', 'blue', 'green', 'green', 'grey', 'grey', 'red', 'red']

plt.subplots(figsize=(10, 4))
sns.boxplot(datasets, orient='h', palette=colors)