"""
based on the extracted food-related stops, plot temporal pattern

Input: 
    food-related stops

Output: 
    line plots (curves) of
    1. daily number of stops
    2. day of week pattern
    3. time of day pattern
    
"""



import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt


# read calculated results
# radius
rad = 200 
# types of stores
type_1 = 'Large Groceries'
type_2 = 'Big Box Stores'
type_3 = 'Small Healthy Outlets'
type_4 = 'Processed Food Outlets'
the_index = [type_1,type_2,type_3,type_4]
# read the and format food stops, with user info and retailer info joined
food_related_stop = pd.read_csv('food_sp_limit2h_{rad}_merged.csv')
food_related_stop['started_at'] = pd.to_datetime(food_related_stop['started_at']).dt.tz_localize(None)
food_related_stop['started_at_tm'] = pd.to_datetime(food_related_stop['started_at'])
food_related_stop['started_at_hour'] = food_related_stop['started_at_tm'].dt.hour
food_related_stop['started_at_date'] = food_related_stop['started_at_tm'].dt.date



# 1. daily visits
# calculate daily visits in the stop dataset
grouped_data = food_related_stop.groupby(['TYPE', pd.Grouper(key='started_at_tm', freq='D')])['id'].count().unstack(level=0)
fig = plt.figure(figsize=(16,10))
grouped_data.plot(ax=plt.gca())
# subset the data by type
food_related_stop_1 = food_related_stop[food_related_stop['TYPE'] == 1]
grouped_data_1 = food_related_stop_1[['started_at_tm','id']].groupby(pd.Grouper(key='started_at_tm', freq='D')).count()
grouped_data_1.index = grouped_data_1.index.date
food_related_stop_2 = food_related_stop[food_related_stop['TYPE'] == 2]
grouped_data_2 = food_related_stop_2[['started_at_tm','id']].groupby(pd.Grouper(key='started_at_tm', freq='D')).count()
grouped_data_2.index = grouped_data_2.index.date
food_related_stop_3 = food_related_stop[food_related_stop['TYPE'] == 3]
grouped_data_3 = food_related_stop_3[['started_at_tm','id']].groupby(pd.Grouper(key='started_at_tm', freq='D')).count()
grouped_data_3.index = grouped_data_3.index.date
food_related_stop_4 = food_related_stop[food_related_stop['TYPE'] == 4]
grouped_data_4 = food_related_stop_4[['started_at_tm','id']].groupby(pd.Grouper(key='started_at_tm', freq='D')).count()
grouped_data_4.index = grouped_data_4.index.date
# plot
fig = plt.figure(figsize=(16,10))
plt.plot(grouped_data_1, label=type_1)
plt.plot(grouped_data_2, label=type_2)
plt.plot(grouped_data_3, label=type_3)
plt.plot(grouped_data_4, label=type_4)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Trip counts')
plt.title('Trip counts by Day')
plt.show()


# 2. time of day (weekends and weekdays)
food_related_stop_weekday = food_related_stop_df[food_related_stop_df['started_at_day']<=5]
tod_1 = food_related_stop_weekday[food_related_stop_weekday['TYPE'] == 1]
tod_1.set_index('started_at_date', inplace=False)
tod_1_grouped = tod_1['started_at_hour'].value_counts()
tod_1_grouped = tod_1_grouped/tod_1_grouped.sum()
new_index_order = range(24) 
tod_1_grouped = tod_1_grouped.reindex(new_index_order)

tod_2 = food_related_stop_weekday[food_related_stop_weekday['TYPE'] == 2]
tod_2.set_index('started_at_date', inplace=False)
tod_2_grouped = tod_2['started_at_hour'].value_counts()
tod_2_grouped = tod_2_grouped/tod_2_grouped.sum()
new_index_order = range(24) 
tod_2_grouped = tod_2_grouped.reindex(new_index_order)

tod_3 = food_related_stop_weekday[food_related_stop_weekday['TYPE'] == 3]
tod_3.set_index('started_at_date', inplace=False)
tod_3_grouped = tod_3['started_at_hour'].value_counts()
tod_3_grouped = tod_3_grouped/tod_3_grouped.sum()
new_index_order = range(24) 
tod_3_grouped = tod_3_grouped.reindex(new_index_order)

tod_4 = food_related_stop_weekday[food_related_stop_weekday['TYPE'] == 4]
tod_4.set_index('started_at_date', inplace=False)
tod_4_grouped = tod_4['started_at_hour'].value_counts()
tod_4_grouped = tod_4_grouped/tod_4_grouped.sum()
new_index_order = range(24) 
tod_4_grouped = tod_4_grouped.reindex(new_index_order)

plt.figure(figsize=(8,5))
plt.plot(tod_1_grouped, marker='.', label=type_1)
plt.plot(tod_2_grouped, marker='.', label=type_2)
plt.plot(tod_3_grouped, marker='.', label=type_3)
plt.plot(tod_4_grouped, marker='.', label=type_4)
plt.legend()
plt.xlabel('Time of Day')
plt.ylabel('Distribution')
plt.title('Time of day pattern (weekday)')
plt.show()



# 3. day of week
dow_1 = food_related_stop[food_related_stop['TYPE'] == 1]
dow_1.set_index('started_at_day', inplace=False)
dow_1_grouped = dow_1['started_at_day'].value_counts()
dow_1_grouped = dow_1_grouped/dow_1_grouped.sum()
new_index_order = range(1,8) 
dow_1_grouped = dow_1_grouped.reindex(new_index_order)

dow_2 = food_related_stop[food_related_stop['TYPE'] == 2]
dow_2.set_index('started_at_day', inplace=False)
dow_2_grouped = dow_2['started_at_day'].value_counts()
dow_2_grouped = dow_2_grouped/dow_2_grouped.sum()
new_index_order = range(1,8) 
dow_2_grouped = dow_2_grouped.reindex(new_index_order)

dow_3 = food_related_stop[food_related_stop['TYPE'] == 3]
dow_3.set_index('started_at_day', inplace=False)
dow_3_grouped = dow_3['started_at_day'].value_counts()
dow_3_grouped = dow_3_grouped/dow_3_grouped.sum()
new_index_order = range(1,8) 
dow_3_grouped = dow_3_grouped.reindex(new_index_order)

dow_4 = food_related_stop[food_related_stop['TYPE'] == 4]
dow_4.set_index('started_at_day', inplace=False)
dow_4_grouped = dow_4['started_at_day'].value_counts()
dow_4_grouped = dow_4_grouped/dow_4_grouped.sum()
new_index_order = range(1,8) 
dow_4_grouped = dow_4_grouped.reindex(new_index_order)

plt.figure(figsize=(8,5))
plt.plot(dow_1_grouped, marker='.', label=type_1)
plt.plot(dow_2_grouped, marker='.', label=type_2)
plt.plot(dow_3_grouped, marker='.', label=type_3)
plt.plot(dow_4_grouped, marker='.', label=type_4)
plt.ylim(0.1, 0.2)
plt.legend()
plt.xlabel('Day of Week')
plt.ylabel('Distribution')
plt.title('Day of week pattern')
plt.show()


