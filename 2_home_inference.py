"""
infer home locations of users. 
two functions for inferring home locations from GPS point, algorithm adopted from Zhao et al. 2022
Input: GPS(lat lon timestamp user_id)
Output: user_id	home_lat home_lon

-Function1:
    input: each user's GPS, nighttime start and end hour
    output: users home location inferred from nighttime stay
             location where user generated most GPS during nighttime 
-Function2:
    input: each user's GPS
    output: users home location inferred from weekend stay
             location where user generated most GPS during weekends
"""

# import the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time as T
import pyproj

# read the GPS data
df = pd.read_csv('data.csv')
# read the staypoints data infered through stop_inference.py
sp = pd.read_csv('staypoints.csv')

# get user id list of the users with staypoints
ID_list=sp['user_id'].unique()
len(ID_list)
# filter the gps point for those users
ndf = df[df['user_id'].isin(ID_list)]
ndf.to_csv('newData.csv')


# reproject the data from EPSG 4326 (WGS84) to EPSG 3857 (Web Mercator)
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
for idx, row in tqdm(ndf.iterrows(), total=ndf.shape[0]):
    lat, lon = row['latitude'], row['longitude']
    x, y = transformer.transform(lon, lat)
    ndf.at[idx, 'Lat-3857'] = y
    ndf.at[idx, 'Lon-3857'] = x



# infer home locations, based on nighttime GPS
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def home_location(input_df,s_hour=22,t_hour=6):
    '''
    ndf: gps data
    s_hour: define start time of night, defult = 22
    t_hour: define end time of night, defult = 6
    '''
    input_df['time']=pd.to_datetime(input_df['time'], infer_datetime_format=True)
    input_df['date']=input_df['time'].dt.date
    input_df['hour']=pd.Series(input_df['time']).dt.hour
    
    input_df['LAT_Grid']=np.round(input_df['Lat-3857']/20)*20  # Grid size = 20m
    input_df['LON_Grid']=np.round(input_df['Lon-3857']/20)*20
    id=input_df.iloc[0,0]
    input_df=input_df[(input_df['hour']>=s_hour)|(input_df['hour']<t_hour)]

    if len(input_df)==0:  # if no signal during the night, return NaN
        return np.nan,np.nan
    home = input_df.groupby(['LAT_Grid','LON_Grid'])['date'].nunique().idxmax()
    return home[0],home[1]

# call the home_location function and store the home inferred in df home
home=pd.DataFrame(columns=['ID','LAT','LON'])
for i in tqdm(ID_list[0:]): 
    dffh=ndf[ndf['user_id']==i].sort_values('time',axis=0,ascending=True)
    h_lat,h_lon=home_location(dffh)
    home=home.append(pd.DataFrame([[i,h_lat,h_lon]],columns=['ID','LAT','LON']))
    
    
    
    
    
# Further adjust to infer home location for more users for weekends

#check if a date is in saturday or sunday.
def is_weekend(date):
    return date.weekday() in [5, 6]

def home_location2(input_df):

    input_df['time']=pd.to_datetime(input_df['time'], infer_datetime_format=True)
    input_df['date']=input_df['time'].dt.date
    input_df['hour']=pd.Series(input_df['time']).dt.hour
    
    input_df['LAT_Grid']=np.round(input_df['Lat-3857']/20)*20
    input_df['LON_Grid']=np.round(input_df['Lon-3857']/20)*20
    input_df= input_df[input_df['date'].apply(is_weekend)]

# call the home_location2 function and store the home inferred in df home2
home2=pd.DataFrame(columns=['ID','LAT','LON'])
for i in tqdm(ID_list[0:]): 
    dffh=ndf[ndf['user_id']==i].sort_values('time',axis=0,ascending=True)
    h_lat,h_lon=home_location2(dffh)
    home2=home2.append(pd.DataFrame([[i,h_lat,h_lon]],columns=['ID','LAT','LON']))
    if len(input_df)==0: 
        return np.nan,np.nan
    home2 = input_df.groupby(['LAT_Grid','LON_Grid']).count()['ID'].idxmax()
    return home2[0],home2[1]

home2.dropna(subset=['LAT'], inplace=True)



# combine the two home data
new_home = pd.concat([home, home2], ignore_index=True)



# convert from EPSG 3857 (Web Mercator) back to EPSG 4326 (WGS84)
inputGrid = pyproj.Proj(projparams='epsg:3857')
wgs84 = pyproj.Proj(projparams='epsg:4326')

new_home = new_home.reset_index(drop=True)

for idx, row in tqdm(new_home.iterrows(), total=new_home.shape[0]):
    lat, lon = pyproj.transform(inputGrid, wgs84, row['LON'], row['LAT'])
    new_home.at[idx, 'LAT-4326'] = lat
    new_home.at[idx, 'LON-4326'] = lon

new_home.to_csv('home_location.csv')