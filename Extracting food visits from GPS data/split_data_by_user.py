
import pandas as pd
import os
from tqdm import tqdm
import datetime
from shapely.geometry import Point
from shapely import wkt


input_folder = input_folder

radius = 'retail_200'

for i in range(1, 7):
    
    pfs_path = os.path.join(input_folder, 'Jackson_pfs', f'pfs{i}.csv')
    stop_df_path = os.path.join(input_folder, 'Jackson_retail',radius, f'food_{i}.csv')
    
    pfs_df = pd.read_csv(pfs_path)
    food_sp_df = pd.read_csv(stop_df_path)
    
    for user_id in food_sp_df['user_id'].unique():
        user_data = pfs_df[pfs_df['user_id'] == user_id]
        user_data.to_csv( os.path.join(input_folder, 'Jackson_pfs',radius,'by_user',f'user_{user_id}_pfs.csv'), index=False)
    
    for user_id in food_sp_df['user_id'].unique():
        user_data = food_sp_df[food_sp_df['user_id'] == user_id]
        user_data.to_csv(os.path.join(input_folder, 'Jackson_retail',radius,'by_user',f'user_{user_id}_food_sp.csv'), index=False)   
        
