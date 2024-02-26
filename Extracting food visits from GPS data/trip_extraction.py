import pandas as pd
import os
from tqdm import tqdm
import datetime
from datetime import timedelta
import numpy as np

retail_folder = retail_folder


radius = 'retail_200'


files = os.listdir(os.path.join(retail_folder, radius , 'by_user'))

# Iterate through the files
for file_name in tqdm(files, desc='Processing Files', unit='file'):
    
    
    user_id = int(file_name.split('_')[1])
    
    food_sp_name = f'user_{user_id}_food_sp.csv'
    food_sp_df_path = os.path.join(retail_folder, radius , 'by_user', food_sp_name)
    food_sp_df = pd.read_csv(food_sp_df_path)
    
    food_sp_df['started_at_tm'] = pd.to_datetime(food_sp_df['started_at']).dt.tz_localize(None) + timedelta(hours=-4)
    food_sp_df['finished_at_tm'] = pd.to_datetime(food_sp_df['finished_at']).dt.tz_localize(None) + timedelta(hours=-4)
    food_sp_df['duration_h'] = (food_sp_df['finished_at_tm']-food_sp_df['started_at_tm']) / np.timedelta64(1, 'h')
    food_sp_df = food_sp_df[(food_sp_df['duration_h']<=2)]
    
    if len(food_sp_df)>0:
        output_file_path = os.path.join(retail_folder, radius , 'by_user_limit2h', f'user_{user_id}_food_sp_limit2h.csv')
        food_sp_df.to_csv(output_file_path, index=False)

