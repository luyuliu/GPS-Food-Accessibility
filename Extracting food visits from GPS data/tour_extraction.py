import pandas as pd
import os
from tqdm import tqdm
import datetime
from shapely.geometry import Point
from shapely import wkt
from datetime import timedelta


input_folder = input_folder
retail_folder = retail_folder
input_sp_folder = input_sp_folder
input_tplg_folder = input_tplg_folder


output_folder = output_folder


radius = 'retail_200'




#files = os.listdir(os.path.join(retail_folder, radius , 'by_user'))
files = os.listdir(os.path.join(input_tplg_folder))

# Iterate through the files
for file_name in tqdm(files, desc='Processing Files', unit='file'):
    
    
    #user_id = 1025741
    user_id = int(file_name.split('_')[1])
    
    tplg_name = f'user_{user_id}_tplg.csv'
    tripleg_path = os.path.join(input_tplg_folder, tplg_name)
    sp_name = f'user_{user_id}_sp.csv'
    stop_df_path = os.path.join(input_sp_folder, sp_name)
    food_sp_name = f'user_{user_id}_food_sp_limit2h.csv'
    food_sp_df_path = os.path.join(retail_folder, radius , 'by_user_limit2h', food_sp_name)

    try:
        tripleg = pd.read_csv(tripleg_path)
        stop_df = pd.read_csv(stop_df_path)
        food_sp_df = pd.read_csv(food_sp_df_path)
        
    except Exception as e:
        print(f"An unexpected error occurred for user_id {user_id}: {e}")
        continue


    
    stop_df['started_at'] = pd.to_datetime(stop_df['started_at']).dt.tz_localize(None) + timedelta(hours=-4)
    stop_df['finished_at'] = pd.to_datetime(stop_df['finished_at']).dt.tz_localize(None) + timedelta(hours=-4)
    tripleg['started_at'] = pd.to_datetime(tripleg['started_at']).dt.tz_localize(None) + timedelta(hours=-4)
    tripleg['finished_at'] = pd.to_datetime(tripleg['finished_at']).dt.tz_localize(None) + timedelta(hours=-4)
    food_sp_df['started_at'] = pd.to_datetime(food_sp_df['started_at']).dt.tz_localize(None) + timedelta(hours=-4)
    food_sp_df['finished_at'] = pd.to_datetime(food_sp_df['finished_at']).dt.tz_localize(None) + timedelta(hours=-4)
    
    stop_df['continuous_check'] = ((stop_df['user_id'] == stop_df['user_id'].shift()) & (stop_df['started_at'] == stop_df['finished_at'].shift())).astype(int)

    ID_list=food_sp_df['id'].unique()
    
    def find_trip_before(stay_point_id, time_window = 15):
        time_window = datetime.timedelta(minutes=time_window)
        sp = stop_df[stop_df['id'] == stay_point_id]
        stop_of_user = stop_df[(stop_df['user_id'] == sp.iloc[0]['user_id']) & (stop_df['id'] <= stay_point_id)]
        related_tripled_by_user = tripleg[tripleg['user_id'] == sp.iloc[0]['user_id']]
        
        trip_traveled = []
        stop_point_traveled = []
        # temp_stop_list = []

        i = len(stop_of_user) - 1
        while (i >= 0 and stop_of_user.iloc[i]['continuous_check'] == 1):
            stop_point_traveled.insert(0, stop_of_user.iloc[i]['id'])
            # temp_stop_list.insert(0, stop_of_user.iloc[i]['id'])
            i -= 1
        # if len(temp_stop_list) == 0: temp_stop_list.append(stay_point_id)

        while i >= 0:

            tripled_found = related_tripled_by_user[
                (related_tripled_by_user['finished_at'] > stop_of_user.iloc[i]['started_at'] - time_window) &
                (related_tripled_by_user['finished_at'] <= stop_of_user.iloc[i]['started_at'])]
            
            #try to find trip
            if len(tripled_found) > 0:
                if len(tripled_found) > 1: tripled_found = tripled_found.nlargest(1, 'finished_at')
                trip_traveled.insert(0, tripled_found.iloc[0]['id'])
            #in this case, no trip or stop point were found, end program.
            elif i - 1 >= 0 and stop_of_user.iloc[i]['started_at'] - stop_of_user.iloc[i - 1]['finished_at'] > time_window: 
                stop_point_traveled.insert(0, stop_of_user.iloc[i]['id'])
                break
            
            # if len(temp_stop_list) == 0: temp_stop_list.insert(0, stop_of_user.iloc[i]['id'])
            # temp_stop_list = []
            stop_point_traveled.insert(0, stop_of_user.iloc[i]['id'])

            i -= 1
            if len(tripled_found) > 0 and tripled_found.iloc[0]['started_at'] != stop_of_user.iloc[i]['finished_at']: break

            while i >= 0 and stop_of_user.iloc[i]['continuous_check'] == 1:
                # temp_stop_list.insert(0, stop_of_user.iloc[i]['id'])
                stop_point_traveled.insert(0, stop_of_user.iloc[i]['id'])
                i -= 1
                
        if len(trip_traveled) == 0: return None
        fist_trip = related_tripled_by_user[related_tripled_by_user['id'] == trip_traveled[0]]
        data_to_append = {'user_id': sp.iloc[0]['user_id'], 'tripleg_ID': trip_traveled, 
                            'trip_start_location': Point(list(wkt.loads(fist_trip.iloc[0]['geom']).coords)[0]),
                            'trip_end_location': sp.iloc[0]['geom'],
                            'trip_start_timestamp': fist_trip.iloc[0]['started_at'],
                            'trip_end_timestamp': sp.iloc[0]['finished_at'],
                            'retail_id': food_sp_df.loc[food_sp_df['id'] == stay_point_id, 'retail_id'].values[0],
                            #'retail_tpye': food_sp_df.loc[food_sp_df['id'] == stay_point_id, 'retail_type'].values[0],
                            'stop_point_between_trips': stop_point_traveled}

        return data_to_append

    data_to_generate = []
    for sp_id in ID_list:
        res = find_trip_before(sp_id)
        if res != None:
            data_to_generate.append(res)
    output_df = pd.DataFrame.from_records(data_to_generate)

    output_file_path = os.path.join(output_folder, radius, f'user_{user_id}_food_tour_limit2h.csv')
    output_df.to_csv(output_file_path, index=False)



