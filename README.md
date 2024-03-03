# Potentials and limitations of large-scale mobile location GPS data for food access analysis

### Introduction 
Key codes used in the submitted paper *Potentials and limitations of large-scale mobile location GPS data for food access analysis*. 

The data that support the findings of this study are available from Gravy Analytics. Restrictions apply to the availability of these data, which were used under license for this study. Aggregate-level data are available from the authors with the permission of Gravy Analytics per the data agreement with Gravy Analytics. We attached the corresponding data structure of the input and output of each script. 

A workflow of the code is presented in the graph (Workflow_and_code.png).

### Code files
####  1_stop_inference.py
Extract staypoints, positionfix linked with the staypoint and triplegs from GPS data.

- Input: 

    - GPS data provided by Gravy Analytics (*https://gravyanalytics.com/data-as-a-service/*). Example:

            grid,latitude,longitude,time,user_id,week,geometry
            001eafd2-aa33-34c0-a022-26af9761ed22,30.35627,-81.54973,2022-09-03 04:24:38,2,0,POINT (-9078000.000000000 3549000.000000000)
            001eafd2-aa33-34c0-a022-26af9761ed22,30.35628,-81.54974,2022-09-03 04:24:48,2,0,POINT (-9078000.000000000 3549000.000000000)
            
- Output:

     - *Trackintel* framework components. 
     *For more information: https://trackintel.readthedocs.io/en/latest/index.html*

    
    - staypoint: 
    
            user_id,started_at,finished_at,geom id	
    - positionfix: 	
    
            user_id,flag,tracked_at,geom,staypoint_id,tripleg_id ID
    - tripleg: 
    
            user_id,started_at,finished_at,geom,id	



####  2_home_inference.py
Based on the GPS data, infer device users' home locations. 

Code based on algorithm by  

    *Zhao, X., Xu, Y., Lovreglio, R., Kuligowski, E., Nilsson, D., Cova, T.J., Wu, A., and Yan, X., 2022. Estimating wildfire evacuation decision and departure timing using large-scale GPS data. Transportation research part D: transport and environment, 107, 103277.*

Contains two functions for inferring home locations from GPS point:

    1. Function1:
    
        input: each user's GPS, nighttime start and end hour.
        
        output: users home location inferred from nighttime stay
                 location where user generated most GPS during nighttime .
    
    2. Function2:
    
        input: each user's GPS.
    
        output: users home location inferred from weekend stay
                 location where user generated most GPS during weekends

- Input: 
    - GPS data (fields see above)
    
- Output: 
    - Inferred home location for each device user
    
            user_id,home_lat,home_lon


####  3_food_stop_extraction.py
In this study, we used the North Florida food retailer location database by Erik Finlay at the University of Florida *GeoPlan* Center. For a quick look at the layers in the dataset: *https://services.arcgis.com/LBbVDC0hKPAnLRpO/arcgis/rest/services/ACFS_Map/FeatureServer*

Based on the extracted staypoints from *1_stop_inference.py* and the food retailer location, 
1. extract food related stops based on distance threshold
2. split the stops by users, later used as input for *4_food_trip_extraction.py*
3. filter the stops, limiting 2-hour visit duration
- Input:
    - stay points (fields see above)
    - retail locations, reclassified from the input dataset and added a new TYPE field (Figure 4., Table 1. in submission): 
        
            ObjectId,Store_Name,Longitude,Latitude,TYPE
    - threshold: numeric value, in meter. From all stay points, find out the stay points within the threshold of the food retails. We infer that the user has visited the location.

- Output:
    - food-related stops (Table 2. in submission)
            
            id,user_id,started_at,finished_at,lat,lon,retail_id,retail_lat,retail_lon


####  4_food_trip_extraction.py
Based on 
1. the extracted triplegs from *1_stop_inference.py*, and 
2. the food stop extracted from *3_food_stop_extraction.py*,

generate food trips. To do this, we used a sliding window method by setting the staying thresholds to a space of 100-meter radius, and a duration of at least 5 minutes and at most 720 minutes.

- Input: 
    - triplegs (fields see above)
    - food-related stops (fields see above)
    - time window thresholds (explanation see above)

- Output: 
    - food-related trips (Table 2. in submission)
    
            user_id,tripleg_ID,trip_started_at,trip_finished_at,trip,retail_id,retail_lat,retail_lon

####  5_metric_calculation.py

Based on 
1. the user home location inferred with *2_home_inference.py*
2. the retail location data
3. the extracted food-related stops from *3_food_stop_extraction.py*
2. the extracted food-related trips from *4_food_trip_extraction.py*

calculate the four food beahvior metrics at individual level.
- number of visit
- number of unique stores visited
- home-to-store distance (visited v.s. nearest)
- proportion of home-based visits

(Table 3., 4. and Figure 14. in sumission. See submission for detailed explanation of the metrics.)



####  6_visual_temporal.py
Based on the four food beahvior metrics calculated in *5_metric_calculation.py*, generate line plots (curves) of
1. daily number of stops (Figure 11. in submission)
2. day of week pattern (Figure 10.3 in submission)
3. time of day pattern (Figure 10.1 and 10.2 in submission)


####  7_visualization_spatial.py
Based on the four food beahvior metrics calculated in *5_metric_calculation.py*, calculate the tract-level aggregated average, and plot spatial distribution maps.
 
- Input: 
    - calculated individual metrics (see *5_metric_calculation.py*)
    - home location info (see *2_home_inference.py*)
    - study area tracts shapefile (from US Census)

- Output: 
    - spatial distribution maps of tract-level metrics (Figure 9 in submission)
    - other related figures (Figure 7., 8. and 13. in submission)
    
####  8_visualization_input.py

Plot the non-metrics-related datasets in maps.

- Input: 
    - study area shapefile and ACS data
    - inferred user home location
    - food-related stops
    
- Output: 
    - sociodemographic characteristics maps (Figure 3. in submission)
    - sampling rate map (Figure 5. in submission)
