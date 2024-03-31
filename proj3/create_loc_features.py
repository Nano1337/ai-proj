# Create the feature vector for each location in the dataset 
import pandas as pd 
import numpy as np

def generate_location_vector():
    attractions_df = pd.read_csv('proj3/road_network_attractions.csv')
    themes_df = pd.read_csv('proj3/road_network_themes.csv')

    themes_list = themes_df.iloc[:, 0].tolist()
    themes_list = [theme.replace(" ", "") for theme in themes_list]
    feature_dict = dict()
    #print(themes_list)

    for index, row in attractions_df.iterrows():
        location = row['Loc or Edge Label']
        if location not in feature_dict:
            feature_dict[location] = np.zeros(len(themes_list))

        try:
            themes = row['Themes'].split(',')
            themes = [theme.strip() for theme in themes]
            for theme in themes:
                feature_dict[location][themes_list.index(theme)] += 1
        except:
            continue 
    
    return feature_dict 

# find the most common theme in the dataset

vect = generate_location_vector()
themes = np.zeros((1, len(vect["NashvilleTN"])))

for key in vect:
    themes += vect[key]

print(themes)