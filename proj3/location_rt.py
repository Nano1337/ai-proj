# function to represent the regression tree for calculating utility of a location 

import numpy as np 

'''
location_utility: function to calculate the utility of a location based on the regression tree

For this regression tree, we will use the following rules:
- if there is no history theme: 
    - if there is no food theme, the utility is 0.1 
    - if there is a food theme, the utility is 0.3 
- if there is a history theme, we will check
    - if there is no food theme, the utility is 0.5
    - if there is a food theme, the utility is 0.7

params: 
- location_vector: the feature vector for a location. This is a numpy array of size 1 x n 
where n is the number of themes in the dataset. 
'''
def location_utility(location_vector):
    # regression tree for calculating utility of a location 

    if location_vector[0] < 1: # if there is no history theme 
        if location_vector[1] < 1:
            return 0.5
        else:
            return 0.6
    else: # there is history theme 
        if location_vector[2] < 1:
            return 0.7
        else:
            return 0.8
