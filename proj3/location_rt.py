# function to represent the regression tree for calculating utility of a location 

import numpy as np 

'''
location_utility: function to calculate the utility of a location based on the regression tree

For this regression tree, we will use the following rules:

- if there is no park theme: 
    - if there is no walking theme, the utility is 0.1 
    - if there is a walking theme, the utility is 0.3 
- if there is a park theme, we will check
    - if there is not a walking theme, the utility is 0.4
    - if there is a walking theme, the utility is 0.6 (complement effect: park and walking > park only + walking only)

- utility += 0.1*history + 0.1*music (additive)

- if there is both environment and music, utility -= 0.1 (substitution effect: environment and music < environment only + music only)

- park is index 3, walking is index 9, history is index 0, music is index 1, environment is index 5

params: 
- location_vector: the feature vector for a location. This is a numpy array of size 1 x n 
where n is the number of themes in the dataset. 
'''
def location_utility(location_vector):
    # regression tree for calculating utility of a location 
    utility = 0

    # First rule: parks and walking (complement effect)
    if location_vector[3] == 0:
        if location_vector[9] == 0:
            utility = 0.1
        else:
            utility = 0.3
    else:
        if location_vector[9] == 0:
            utility = 0.4
        else:
            utility = 0.6
    
    # Second rule: history and music (additive)
    utility += 0.1*location_vector[0] + 0.1*location_vector[1]

    # Third rule: environment and music (substitution effect)
    if location_vector[5] == 1 and location_vector[1] == 1:
        utility -= 0.1
    
    return utility 