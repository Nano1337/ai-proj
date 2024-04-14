# function to represent the regression tree for calculating utility of a location 

import numpy as np 

'''
calc_location_utility: function to calculate the utility of a location based on the regression tree

For this regression tree, we will use the following rules:

- if there is no park theme: 
    - if there is no walking theme, the utility is 0.1 
    - if there is a walking theme, the utility is 0.3 
- if there is a park theme, we will check
    - if there is not a walking theme, the utility is 0.4
    - if there is a walking theme, the utility is 0.8 (complement effect: park and walking > park only + walking only)

- utility += 0.1*history + 0.1*music (additive)

- if there is both environment and music, utility -= 0.1 (substitution effect: environment and music < environment only + music only)

- park is index 3, walking is index 9, history is index 0, music is index 1, environment is index 5

params: 
- location_vector: the feature vector for a location. This is a numpy array of size 1 x n 
where n is the number of themes in the dataset. 
'''
class Node:
    def __init__(self, condition=None, true_branch=None, false_branch=None, value=None):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.value = value

# Define conditions
def has_park(location_vector):
    return location_vector[3] >= 1

def has_walking(location_vector):
    return location_vector[9] >= 1

def has_history(location_vector):
    return location_vector[0] >= 1

def has_music(location_vector):
    return location_vector[1] >= 1

def has_environment(location_vector):
    return location_vector[5] >= 1

def has_environment_and_music(location_vector):
    return has_environment(location_vector) and has_music(location_vector)

# Build tree
tree = Node(
    condition=has_park,
    true_branch=Node(
        condition=has_walking,
        true_branch=Node(
            condition=has_history,
            true_branch=Node(
                condition=has_music,
                true_branch=Node(
                    condition=has_environment,
                    true_branch=Node(value=0.8 + 0.1 + 0.1 - 0.1),  # park, walking, history, music, environment
                    false_branch=Node(value=0.8 + 0.1 + 0.1)  # park, walking, history, music, no environment
                ),
                false_branch=Node(value=0.8 + 0.1)  # park, walking, history, no music
            ),
            false_branch=Node(value=0.7)  # park, walking, no history
        ),
        false_branch=Node(value=0.4)  # park, no walking
    ),
    false_branch=Node(
        condition=has_walking,
        true_branch=Node(value=0.3),  # no park, walking
        false_branch=Node(value=0.1)  # no park, no walking
    )
)

def calc_location_utility(location_vector, location_name, inclusion_list):
    # regression tree for calculating utility of a location 
    utility = 0
    if location_name in inclusion_list:
        utility += 10

    # Traverse tree
    node = tree
    while node.value is None:
        if node.condition(location_vector):
            node = node.true_branch
        else:
            node = node.false_branch
    utility += node.value
    
    return utility