import pandas as pd
import numpy as np

# global variable: edge_map is a dictionary mapping out the edges as sets to their distances 
edge_map = dict() 
locations = list()
loc_prefs= {} 
edge_prefs = {}

# Assign preference values between 0 and 1 for each location
def location_preference_assignments(a, b):
    global locations, loc_prefs
    loc_prefs = {}
    for loc in locations:
        loc_prefs[loc] = np.random.uniform(a, b)
    return loc_prefs

#Assigns random values between a=0 and b=0.1 inclusive using a uniform distribution to each edge independently
def edge_preference_assignments(a=0, b=0.1):
    global edge_map, edge_prefs
    edge_prefs = {}
    for edges in edge_map:
        edge_prefs[edges] = np.random.uniform(a, b)
    return edge_prefs

"""
total_preference

params: 
    - roadtrip: list of sets with each set representing an undirected edge
returns: 
    - total preference value of locations and edges, respectively 
"""
def total_preference(roadtrip): 
    locs = set()

    total_loc_val = 0
    total_edge_val = 0

    for edge in roadtrip: 

        # add vertices to set
        locs.add(edge[0])
        locs.add(edge[1])

        # do lookup for edge and add to total value
        total_edge_val += edge_prefs[edge]

    locs_list = list(locs)
    for loc in locs_list: 
        total_loc_val += loc_prefs[loc]

    return total_loc_val, total_edge_val



'''
roadtrip is a list of edges, which are represented as fixedsets of locations
'''
def time_estimate(roadtrip, x):
    # The time spent at a location as a function of its preference 
    def time_at_location (vloc): 
        return vloc 
    
    unique_locations = set()
    total_time = 0 
    for edge in roadtrip: 
        unique_locations.add(edge[0])
        unique_locations.add(edge[1])
        total_time += (
            (edge_map[edge] / x) +
            time_at_location(loc_prefs[edge[0]]) +
            time_at_location(loc_prefs[edge[1]]) +
            time_at_location(edge_prefs[edge]))
    return total_time

def main(): 
    # read in the csv files and construct edge_map 
    locs_df = pd.read_csv("road_network_locs.csv")
    edges_df = pd.read_csv("road_network_edges.csv")

    locA = edges_df["locationA"]
    locB = edges_df["locationB"]

    locations = list(locs_df["Location Label"])

    for i in range(len(locA)): 
        A = locA[i]
        B = locB[i]
        dist_A_B = edges_df["actualDistance"][i]
        path = frozenset([A, B])
        edge_map[path] = dist_A_B
    pass

if __name__ == "__main__": 
    main()