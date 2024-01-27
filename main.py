import pandas as pd
import numpy as np

# global variable: edge_map is a dictionary mapping out the edges as sets to their distances 
edge_map = dict() 
locations = list() 

"""
total_preference
params: 
    - roadtrip: list of sets with each set representing an undirected edge
returns: 
    - total preference value of locations and edges, respectively 
"""
def total_preference(roadtrip): 
    locs = set()
    
    # call preference edge and vertex list from 2a and 2b
    edge_dict = None
    loc_dict = None

    total_loc_val = 0
    total_edge_val = 0

    for edge in roadtrip: 

        # add vertices to set
        locs.add(edge[0])
        locs.add(edge[1])

        # do lookup for edge and add to total value
        total_edge_val += edge_dict[edge]

    locs_list = list(locs)
    for loc in locs_list: 
        total_loc_val += loc_dict[loc]

    return total_loc_val, total_edge_val



def time_estimate(roadtrip, x):

    pass

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