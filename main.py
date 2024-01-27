import pandas as pd
import numpy as np

# global variable: edge_map is a dictionary mapping out the edges as sets to their distances 
edge_map = dict() 
locations = list() 

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
            time_at_location(edge_prefs[edge])
)

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