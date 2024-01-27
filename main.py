import pandas as pd
import numpy as np

# global variable: edge_map is a dictionary mapping out the edges as sets to their distances 
edge_map = dict() 
locations = list() 

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