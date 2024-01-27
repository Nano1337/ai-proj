import pandas as pd
import numpy as np

def main(): 
    locs_df = pd.read_csv("road_network_locs.csv")
    edges_df = pd.read_csv("road_network_edges.csv")

    locA = edges_df["locationA"]
    locB = edges_df["locationB"]

    # make a dictionary mapping out the edges as sets to their distances 
    edge_map = dict() 
    for i in range(len(locA)): 
        A = locA[i]
        B = locB[i]
        dist_A_B = edges_df["actualDistance"][i]
        path = frozenset([A, B])
        edge_map[path] = dist_A_B

    print(edge_map) 
    pass

if __name__ == "__main__": 
    main()