import pandas as pd
import numpy as np
from queue import PriorityQueue
import time 

# global variable: edge_map is a dictionary mapping out the edges as sets to their distances 
edge_map = dict() 
locations = list()
adjacency_list = {}
loc_prefs= {} 
edge_prefs = {}
locs_df = None
edges_df = None

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

# a function for getting preference of edge depending on if it's a self edge or not 
def get_edge_pref(edge): 
    return edge_prefs[edge] if len(edge_prefs) == 2 else 0
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

        # add vertices to set to avoid redundancies 
        locs.update(edge)

        # do lookup for edge and add to total value
        total_edge_val += get_edge_pref(edge)

    locs_list = list(locs)
    for loc in locs_list: 
        total_loc_val += loc_prefs[loc]

    return total_loc_val, total_edge_val

# The time spent at a location as a function of its preference 
def time_at_location (vloc): 
    return vloc * 10 

'''
roadtrip is a list of edges, which are represented as fixedsets of locations
'''
def time_estimate(roadtrip, x):
    unique_locations = set()
    total_time = 0 
    for edge in roadtrip: 
        edge_length = 0 if len(edge) == 1 else edge_map[edge]
        if len(edge) == 1:
            unique_locations.update(edge)
        total_time += (edge_length / x) + time_at_location(get_edge_pref(edge))
    
    for loc in unique_locations: 
        total_time += time_at_location(loc_prefs[loc])

    return total_time 

"""
print_roundtrip

params: 
- roundtrip: solution roundtrip stored as list of frozensets
- output_file: name of output file to write to
output: 
- print in format as seen in specification
"""
def print_roundtrip(roundtrip, speed, output_file): 
    output = []
    # sliding window of two sets at a time

    out_edge, out_set_intersection = None, None
    print(roundtrip)
    return 
    for edge1, edge2 in zip(roundtrip, roundtrip[1:]): 
        set_intersection = edge1.intersection(edge2)
        if len(output) == 0: 
            output.append(list(edge1 - set_intersection)[0])
        output.append(list(set_intersection)[0])

        out_edge, out_set_intersection = edge2, set_intersection

    # append last vertex (which is also start vertex)
    output.append(list(out_edge-out_set_intersection)[0])

    # write to the end of the output file 
    with open(output_file, "a") as f: 
        for i in range(1, len(output)): 
            # find the index where loc_df["locationA"] is equal to output[i-1] and loc_df["locationB"] is equal to output[i]
            row = locs_df.loc[(locs_df["locationA"] == output[i-1]) & (locs_df["locationB"] == output[i])]
            if row is None: 
                row = locs_df.loc[(locs_df["locationA"] == output[i]) & (locs_df["locationB"] == output[i-1])]
            edge_label = row["edgeLabel"]
            f.write(output[i-1] + "," + output[i] + "," + edge_label + "," + get_edge_pref(frozenset([output[i-1], output[i]])) + "," + 
                    edge_map[frozenset([output[i-1], output[i]])]/speed + "," + loc_prefs[output[i]] + "," + time_at_location(output[i]) + "\n")
        f.write("\n")

# Highest level round trip road trip function 
def RoundTripRoadTrip(startLoc, locFile, edgeFile, maxTime, x_mph, resultFile):
    global edge_map, locations, loc_prefs, edge_prefs, locs_df, edges_df, adjacency_list

    # read in the csv files and construct edge_map 
    locs_df = pd.read_csv(locFile)
    edges_df = pd.read_csv(edgeFile)
    edges_df = edges_df[(edges_df["actualDistance"] > 50) & (edges_df["actualDistance"] < 200)]

    locA = list(edges_df["locationA"])
    locB = list(edges_df["locationB"])

    locations = list(locs_df["Location Label"])
    

    for i in range(len(locA)): 
        A = locA[i]
        B = locB[i]
        dist_A_B = list(edges_df["actualDistance"])[i]
        path = frozenset([A, B])
        edge_map[path] = dist_A_B
    
    # parse edge_map into bidirectional adjacency_list
    for edge in edge_map:
        locA, locB = edge
        if locA not in adjacency_list:
            adjacency_list[locA] = [locB]
        else:
            adjacency_list[locA].append(locB)

        if locB not in adjacency_list:
            adjacency_list[locB] = [locA]
        else:
            adjacency_list[locB].append(locA)  
    
    # assign preference values 
    location_preference_assignments(0, 1)
    edge_preference_assignments(0, 0.1)

    # do priority search 
    pq = PriorityQueue()
    start_time = time.time() 
    pq.put((-1*loc_prefs[startLoc], [frozenset([startLoc])], 0))

    while pq.qsize() > 0: 
        print(list(pq.queue))
        elt = pq.get()
        curr_roadtrip = elt[1]

        if elt[2] > maxTime: 
            continue 

        curr_loc = list(curr_roadtrip[-1])[-1]
        print(elt)

        if (curr_loc == startLoc and len(curr_roadtrip) > 1):
            print("reached")
            print_roundtrip(curr_roadtrip, x_mph, resultFile)   
            
            # if (input("Should another solution be returned?") == "yes"):
            #     continue 
            # else:
            #     current_time = time.time() 
            #     break 

        for neighbor in adjacency_list[curr_loc]: 
            print(neighbor)
            new_roadtrip = curr_roadtrip.copy()
            new_roadtrip.append(frozenset([curr_loc, neighbor]))
            pq.put((-1*total_preference(new_roadtrip), new_roadtrip, time_estimate(new_roadtrip, x_mph)))

def main(): 
    RoundTripRoadTrip("NashvilleTN", "road_network_locs.csv", "road_network_edges.csv", 1000, 50, "result.csv")

    ''' 
    random test 

    loc_prefs = location_preference_assignments(0, 1)
    edge_prefs = edge_preference_assignments(0, 1)

    fake_roadtrip = [frozenset(["NashvilleTN", "JacksonTN"]), frozenset(["JacksonTN", "MemphisTN"]), frozenset(["NashvilleTN", "BowlingGreenKY"])]
    print(total_preference(fake_roadtrip))
    print(time_estimate(fake_roadtrip, 50))
    print(loc_prefs["NashvilleTN"])
    print(loc_prefs["JacksonTN"])
    print(loc_prefs["MemphisTN"])
    print(loc_prefs["BowlingGreenKY"])
    print(edge_prefs[frozenset(["NashvilleTN", "JacksonTN"])])
    print(edge_prefs[frozenset(["JacksonTN", "MemphisTN"])])
    print(edge_prefs[frozenset(["NashvilleTN", "BowlingGreenKY"])])
    '''

if __name__ == "__main__": 
    main()