"""
Team Members (Group 3):
- Haoli Yin
- Lydia Liu
- Richard Song
- Shreya Reddy

How to Run Code:
Create a python virtual environment
Install dependencies (Linux instructions, slightly different for Windows)
Run code
```bash 

python -m venv venv 
source venv/bin/activate
python main.py
```

Search Strategy:
Our code implements a utility-driven search using a priority queue, where the search prioritizes
paths using the total_preference as the heuristic value. The time_estimate function works to filter
out routes that take longer than the specified maxTime. The algorithm aims to maximize the overall
preference of the roadtrip within the given time constraints. 
"""
import pandas as pd
import numpy as np
from queue import PriorityQueue
import time 
from create_loc_features import generate_location_vector

# global variable: edge_map is a dictionary mapping out the edges as sets to their distances 
edge_map = dict() 
locations = list()
edges = list() 
all_trip_prefs = list()
# all_runtimes = list()
adjacency_list = {}
loc_prefs= {} 
edge_prefs = {}
locs_df = None
edges_df = None
total_distance = 0

# global variables for location and edge utilities
location_utilities = dict()
edge_utilities = dict()


"""
location_preference_assignments

params: 
    - a: lower bound of assigned location preference values (default = 0)
    - b: upper bound of assigned location preference values (default = 1)
returns: 
    - dict assignment of all location preference values
"""
def location_preference_assignments(a, b):
    global locations, loc_prefs
    loc_prefs = {}
    for loc in locations:
        loc_prefs[loc] = np.random.uniform(a, b)
    return loc_prefs

"""
edge_preference_assignments

params: 
    - a: lower bound of assigned edge preference values (default = 0)
    - b: upper bound of assigned edge preference values (default = 0.1)
returns: 
    - dict assignment of all edge preference values
"""
def edge_preference_assignments(a=0, b=0.1):
    global edge_map, edge_prefs
    edge_prefs = {}
    for edges in edge_map:
        edge_prefs[edges] = np.random.uniform(a, b)
    return edge_prefs

"""
get_edge_pref

params: 
    - edge: individual edge within the roadtrip
returns: 
    - preference of edge depending on if it's a self edge or not, else 0
"""
def get_edge_pref(edge): 
    return 0 if len(edge) != 2 else edge_prefs[edge]

"""
total_preference

params: 
    - roadtrip: list of sets with each set representing an undirected edge
returns: 
    - total sum of all location and all edge preferences in a road trip
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

    return total_loc_val + total_edge_val

"""
time_at_location

params:
    - vloc: preference value of a location
returns:
    - time spent at a location as a function of its preference
""" 
def time_at_location (vloc): 
    return vloc * 10 

"""
time_estimate

params: 
    - roadtrip: list of edges, which are represented as fixedsets of locations
    - x: assumed travel speed (in mph) of all edges 
returns: 
    - total time required by a road trip in terms of its constituent edges and locations
"""
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
    - startLoc: start location of roadtrip
returns: 
    - print in format as seen in specification
"""
def print_roundtrip(output, speed, file_counter, startLoc, maxTime): 
    '''
    output = []
    # sliding window of two sets at a time
    out_edge, out_set_intersection = None, None
    print(roundtrip)
    for edge1, edge2 in zip(roundtrip, roundtrip[1:]): 
        set_intersection = edge1.intersection(edge2)
        if len(output) == 0: 
            print(edge1, edge2)
            print(set_intersection)
            exit()
            output.append(list(edge1 - set_intersection)[0])
        output.append(list(set_intersection)[0])

        out_edge, out_set_intersection = edge2, set_intersection

    # append last vertex (which is also start vertex)
    output.append(list(out_edge-out_set_intersection)[0])
    print(output)
    '''
    global total_distance
    total_distance = 0
    
    # write to the end of the output file 
    with open("resultFile.csv", "a") as f: 
        f.write(f'Solution: {file_counter}, Start Location: {startLoc}, Max Time: {maxTime} hrs, Speed: {speed} mph\n\n')
        print(f'\n\nSolution: {file_counter}, Start Location: {startLoc}, Max Time: {maxTime} hrs, Speed: {speed} mph\n')

        for i in range(1, len(output)): 
            # find the index where loc_df["locationA"] is equal to output[i-1] and loc_df["locationB"] is equal to output[i]
            row = edges_df[(edges_df["locationA"] == output[i-1]) & (edges_df["locationB"] == output[i])]
            if row.shape[0] == 0: 
                row = edges_df[(edges_df["locationA"] == output[i]) & (edges_df["locationB"] == output[i-1])]
            edge_label = list(row["edgeLabel"])[0]

            edge_distance = edge_map.get(frozenset([output[i-1], output[i]]), 0)
            total_distance += edge_distance

            printing = f'{i}. {output[i-1]}, {output[i]}, {edge_label}, {get_edge_pref(frozenset([output[i-1], output[i]]))}, {edge_map[frozenset([output[i-1], output[i]])]/speed}, {loc_prefs[output[i]]}, {time_at_location(loc_prefs[output[i]])} \n'
            f.write(f'{printing}\n')
            print(printing)

"""
print_summary

params: 
    - all_trip_prefs: list of each solution's total trip preference
returns: 
    - prints summary to resultFile.csv and to screen
"""    
def print_summary(trip_pref_list, all_runtimes):
    avg_runtime = sum(all_runtimes) / len(all_runtimes) if all_runtimes else 0
    max_pref = max(trip_pref_list)
    avg_pref = sum(trip_pref_list) / len(trip_pref_list) if trip_pref_list else 0
    min_pref = min(trip_pref_list)

    with open(f"resultFile.csv", "a") as f:
        f.write(f'Summary File\n\n')
        f.write(f'Average instrumented runtime of all continuations of the search: {avg_runtime} seconds\n\n')
        f.write(f'Maximum Total Trip Preference found across all solution paths: {max_pref}\n\n')
        f.write(f'Average Total Trip Preference found across all solution paths: {avg_pref}\n\n')
        f.write(f'Minimum Total Trip Preference found across all solution paths: {min_pref}')

    print(f'\n\nSummary:\n')
    print(f'Average instrumented runtime of all continuations of the search: {avg_runtime} seconds\n')
    print(f'Maximum Total Trip Preference found across all solution paths: {max_pref}\n')
    print(f'Average Total Trip Preference found across all solution paths: {avg_pref}\n')
    print(f'Minimum Total Trip Preference found across all solution paths: {min_pref}\n')    
        

"""
RoundTripRoadTrip

params: 
    - startLoc: start (and end)location of the round trip
    - locFile: csv file of each location and their respective latitude, longitude, contributer, and notes
    - edgeFile: csv file of each edge between locations and their respective actualDistance, contributer, and notes
    - maxTime: the maximum amount of time (in hours) that the roadtrip cannot exceed
    - x_mph: assumed travel speed (in mph) of all edges 
    - resultFile: solution path and summary of the entire roadtrip
"""
def RoundTripRoadTrip(startLoc, locFile, edgeFile, maxTime, x_mph, resultFile):
    global edge_map, locations, loc_prefs, edge_prefs, locs_df, edges_df, adjacency_list, edges 

    # read in the csv files and construct edge_map 
    locs_df = pd.read_csv(locFile)
    edges_df = pd.read_csv(edgeFile)
    edges_df = edges_df[(edges_df["actualDistance"] > 50) & (edges_df["actualDistance"] < 200)]

    locA = list(edges_df["locationA"])
    locB = list(edges_df["locationB"])

    locations = list(locs_df["Location Label"])
    edges = list(edges_df["edgeLabel"])

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
    pq.put((-1*loc_prefs[startLoc], [frozenset([startLoc])], 0, [startLoc]))
    all_runtimes = list()
    file_counter = 1

    # clear/create new resultFile.csv
    with open("resultFile.csv", "w") as f:
        f.write("")

    start_time = time.time()
    pause_time = 0

    while pq.qsize() > 0: 

        # print(list(pq.queue))
        elt = pq.get()
        curr_roadtrip = elt[1]

        if elt[2] > maxTime: 
            continue 

        curr_loc = elt[3][-1]
        #print(elt)

        if (curr_loc == startLoc and len(curr_roadtrip) > 1):
            #print("reached")
            
            print_roundtrip(elt[3], x_mph, file_counter, startLoc, maxTime)
            total_pref = total_preference(curr_roadtrip)
            all_trip_prefs.append(total_pref)

            with open(f"resultFile.csv", "a") as f: 
                f.write(f'Start Location: {startLoc}, Total Trip Preference: {total_pref} , Total Trip Distance: {total_distance} miles, Total Trip Time: {time_estimate(curr_roadtrip, x_mph)} hrs\n\n\n\n')
            print(f'Start Location: {startLoc}, Total Trip Preference: {total_pref} , Total Trip Distance: {total_distance} miles, Total Trip Time: {time_estimate(curr_roadtrip, x_mph)} hrs\n\n\n\n')

            file_counter +=1
            

            
            pause_start = time.time()
            if (input("Should another solution be returned? ") == "yes"):
                pause_end = time.time()
                pause_time += pause_end - pause_start
                end_time = time.time()
                search_time = end_time - start_time - pause_time
                print(f"search time: {search_time} seconds")
                all_runtimes.append(search_time)
                start_time = time.time()
                pause_time = 0
                continue 
            else:
                # current_time = time.time() 
                pause_end = time.time()
                pause_time += pause_end - pause_start
                end_time = time.time()
                search_time = end_time - start_time - pause_time
                print(f"search time: {search_time} seconds")
                all_runtimes.append(search_time)
                print_summary(all_trip_prefs, all_runtimes)
                start_time = time.time()
                pause_time = 0
                break
            
            

        for neighbor in adjacency_list[curr_loc]: 
            new_roadtrip = curr_roadtrip.copy()
            new_roadtrip.append(frozenset([curr_loc, neighbor]))
            pq.put((-1*total_preference(new_roadtrip), new_roadtrip, time_estimate(new_roadtrip, x_mph), elt[3] + [neighbor]))

'''
generate_utilities: generates the utility vectors for locations and edges. most locations will have a utility 
vector that is specified by the number of themes that the location has. The edge utility vector is randomly generated.

params:
- None
'''
def generate_utilities():
    global locations, edges, location_utilities, edge_utilities 

    # first get the utility vectors that were provided for us by the dataset 
    utility_dictionary = generate_location_vector() 

    # next, parse through all of the locations. If they are already in the utility_vector, 
    # then we add them to the location_utilities dictionary. If not, we assign them 0 utility.

    for loc in locations:
        if loc in utility_dictionary:
            location_utilities[loc] = utility_dictionary[loc]
        else:
            location_utilities[loc] = np.zeros(67) # 67 is the number of themes in the dataset
    
    # assign the edges random utility values 
    for edge in edges:
        # vector of 67 random values between 0 and 1
        edge_utilities[edge] = np.random.uniform(0, 1, 67)


def main(): 
    RoundTripRoadTrip("NashvilleTN", "road_network_locs.csv", "road_network_edges.csv", 20, 50, "result.csv")

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

"""
Qualitative Comments:
Test Cases with No Solution: The design of the algorithm inherently filters out routes exceeding 
the specified maximum time (maxTime) through its time_estimate function. This implies that for certain start locations, 
configurations of location and edge preferences, and stringent time constraints, it's plausible that 
no viable solution could be found. However, this is unlikely as the edges are undirected and a cycle on an edge could be 
considered a valid road trip, albeit one that's not preferred. 

Solution Value Trends: The search prioritizes the highest preference scores first, without ensuring a monotonic decrease in 
solution values. The variability in preferences means the sequence of solution values could fluctuate based on the explored paths.

Adhering to Time Constraints: The time_estimate function ensures solutions adhere to specified time constraints by 
discarding paths that exceed maxTime, assuming the function accurately estimates travel times.


Quantitative Summary:

For 3 runs: 

Average instrumented runtime of all continuations of the search: 0.0017745494842529297 seconds

Maximum Total Trip Preference found across all solution paths: 3.9694293212102565

Average Total Trip Preference found across all solution paths: 3.664504129915747

Average Minimum Total Trip Preference found across all solution paths: 3.1882612435306763

This data is directly shown after answering no to "Should another solution be returned?"
"""    