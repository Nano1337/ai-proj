import pandas as pd
import textwrap

def get_cities(): 
    return pd.read_csv("road_network_locs.csv")['Location Label'].tolist()

def get_states(): 
    """
    Returns: 
        - states: list of deduplicated, two-character states present in the cities list from the locations csv
    """
    cities = get_cities()
    states = set()
    for city in cities: 
        states.add(city[-2:])
    return states

def include_exclude_cities(): 
    include_states, exclude_states = include_exclude_states()
    cities = get_cities()

    include_cities = []
    exclude_cities = []
    
    for city in cities:
        # Extract the state abbreviation from the last two characters of the city string
        state_abbr = city[-2:]
        
        # If the state is in the include list and not in the exclude list, add to include_cities
        if state_abbr in include_states and state_abbr not in exclude_states:
            include_cities.append(city)
        
        # If the state is in the exclude list, add to exclude_cities
        elif state_abbr in exclude_states:
            exclude_cities.append(city)
    
    return include_cities, exclude_cities

    

def include_exclude_states(): 
    """
    Returns: 
        - include: list of included states
        - exclude: list of excluded states
    """
    include = set()
    exclude = set()

    states = sorted(list(get_states()))[:-1]
    print()
    print("Here are the list of states to pick to include or exclude:")
    num_rows = -(-len(states) // 8) 
    states_table = [["" for _ in range(8)] for _ in range(num_rows)]
    for i, state in enumerate(states):
        row = i // 8
        col = i % 8
        states_table[row][col] = state
    for row in states_table:
        row_str = " | ".join([f"{state:<8}" for state in row if state])
        print(row_str)
    print()

    while True:
        user_input = input("Input state to include and press Enter, or 'done' to finish: ")
        if user_input.lower() == "done":
            break
        else:
            user_input = user_input.replace(" ", "")
            user_input = user_input.upper()
            if user_input not in states: 
                print(user_input, "is not a valid state, please try again")
            else: 
                include.add(user_input)

    include = list(include)
    print()
    print("Your preferred states are:", include)
    print()

    print("Here are the list of states to pick to exclude, again:")
    num_rows = -(-len(states) // 8) 
    states_table = [["" for _ in range(8)] for _ in range(num_rows)]
    for i, state in enumerate(states):
        row = i // 8
        col = i % 8
        states_table[row][col] = state
    for row in states_table:
        row_str = " | ".join([f"{state:<8}" for state in row if state])
        print(row_str)
    print()

    while True:
        user_input = input("Input state to exclude and press Enter, or 'done' to finish: ")
        if user_input.lower() == "done":
            break
        else:
            user_input = user_input.replace(" ", "")
            user_input = user_input.upper()
            if user_input not in states: 
                print(user_input, "is not a valid state, please try again")
            elif user_input in include: 
                print(user_input ,"is already in preferred states, please try again")
            else: 
                exclude.add(user_input)
    
    print()
    exclude = list(exclude)
    print()
    print("Your excluded states are:", exclude)

    return include, exclude

if __name__ == "__main__": 
    # include, exclude = include_exclude()
    # print(include, exclude)

    print(include_exclude_cities())
    