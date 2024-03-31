import pandas as pd
import textwrap

def get_states(): 
    cities = pd.read_csv("road_network_locs.csv")['Location Label'].tolist()
    states = set()
    for city in cities: 
        states.add(city[-2:])
    return states

def include_exclude(): 
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
    include, exclude = include_exclude()
    print(include, exclude)
    