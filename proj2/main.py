import pandas as pd

# Preprocess data
def utility_to_label(utility):
    if utility <= 0.2:
        return 'very low'
    elif utility <= 0.4:
        return 'low'
    elif utility <= 0.6:
        return 'average'
    elif utility <= 0.8:
        return 'good'
    else:
        return 'great'
df = pd.read_csv('data.txt', sep='\t', encoding='utf-16', names=["utility", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"])
df['utility'] = df['utility'].apply(utility_to_label)

