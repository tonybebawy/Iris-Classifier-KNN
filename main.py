import pandas as pd
import math
import statistics
from statistics import mode
import numpy as np


def eucledian_distance(x, y):
    dx = x[0] - x[1]
    dy = y[0] - y[1]
    return math.sqrt(dx**2 + dy**2)


data_file = 'Iris.csv'
df = pd.read_csv(data_file)

number_of_training_points = math.floor(0.7*len(df))

train_data = df[:number_of_training_points]
test_data = df[number_of_training_points:]

# initialize k
k = 10

correct_count = 0
for test_index, test_row in test_data.iterrows():
    x1 = test_row["SepalLengthCm"]/test_row["SepalWidthCm"]
    y1 = test_row["PetalLengthCm"]/test_row["PetalWidthCm"]

    neighbors = {}

    for index, row in train_data.iterrows():
        x2 = row["SepalLengthCm"]/row["SepalWidthCm"]
        y2 = row["PetalLengthCm"]/row['PetalWidthCm']

        x = (x1, x2)
        y = (y1, y2)

        # computing the eucledian distance
        distance = eucledian_distance(x, y)
        neighbors[distance] = row["Species"]

    top_k = []

    count = 0
    for key in sorted(neighbors.keys()):
        if count == k:
            break
        top_k.append(neighbors[key])
        count += 1

    prediction = ''
    try:
        prediction = mode(top_k)
    except statistics.StatisticsError:
        prediction = top_k[0]

    if (prediction == test_row["Species"]):
        correct_count += 1

    print(test_row["Species"])
    print(prediction)
    print(correct_count)

print('Accuracy of model: ' + "{0:.1%}".format(correct_count/len(test_data)))
