import pandas as pd
import math
import statistics
from statistics import mode


def eucledian_distance(x, y, z, j):
    dx = x[0] - x[1]
    dy = y[0] - y[1]
    dz = z[0] - z[1]
    dj = j[0] - j[1]

    return math.sqrt(dx**2 + dy**2 + dz**2 + dj**2)

data_file = 'Iris.csv'
df = pd.read_csv(data_file)

number_of_training_points = math.floor(0.7*len(df))

train_data = df[:number_of_training_points]
test_data = df[number_of_training_points:]

# initialize k
k = 10

correct_count = 0
for test_index, test_row in test_data.iterrows():
    x1 = test_row["SepalLengthCm"]
    y1 = test_row["PetalLengthCm"]
    z1 = test_row["SepalWidthCm"]
    j1 = test_row["PetalWidthCm"]

    neighbors = {}

    for index, row in train_data.iterrows():
        x2 = row["SepalLengthCm"]
        y2 = row["PetalLengthCm"]
        z2 = row["SepalWidthCm"]
        j2 = row['PetalWidthCm']

        x = (x1, x2)
        y = (y1, y2)
        z = (z1, z2)
        j = (j1, j2)

        # computing the eucledian distance
        distance = eucledian_distance(x, y, z, j)
        neighbors[distance] = row["Species"]
    top_k = []

    count = 0

    for key in sorted(neighbors.keys()):
        if len(top_k) != k:
            top_k.append(neighbors[key])
    # print(top_k)
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
