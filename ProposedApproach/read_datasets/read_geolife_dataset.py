import os
import pickle

def process_file(f):
    file = open(f, 'r')
    lines = file.readlines()[6:]

    result = []

    for line in lines:
        split_line = line.split(",")
        latitude = float(split_line[0])
        longitude = float(split_line[1])
        timestamp = '-'.join(split_line[5:]).strip()
        result.append([latitude, longitude, timestamp])

    return result

def create_trajectories(data_path):
    index = 1
    for i in os.listdir(data_path):
        for j in os.listdir(data_path + i + '/Trajectory/'):
            data = process_file(data_path + i + '/Trajectory/' + j)
            to_save = {'path': data, 'label': -1}
            pickle.dump(to_save, open('datasets/GeoLife_Preprocessed/'+str(index)+'.pkl', 'wb'))
        index += 1
        