import polyline
import pandas as pd
import json

df = pd.read_csv("Datasets/Istognosis/data_trips_er.csv")
transformed_data = {}
for (index, row) in df.iterrows():
    transformed_data[row["trip_id"]] = polyline.decode(row["points"])

with open('Datasets/Istognosis/istognosis_data.json', 'w') as fp:
    json.dump(transformed_data, fp)

