# Traffic Outlier Detection

## Introduction
This repository holds the code for the dissertation of Angelos Moavinis for the Data and Web Science MSc Programme of the Computer Science Department of the Aristotle University of Thessaloniki. The PDF document can be found [here](http://ikee.lib.auth.gr/record/342389). Modifications have been made for the needs of a conference submission at EDBT/ICDT 2023.

In case you use anything from this work, please mention it in your work :-)

## Usage guide
You will need `python3` to be installed in your system, the code has been tested with Python version 3.10.6. Before execution, firstly run 
```
pip install -r requirements.txt
```
in order to install all the needed libraries.

After that, run the `get_datasets.sh` to download the Geolife dataset and set it up so that it is ready to read by the models. Give it execution permissions if necessary (with `chmod` for example, if your system includes it).

Finally, run the automatic labeling process for each dataset by executing the respective commands found in the `commands.sh` file and then choose any command from the listed ones (in the file) in order to train and evaluate the different model setups.

NOTE: the Path Clustering-based models build a distance matrix on their first run and they read it in subsequent runs in order to save hours of computation. For example, the Hausdorff distance-based clustering model with the Cyprus dataset produces the `cyprus_hausdorff_distances.pkl` pickle file. If you want the model to run and recalculate the distances (and thus run in the same time that it does on its first run), you can delete the related file and run the command again.