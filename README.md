# Traffic Outlier Detection

## Introduction
This repository holds the code for the dissertation of Angelos Moavinis for the Data and Web Science MSc Programme of the Computer Science Department of the Aristotle University of Thessaloniki. The PDF document can be found [here](http://ikee.lib.auth.gr/record/342389). Modifications have been made for the needs of a conference submission at EDBT/ICDT 2023.

In case you use anything from this work, please mention it in your work :-)

Contact the author at `amoavinis@gmail.com` if you need any help with the code or the published documents.

## Usage guide
You will need `python3` to be installed in your system, the code has been tested with Python version 3.10.6. Before execution, firstly run 
```
pip install -r requirements.txt
```
in order to install all the needed libraries.

After that, run the `get_datasets.sh` to download the Geolife dataset and set it up so that it is ready to read by the models. Give it execution permissions if necessary (with `chmod` for example, if your system includes it).

Finally, run the automatic labeling process for each dataset by executing the respective commands found in the `commands.sh` file and then choose any command from the listed ones (in the file) in order to train and evaluate the different model setups.

NOTE: the Path Clustering-based models build a distance matrix on their first run and they read it in subsequent runs in order to save hours of computation. For example, the Hausdorff distance-based clustering model with the Cyprus dataset produces the `cyprus_hausdorff_distances.pkl` pickle file. If you want the model to run and recalculate the distances (and thus run in the same time that it does on its first run), you can delete the related file and run the command again.

## Parameters guide

- `dataset`: The dataset to be used. Available options are `cyprus` and `geolife`, the Cyprus dataset is not publically available and can only be given after communication with the author.
- `method`: The model type to be used, options are `clustering`, `svm`, or `both`.
- `G`: The number of grid squares per dimension of the 2D plane.
- `eps`: The epsilon parameter of the DBSCAN model. Ignored if `method` is `svm`.
- `minPts`: The DBSCAN minPts parameter. Ignored if `method` is `svm`.
- `distance_fn`: The distance function used for the path clustering method (`hausdorff`, `dtw` or `dtw_hilbert`). Ignored if `method` is `svm`.
- `C`: The C parameter of the SVM model. Ignored if `method` is `clustering`.
- `gamma`: The gamma parameter of the SVM model. Ignored if `method` is `clustering`.
- `kernel`: The kernel parameter of the SVM model. Ignored if `method` is `clustering`.
- `do_gsp`: Choose if the SVM model will include the GSP column. Options are `0` or `1`. Ignored if `method` is `clustering`.
- `gsp_support`: The support parameter for the GSP algorithm. Options are float numbers from `0` to `1`. Ignored if `method` is `clustering`.
- `seed` : The random seed to be used in calculations that include random state. Must be a positive integer.
