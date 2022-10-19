from copyreg import pickle
import gpxpy
import os
import pickle

dir = os.getcwd()+"/ProposedApproach/gpx_files/"
gpx_array = [gpxpy.parse(open(dir+file, 'r')) for file in os.listdir(dir)]
trajectories = [gpx.tracks[0].segments[0].points for gpx in gpx_array]
trajectories = [[[p.longitude, p.latitude] for p in gpx] for gpx in trajectories]
pickle.dump(trajectories, open('manual_outliers.pkl', "wb"))