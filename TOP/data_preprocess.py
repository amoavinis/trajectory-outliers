import os
from CustomScaler import Scaler
import tqdm

class Preprocessor:
    def __init__(self, data_path, cells_per_dim, minSup, seqGap):
        self.data_path = data_path
        self.cells_per_dim = cells_per_dim
        self.minSup = minSup
        self.seqGap = seqGap
        self.all_trajectories = []
        self.all_points = []
        self.scaler = Scaler()
        self.counts_of_events = dict()
        self.freq_events = set()
        self.search_spaces = dict()
    
    def process_file(self, f):
        file = open(f, 'r')
        lines = file.readlines()[6:]

        result = []

        for line in lines:
            split_line = line.split(",")
            latitude = float(split_line[0])
            longitude = float(split_line[1])
            result.append([latitude, longitude])

        return result

    def create_trajectories(self):
        for i in os.listdir(self.data_path):
            for j in os.listdir(self.data_path+i+'/Trajectory/'):
                self.all_trajectories.append(self.process_file(self.data_path+i+'/Trajectory/'+j))

    def take_points(self):
        for t in self.all_trajectories:
            for p in t:
                self.all_points.append(p)

    def fit_scaler_and_transform_trajectories(self):
        self.scaler.fit(self.all_points)
        all_trajectories_transformed = []
        for t in self.all_trajectories:
            t1 = []
            for p in t:
                t1.append(self.scaler.transform([p])[0])
            all_trajectories_transformed.append(t1)
        self.all_trajectories = all_trajectories_transformed

    def coords_to_grid(self, coords, grid_scale):
        grid_coords = [str(int(coords[0]*grid_scale)), str(int(coords[1]*grid_scale))]
        return '-'.join(grid_coords)

    def remove_repetitions(self, traj):
        previous = None
        result = []
        for p in traj:
            if p != previous:
                result.append(p)
            previous = p
        return result

    def add_to_counts(self, d, e):
        if d.get(e) != None:
            d[e] += 1
        else:
            d[e] = 1
        return d

    def trajectories_to_grid(self):
        for i in range(len(self.all_trajectories)):
            for j in range(len(self.all_trajectories[i])):
                self.all_trajectories[i][j] = self.coords_to_grid(self.all_trajectories[i][j], self.cells_per_dim)
            #self.all_trajectories[i] = self.remove_repetitions(self.all_trajectories[i])
            for e in self.all_trajectories[i]:
                self.counts_of_events = self.add_to_counts(self.counts_of_events, e)

    def find_freq_events(self):
        for e in self.counts_of_events:
            if self.counts_of_events[e] >= self.minSup:
                self.freq_events.add(e)
    
    def remove_infrequent(self, freqs, traj):
        filtered = []
        for p in traj:
            if p[0] in freqs:
                filtered.append(p)
        return filtered

    def find_all(self, arr, elem):
        indices = []
        for i in range(len(arr)):
            if arr[i] == elem:
                indices.append(i)
        return indices

    def zip_with_index(self, arr):
        zipped = []
        for i in range(len(arr)):
            zipped.append((arr[i], i))
        return zipped

    def find_subsequences(self, traj, indices, seqGap):
        subsequences = []
        traj_with_index = self.zip_with_index(traj)
        for i in indices:
            subsequence = [traj_with_index[i]]
            if len(traj) - 1 > i:
                for p in traj_with_index[i+1:]:
                    if len(subsequence) < seqGap + 2:
                        subsequence.append(p)
                    else:
                        break
            subsequences.append(subsequence)
        return subsequences
        
    def create_search_spaces(self):
        for traj in self.all_trajectories:
            for e in set(traj):
                indices_ = self.find_all(traj, e)
                if e not in self.freq_events:
                    continue
                subsequences = self.find_subsequences(traj, indices_, self.seqGap)
                subsequences = [self.remove_infrequent(self.freq_events, t) for t in subsequences]
                subsequences_filtered = []
                for s in subsequences:
                    if len(s) > 0:
                        subsequences_filtered.append(s)
                if not e in self.search_spaces:
                    self.search_spaces[e] = subsequences_filtered
                else:
                    self.search_spaces[e].extend(subsequences_filtered)
            

    def preprocess(self):
        self.create_trajectories()
        print("Created trajectories")
        self.take_points()
        print("Took points")
        self.fit_scaler_and_transform_trajectories()
        print("Scaled trajectories")
        self.trajectories_to_grid()
        print("Grid created")
        #self.all_trajectories = [['a', 'b', 'c', 'd', 'a', 'b', 'c', 'a', 'c', 'b', 'e', 'a', 'c']]
        for i in tqdm.tqdm(range(len(self.all_trajectories))):
            for e in self.all_trajectories[i]:
                self.counts_of_events = self.add_to_counts(self.counts_of_events, e)
        self.find_freq_events()
        print("Found frequent events")
        self.create_search_spaces()
        print("Created search spaces")