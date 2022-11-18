class Preprocessor:
    def __init__(self, trajectories, minSup, seqGap):
        self.minSup = minSup
        self.seqGap = seqGap
        self.all_trajectories = trajectories
        self.all_points = []
        self.counts_of_events = dict()
        self.freq_events = set()
        self.search_spaces = dict()

    def add_to_counts(self, d, e):
        if d.get(e) != None:
            d[e] += 1
        else:
            d[e] = 1
        return d

    def count_events(self):
        for t in self.all_trajectories:
            for e in t:
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
        self.count_events()
        #self.all_trajectories = [['a', 'b', 'c', 'd', 'a', 'b', 'c', 'a', 'c', 'b', 'e', 'a', 'c']]
        self.find_freq_events()
        print("Found frequent events")
        self.create_search_spaces()
        print("Created search spaces")
