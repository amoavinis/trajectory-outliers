import itertools
import ast

class TOP:
    def __init__(self, all_trajectories, search_spaces, freq_events, minSup):
        self.all_trajectories = all_trajectories
        self.search_spaces = search_spaces
        self.freq_events = freq_events
        self.global_availability = set()
        self.minSup = minSup
        self.freq_patterns = []

    def max_subsequence(self, arr):
        max_len = 0
        for x in arr:
            if len(x) > max_len:
                max_len = len(x)
        return max_len

    def produceSubsequences(self, seq, current_len, global_availability):
        results = []
        if len(seq) >= current_len:
            combinations = list(itertools.combinations(seq[1:], current_len-1))
            combinations = [[seq[0]]+list(c) for c in combinations]
            for x in combinations:
                if all([y[0] not in global_availability for y in x]):
                    results.append(x)
        return results

    def notUsed(self, op, usedPositions):
        if usedPositions == None:
            return True
        return all([x[1] not in usedPositions for x in op])

    def setUsed(self, op, usedPositions):
        if usedPositions == None:
            usedPositions = set()
        for x in op:
            usedPositions.add(x[1])
        return usedPositions

    def constructCF(self, search_space, current_len, minSup, global_availability):
        subseqs = dict()
        usedPositions = dict()
        freq_seqs = []
        subs = []     
        for seq in search_space:
            subs.extend(self.produceSubsequences(seq, current_len, global_availability))
        for seq in subs:
            if self.notUsed(seq, usedPositions.get(str(self.getPatternFromOccurrence(seq)))):
                if not str(self.getPatternFromOccurrence(seq)) in subseqs:
                    subseqs[str(self.getPatternFromOccurrence(seq))] = 1
                else:
                    subseqs[str(self.getPatternFromOccurrence(seq))] += 1
                usedPositions[str(self.getPatternFromOccurrence(seq))] = self.setUsed(seq, usedPositions.get(str(self.getPatternFromOccurrence(seq))))
        for p in subseqs:
            if subseqs[p] >= minSup:
                freq_seqs.append(p)
        return freq_seqs

    def cur_freq_events(self, arr):
        all_event_timesteps = set()
        for traj in arr:
            if len(traj) == 0:
                continue
            traj_arr = ast.literal_eval(traj)
            for elem in traj_arr:
                all_event_timesteps.add(elem)
        return all_event_timesteps

    def updatePrefix(self, search_space):
        new_search_space = []
        for elem in search_space:
            if any([x[1] not in self.global_availability for x in elem]):
                new_search_space.append(elem)                
        return new_search_space

    def getPatternFromOccurrence(self, occ):
        return [x[0] for x in occ];

    def fit(self):
        g_max = 0
        for e in self.freq_events:
            if self.max_subsequence(self.search_spaces[e]) > g_max:
                g_max = self.max_subsequence(self.search_spaces[e])

        current_len = g_max

        prefix_set = self.freq_events

        freq_patterns = []
        while current_len > 0 and len(prefix_set) > 0:
            cur_freq = []
            for p_set in prefix_set:
                if not p_set in self.search_spaces:
                    continue
                if self.max_subsequence(self.search_spaces[p_set]) >= current_len:
                    cur_freq.extend(self.constructCF(self.search_spaces[p_set], current_len, self.minSup, self.global_availability))
            for e in self.cur_freq_events(cur_freq):
                self.global_availability.add(e)
            for p_set in prefix_set:
                if not p_set in self.search_spaces:
                    continue
                self.search_spaces[p_set] = self.updatePrefix(self.search_spaces[p_set])
                if len(self.search_spaces[p_set]) == 0:
                    self.search_spaces.pop(p_set)
            freq_patterns.extend(cur_freq)
            current_len -= 1
        self.freq_patterns = freq_patterns

    def predict(self, X):
        freq_patterns_set = set()
        for p in self.freq_patterns:
            freq_patterns_set.add(','.join(p))
        
        labels = []
        for x in X:
            x_str = ','.join(x)
            if x_str in freq_patterns_set:
                labels.append(0)
            else:
                labels.append(1)
        
        return labels