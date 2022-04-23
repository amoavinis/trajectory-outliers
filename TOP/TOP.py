import itertools
import ast

import tqdm

class TOPClassifier:
    def __init__(self, all_trajectories, search_spaces, freq_events, minSup, seqGap):
        self.all_trajectories = all_trajectories
        self.search_spaces = search_spaces
        self.freq_events = freq_events
        self.global_unavailability = set()
        self.minSup = minSup
        self.freq_patterns = []
        self.seqGap = seqGap

    def max_subsequence(self, arr):
        max_len = 0
        for x in arr:
            if len(x) > max_len:
                max_len = len(x)
        return max_len

    def produceSubsequences(self, seq, current_len, global_unavailability):
        results = []
        if len(seq) >= current_len:
            permutations = list(itertools.permutations(seq[1:]))
            permutations = [[seq[0]]+list(p) for p in permutations]

            for x in permutations:
                if all([y[0] not in global_unavailability for y in x]):
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

    def constructCF(self, search_space, current_len, minSup, global_unavailability):
        subseqs = dict()
        usedPositions = dict()
        freq_seqs = []
        subs = []
        for seq in search_space:
            subs.extend(self.produceSubsequences(seq, current_len, global_unavailability))
        for seq in subs:
            if self.notUsed(seq, usedPositions.get(str(self.getPatternFromOccurrence(seq)))):
                if not str(self.getPatternFromOccurrence(seq)) in subseqs:
                    subseqs[str(self.getPatternFromOccurrence(seq))] = 1
                else:
                    subseqs[str(self.getPatternFromOccurrence(seq))] += 1
                usedPositions[str(self.getPatternFromOccurrence(seq))] = self.setUsed(seq, usedPositions.get(str(self.getPatternFromOccurrence(seq))))
        for p in subseqs:
            if subseqs[p] >= minSup:
                freq_seqs.append(ast.literal_eval(p))
        return freq_seqs

    def cur_freq_events(self, arr):
        all_event_timesteps = set()
        for traj in arr:
            if len(traj) == 0:
                continue
            traj_arr = "->".join(traj)
            for elem in traj_arr:
                all_event_timesteps.add(elem)
        return all_event_timesteps

    def updatePrefix(self, search_space):
        new_search_space = []
        for elem in search_space:
            if any([x[1] not in self.global_unavailability for x in elem]):
                new_search_space.append(elem)
        return new_search_space

    def getPatternFromOccurrence(self, occ):
        return [x[0] for x in occ]

    def removeRepetitions(self, trajectory):
        cleaned = [trajectory[0]]
        for p in trajectory[1:]:
            if cleaned[-1] != p:
                cleaned.append(p)
        return cleaned

    def fit(self):
        g_max = 0
        for e in self.freq_events:
            if self.max_subsequence(self.search_spaces[e]) > g_max:
                g_max = self.max_subsequence(self.search_spaces[e])

        current_len = g_max

        prefix_set = self.freq_events

        freq_patterns = []
        while current_len > 0 and len(prefix_set) > 0:
            print("Mining CF patterns for length-" + str(current_len) + " sequences." )
            cur_freq = []
            for p_set in tqdm.tqdm(prefix_set):
                if not p_set in self.search_spaces:
                    continue
                if self.max_subsequence(self.search_spaces[p_set]) >= current_len:
                    cur_freq.extend(self.constructCF(self.search_spaces[p_set], current_len, self.minSup, self.global_unavailability))
            for e in self.cur_freq_events(cur_freq):
                self.global_unavailability.add(e)
            for p_set in prefix_set:
                if not p_set in self.search_spaces:
                    continue
                self.search_spaces[p_set] = self.updatePrefix(self.search_spaces[p_set])
                if len(self.search_spaces[p_set]) == 0:
                    self.search_spaces.pop(p_set)
            freq_patterns.extend(cur_freq)
            current_len -= 1
        self.freq_patterns = [self.removeRepetitions(fp) for fp in freq_patterns]

    def traj_to_string(self, traj):
        return "->".join(traj)

    def predict_for_one(self, freq_patterns, x):
        return 0 if self.traj_to_string(x) in freq_patterns else 1

    def predict(self, X):
        freq_patterns_set = set()
        for p in self.freq_patterns:
            freq_patterns_set.add('->'.join(p))

        labels = [self.predict_for_one(freq_patterns_set, x) for x in X]

        return labels
