from data_preprocess import Preprocessor
from TOP import TOP

DATA_PREFIX = "Datasets/Geolife Trajectories 1.3/Data/"
CELLS_PER_DIMENSION = 1000
minSup = 2
seqGap = 2
preprocessor = Preprocessor(DATA_PREFIX, CELLS_PER_DIMENSION, minSup, seqGap)

print("Preprocessing started")
preprocessor.preprocess()
print("Preprocessing ended")

top = TOP(preprocessor.all_trajectories, preprocessor.search_spaces,
          preprocessor.freq_events, minSup)
top.fit()

freqPatterns = top.freq_patterns

#print(freqPatterns)

