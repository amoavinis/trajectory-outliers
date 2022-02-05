from STO.STO import STO

DATA_PREFIX = "Datasets/Geolife Trajectories 1.3/Data/"
CELLS_PER_DIMENSION = 1000
sto = STO(DATA_PREFIX, CELLS_PER_DIMENSION, 7200, 3)
sto.fit()
