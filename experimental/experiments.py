from dp_tensor import DPTensor, DatasetTensor
import numpy as np


# Example 3.3 in Dwork's book

#db = DatasetTensor(data=np.array([0.,1,1,0,1,1,0]), epsilon=np.zeros(7) + 0.1, entities=np.array(range(0,6)))
#db2 = db.minimum(1).maximum(0)

#print(db, db2.max_values)


# names 1-hot encoded
TOTAL_CLASSES = 5
name_ids = np.arange(0,TOTAL_CLASSES)
NUM_NAMES = 10
database = np.zeros((NUM_NAMES, TOTAL_CLASSES))
epsilons = np.zeros((NUM_NAMES, TOTAL_CLASSES)) + 0.1



datasets = []
for row_num, idx in enumerate(np.random.choice(name_ids, NUM_NAMES)):
    database[row_num,(idx - 1)] = 1

    datasets.append(DatasetTensor(database[row_num], epsilon=epsilons[idx],
        max_values=np.ones(TOTAL_CLASSES),
        min_values=np.zeros(TOTAL_CLASSES),
        entities=[idx]))

#ds = DatasetTensor(database, epsilon=epsilons, entities=np.arange(0, NUM_NAMES))
#db2 = ds.maximum(1).minimum(0)


print("Numpy Ground Truth", np.sum(database, axis=0))


# Now perform a SUM:
accumulator = DPTensor(np.zeros(TOTAL_CLASSES), max_values=0, min_values=0)
for ds in datasets:
    accumulator = ds + accumulator


#accumulator = accumulator.minimum(0).maximum(10)
acc = DatasetTensor(accumulator.data, epsilon=epsilons, max_values=np.ones(TOTAL_CLASSES),
        entities=[],
        min_values=np.zeros(TOTAL_CLASSES))
print("DS result, sensitivity", acc.laplace(), accumulator.sensitivity)
