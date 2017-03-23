import scipy.io as sio
import os
import numpy as np
from methods import method
from data import data as data_lib
from data_sets.create_data_split import DataSplitter
from copy import deepcopy
from loss_functions import loss_function
from utility import array_functions
from utility import helper_functions

data = helper_functions.load_object('raw_data.pkl')

data_splitter = DataSplitter()
data_splitter.data = data

splits = data_splitter.generate_splits(data.y)

split_data = data_lib.SplitData(
    data,
    splits
)
learner = method.SKLLogisticRegression()
#learner = method.SKLRidgeClassification()
learner.configs.cv_loss_function = loss_function.ZeroOneError()
loss_function = loss_function.ZeroOneError()
num_classes = data.classes.size

#num_labels = [5, 10, 20]
num_labels = [3, 5, 10]
avg_perf = np.zeros((num_classes, len(num_labels)))
all_learners = []
base_class_idx = 0

#rows are sources, columns are targets
transfer_error = np.zeros((num_classes, num_classes))
num_splits = 10
for split_idx in range(num_splits):
    data_copy = split_data.get_split(split_idx, num_labeled=200)
    for source_idx in range(num_classes):
        if source_idx == base_class_idx:
            continue
        I = (data_copy.true_y == data.classes[source_idx]) | (data_copy.true_y == data.classes[base_class_idx])
        data_source = data_copy.get_subset(I)
        source_results = learner.train_and_test(data_source)
        transfer_error[source_idx, source_idx] += source_results.error_on_test_data
        for target_idx in range(num_classes):
            if target_idx == source_idx or target_idx == base_class_idx:
                continue
            I = (data_copy.true_y == data.classes[target_idx]) | (data_copy.true_y == data.classes[base_class_idx])
            data_target = data_copy.get_subset(I)
            data_target.change_labels([data.classes[target_idx]], [data.classes[source_idx]])
            transfer_results = learner.predict(data_target)
            error = loss_function.compute_score(transfer_results)
            transfer_error[source_idx, target_idx] += error
        print str(source_idx) + ' done with source class: ' + data.label_names[source_idx]
        print [str(idx) + ':%0.3f' % i for idx, i in enumerate(transfer_error[source_idx,:])]
    print 'done with split ' + str(split_idx)

transfer_error /= num_splits

#subtract error when training and testing with target data

for i in range(num_classes):
    transfer_error[:, i] -= transfer_error[i, i]

array_functions.plot_matrix(transfer_error)

print 'hello'