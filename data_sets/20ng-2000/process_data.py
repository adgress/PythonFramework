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
from methods import transfer_methods
data = helper_functions.load_object('raw_data.pkl')
from configs import base_configs

data_splitter = DataSplitter()
data_splitter.data = data

splits = data_splitter.generate_splits(data.y)

split_data = data_lib.SplitData(
    data,
    splits
)
use_transfer = False
use_regression = True
m = base_configs.MethodConfigs()
m.use_validation = True
if use_transfer:
    assert not use_regression
    m.loss_function = loss_function.ZeroOneError()
    m.cv_loss_function = loss_function.ZeroOneError()
    transfer_learner = transfer_methods.StackingTransfer(deepcopy(m))
    transfer_learner.base_learner = method.SKLLogisticRegression(deepcopy(m))
    #transfer_learner.source_learner = method.SKLLogisticRegression(deepcopy(m))
    transfer_learner.source_learner = method.SKLKNN(deepcopy(m))
    transfer_learner.source_learner.configs.use_validation = False
    transfer_learner.use_all_source = True
    #transfer_learner.target_learner = method.SKLLogisticRegression(deepcopy(m))
    transfer_learner.target_learner = method.SKLKNN(deepcopy(m))

#learner = method.SKLKNN(deepcopy(m))
#learner = method.SKLLogisticRegression(deepcopy(m))
#learner = method.SKLRidgeClassification()
if use_regression:
    learner = method.SKLKNNRegression(deepcopy(m))
    learner.configs.cv_loss_function = loss_function.MeanAbsoluteError()
    loss_function = loss_function.MeanAbsoluteError()
else:
    learner = method.SKLKNN(deepcopy(m))
    learner.configs.cv_loss_function = loss_function.ZeroOneError()
    loss_function = loss_function.ZeroOneError()
num_classes = data.classes.size

all_learners = []
base_class_idx = 0
base_label = data.classes[base_class_idx]

#rows are sources, columns are targets
transfer_error = np.zeros((num_classes, num_classes))
num_splits = 10
for split_idx in range(num_splits):
    data_copy = split_data.get_split(split_idx, num_labeled=20)
    for source_idx in range(num_classes):
        if source_idx == base_class_idx:
            continue
        source_label = data.classes[source_idx]
        I = (data_copy.true_y == source_label) | (data_copy.true_y == base_label)
        data_source = data_copy.get_subset(I)
        source_results = learner.train_and_test(data_source)
        transfer_error[source_idx, source_idx] += source_results.error_on_test_data
        for target_idx in range(num_classes):
            if target_idx == source_idx or target_idx == base_class_idx:
                continue
            target_label = data.classes[target_idx]
            if use_transfer:
                all_labels = np.asarray([base_label, target_label, source_label])
                I = array_functions.find_set(data_copy.true_y, np.asarray(all_labels))
                data_target = data_copy.get_subset(I)
                data_base = data_copy.get_subset(data_copy.true_y == base_label)
                # Create a new label to duplicate base data
                new_label = data.classes.max() + 1
                data_base.change_labels([base_label], [new_label])
                data_target.combine(data_base)
                data_target.data_set_ids = None
                transfer_learner.configs.source_labels = np.expand_dims(np.asarray([source_label, base_label]), 0)
                transfer_learner.configs.target_labels = np.asarray([target_label, new_label])
                transfer_results = transfer_learner.train_and_test(data_target).prediction
            else:
                I = (data_copy.true_y == target_label) | (data_copy.true_y == base_label)
                data_target = data_copy.get_subset(I)
                transfer_results = learner.predict(data_target)
                transfer_results.y[transfer_results.y == source_label] = target_label

            assert False, 'Are we computing error on test data?'
            error = loss_function.compute_score(transfer_results)
            transfer_error[source_idx, target_idx] += error
        print str(source_label) + ' done with source class: ' + data.label_names[source_idx]
        print [str(idx) + ':%0.3f' % i for idx, i in enumerate(transfer_error[source_idx,:]/(split_idx+1))]
    print 'done with split ' + str(split_idx)

transfer_error /= num_splits

#subtract error when training and testing with target data

transfer_error = transfer_error[1:,:]
transfer_error = transfer_error[:,1:]
normalize_relative_difference = True
if normalize_relative_difference:
    for i in range(transfer_error.shape[0]):
        transfer_error[:, i] = (transfer_error[:, i] - transfer_error[i, i]) / transfer_error[i, i]
else:
    for i in range(transfer_error.shape[0]):
        transfer_error[:, i] = transfer_error[:, i] - transfer_error[i, i]

print str(transfer_error.T)
'''
if use_transfer:
    transfer_error -= transfer_error[:].min()
'''
print str(transfer_error.T)
array_functions.plot_matrix(transfer_error)

print 'hello'