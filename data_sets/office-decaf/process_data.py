import scipy.io as sio
import os
import numpy as np
from methods import method
from data import data as data_lib
from data_sets.create_data_split import DataSplitter
from copy import deepcopy
from loss_functions import loss_function
from utility import array_functions

root_dir = 'amazon/decaf-fts/'
base_class = 'mug'


#class_dirs = ['mug', 'bottle']
class_dirs = os.listdir(root_dir)

x = []
y = []
for class_idx, d in enumerate(class_dirs):
    curr_dir = root_dir + d
    all_files = os.listdir(curr_dir)
    x_d = None
    for i, f in enumerate(all_files):
        xi = sio.loadmat(curr_dir + '/' + f)
        val = xi['fc8']
        if x_d is None:
            x_d = np.zeros((len(all_files), val.size))
        x_d[i] = np.squeeze(val)
    x.append(x_d)
    y.append(class_idx*np.ones((x_d.shape[0], 1)))
    print 'done loading ' + d
#learner = method.SKLRidgeClassification()
learner = method.SKLLogisticRegression()
learner.configs.cv_loss_function = loss_function.ZeroOneError()
loss_function = loss_function.ZeroOneError()
x_all = np.vstack(x)
y_all = np.squeeze(np.vstack(y))
#y_all = np.zeros(x_all.shape[0])
#y_all[x[0].shape[0]:] = 1
#y_all[::2] = 1

data = data_lib.Data(x_all, y_all)
data.is_regression = False

data_splitter = DataSplitter()
data_splitter.data = data

splits = data_splitter.generate_splits(data.y)

split_data = data_lib.SplitData(
    data,
    splits
)

num_classes = len(class_dirs)

#num_labels = [5, 10, 20]
num_labels = [3, 5, 10]
avg_perf = np.zeros((num_classes, len(num_labels)))
all_learners = []
base_class_idx = 0

'''
for class_idx in range(num_classes):
    if class_idx == base_class_idx:
        continue
    for num_labels_idx, n in enumerate(num_labels):
        n_perf = []
        for split_idx in range(30):
            data_copy = split_data.get_split(split_idx, n)
            I = (data_copy.true_y == class_idx) | (data_copy.true_y == base_class_idx)
            data_copy = data_copy.get_subset(I)
            results = learner.train_and_test(data_copy)
            n_perf.append(results.error_on_test_data)
        avg_perf[class_idx, num_labels_idx] = np.asarray(n_perf).mean()
    print 'done with: %s vs %s' % (class_dirs[base_class_idx], class_dirs[class_idx])
    print ['%0.3f' % i for i in avg_perf[class_idx,:]]
print avg_perf
'''
transfer_error = np.zeros((num_classes, num_classes))
data_copy = split_data.get_split(0, num_labeled=20)
for target_class_idx in range(num_classes):
    if target_class_idx == base_class_idx:
        continue
    I = (data_copy.true_y == target_class_idx) | (data_copy.true_y == base_class_idx)
    data_target = data_copy.get_subset(I)
    target_results = learner.train_and_test(data_target)
    for source_class_idx in range(num_classes):
        if target_class_idx == source_class_idx or source_class_idx == base_class_idx:
            continue
        I = (data_copy.true_y == source_class_idx) | (data_copy.true_y == base_class_idx)
        data_source = data_copy.get_subset(I)
        data_source.change_labels([source_class_idx], [target_class_idx])
        transfer_results = learner.predict(data_source)
        error = loss_function.compute_score(transfer_results)
        transfer_error[target_class_idx, source_class_idx] = error
    print 'done with target class: ' + class_dirs[target_class_idx]
    print ['%0.3f' % i for i in transfer_error[target_class_idx,:]]
    #print 'done with: %s vs %s' % (class_dirs[base_class_idx], class_dirs[class_idx])
    #print ['%0.3f' % i for i in avg_perf[class_idx,:]]
array_functions.plot_matrix(transfer_error)

print 'hello'