from utility import helper_functions, array_functions
import numpy as np
from numpy.linalg import *
from methods import method
from scipy.stats import pearsonr
from utility import array_functions
estimator = method.SKLRidgeRegression()

def run_main():
    #data_dir = 'data_sets/concrete'
    data_dir = 'data_sets/boston_housing'
    #data_dir = 'data_sets/kc_housing'
    #data_dir = 'data_sets/synthetic_linear_reg500-50-1.01'
    #data_dir = 'data_sets/drosophilia'
    data_dir = 'data_sets/synthetic_linear_reg500-10-1.01'
    data_file = data_dir + '/split_data.pkl'
    data = helper_functions.load_object(data_file).data
    data.set_target()
    data.set_train()
    data.set_true_y()
    #data.x = array_functions.select_k_features(data.x, data.y, 50)
    estimator.train_and_test(data)
    w_normalized = estimator.w / norm(estimator.w)
    w_normalized = np.expand_dims(w_normalized, 1)
    print w_normalized
    p = estimator.w.size
    corr = np.zeros((p,1))
    for i in range(p):
        xi = data.x[:,i]
        y = data.true_y
        corr[i] = pearsonr(xi, y)[0]
    print corr
    m = np.concatenate((w_normalized, corr), 1)
    print m

if __name__ == '__main__':
    run_main()