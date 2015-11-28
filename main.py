
__author__ = 'Aubrey Gress'

import sys
import os
import importlib


import configs.base_configs as configs_lib
from experiment.experiment_manager import ExperimentManager
import numpy as np
from methods import method
from data.data import Data
from loss_functions.loss_function import MeanSquaredError
from loss_functions import loss_function

def run_main():

    pc = configs_lib.ProjectConfigs
    bc = configs_lib.MainConfigs
    #exp_exec = ExperimentManager(configs)
    #exp_exec.run_experiments()

if __name__ == "__main__":
    #run_main()
    reg = .001
    n = 100
    p = 10
    x = np.random.rand(n,p)
    beta = np.random.rand(p)
    y = x.dot(beta) + np.random.normal(0,.5,n)

    is_train = np.random.rand(100) < .8
    d = Data()
    d.x = x
    d.y = y
    d.is_train = is_train
    '''
    m = method.SKLRidgeRegression()
    params = {
        'alpha': reg
    }
    loss = MeanSquaredError()
    '''
    mu = y.mean()
    I = d.y < mu
    d.y[I] = 1
    d.y[~I] = 0
    m = method.SKLLogisticRegression()
    loss = loss_function.ZeroOneError()
    params = {
        'C': 1000
    }
    m.set_params(**params)
    m.train(d)
    r = m.predict(d)

    print 'Train Error: ' + str(loss.compute_score(r.y,d.y,d.is_train))
    print 'Test Error: ' + str(loss.compute_score(r.y,d.y,~d.is_train))
    print 'Test Run'





