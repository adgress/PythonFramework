import scipy.io as sio
import os
import numpy as np
from methods import method
from data import data as data_lib
from data_sets.create_data_split import DataSplitter
from copy import deepcopy
from loss_functions import loss_function
from utility import array_functions

mat = sio.loadmat('image-net-sample.mat')