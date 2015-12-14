__author__ = 'Aubrey'

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import pandas as pd

ng_a = np.asarray(0)
ng_c = np.asarray(range(1,6))
ng_m = np.asarray(6)
ng_r = np.asarray(range(7,11))
ng_s = np.asarray(range(11,15))
ng_o = np.asarray(15)
ng_t = np.asarray(range(16,20))

boston_housing_raw_data_file = 'boston_housing/raw_data.pkl'
ng_raw_data_file = '20ng/raw_data.pkl'
synthetic_step_transfer_file = 'synthetic_step_transfer/raw_data.pkl'
synthetic_step_linear_transfer_file = 'synthetic_step_linear_transfer/raw_data.pkl'
def create_uci_yeast():
    pass

def create_iris():
    pass

def create_20ng_data(file_dir=''):
    newsgroups_train = datasets.fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes')
    )
    data = data_class.Data()
    short_names = [
        #0
        'A',
        #1-5
        'C1','C2','C3','C4','C5',
        #6
        'M',
        #7-10
        'R1','R2','R3','R4',
        #11-14
        'S1','S2','S3','S4',
        #15
        'O',
        #16-19
        'T1','T2','T3','T4'
    ]
    data.label_names = short_names
    tf_idf = TfidfVectorizer(
        stop_words='english',
        max_df=.95,
        min_df=.001,
        max_features=10000
    )
    vectors = tf_idf.fit_transform(newsgroups_train.data)
    feature_counts = (vectors > 0).sum(0)
    data.x = vectors
    data.y = newsgroups_train.target
    data.set_defaults()
    data.is_regression = False
    class_counts = array_functions.histogram_unique(data.y)
    s = ng_raw_data_file
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

'''
def create_synthetic_step_linear_transfer(file_dir=''):
    n_target = 100
    n_source = 100
    n = n_target + n_source
    sigma = .075
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,1))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.zeros(n)
    data.y[(data.data_set_ids == 1) & (data.x[:,0] >= .5)] = 1
    data.y += np.random.normal(0,sigma,n)
    data.set_defaults()
    data.is_regression = True
    array_functions.plot_2d(data.x,data.y,data.data_set_ids)
    s = synthetic_step_transfer_file
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)
'''

def create_synthetic_step_linear_transfer(file_dir=''):
    n_target = 100
    n_source = 100
    n = n_target + n_source
    sigma = .5
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,1))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.reshape(data.x*5,data.x.shape[0])
    data.y[(data.data_set_ids == 1) & (data.x[:,0] >= .5)] += 2
    data.y += np.random.normal(0,sigma,n)
    data.set_defaults()
    data.is_regression = True
    array_functions.plot_2d(data.x,data.y,data.data_set_ids,title='Linear Step Data Set')
    s = synthetic_step_linear_transfer_file
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

def create_boston_housing(file_dir=''):
    boston_data = datasets.load_boston()
    data = data_class.Data()
    data.x = boston_data.data
    data.y = boston_data.target
    data.feature_names = list(boston_data.feature_names)
    data.set_defaults()
    data.is_regression = True
    s = boston_housing_raw_data_file
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

def create_diabetes():
    diabetes_data = datasets.load_diabetes()
    x = diabetes_data.data
    y = diabetes_data.target
    for i in range(x.shape[1]):
        xi = array_functions.normalize(x[:,i])
        yi = array_functions.normalize(y)
        array_functions.plot_2d(xi,yi)
        pass
    assert False

def create_linnerud():
    linnerud_data = datasets.load_linnerud()
    assert False

def create_digits():
    digits_data = datasets.load_digits()
    x = digits_data.data
    y = digits_data.target
    for i in range(x.shape[1]):
        xi = array_functions.normalize(x[:,i])
        yi = y
        array_functions.plot_2d(xi,yi,alpha=.01)
        pass
    pass

def create_covtype():
    covtype_data = datasets.fetch_covtype()
    print covtype_data.__dict__
    data = data_class.Data()
    data.x = covtype_data.data
    data.y = covtype_data.target
    helper_functions.save_object('data_sets/covtype/raw_data.pkl')
    pass

def create_landmine():
    landmine_data = scipy.io.loadmat('LandMine/LandmineData.mat')
    pass

def create_bike_sharing():
    file = 'bike_sharing/day.csv'
    columns = [0] + range(2,16)
    all_field_names = pd.read_csv(file,nrows=1,dtype='string')
    all_field_names = np.asarray(all_field_names.keys())
    used_field_names = all_field_names[columns]
    bike_data = np.loadtxt(file,skiprows=1,delimiter=',',usecols=columns)
    domain_ind = used_field_names == 'yr'
    domain_ids = bike_data[:,domain_ind]
    inds_to_keep = (used_field_names == 'temp') | (used_field_names == 'atemp')
    bike_data = bike_data[:,inds_to_keep]
    used_field_names = used_field_names[inds_to_keep]
    for i in range(bike_data.shape[1]):
        xi = bike_data[:,i]
        yi = array_functions.normalize(bike_data[:,-1])
        #array_functions.plot_2d(xi,yi,alpha=.1,title=used_field_names[i],data_set_ids=domain_ids)
        array_functions.plot_2d_sub(xi,yi,alpha=.1,title=used_field_names[i],data_set_ids=domain_ids)
        pass
    pass

if __name__ == "__main__":
    #create_boston_housing()
    #create_20ng_data()
    #create_synthetic_step_transfer()
    create_synthetic_step_linear_transfer()
    #create_bike_sharing()
