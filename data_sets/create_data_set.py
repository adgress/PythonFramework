__author__ = 'Aubrey'

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import pandas as pd
import math
from PyMTL_master.src import PyMTL
from viz_data import viz_features
from PyMTL_master.src.PyMTL import data as PyMTL_data

ng_a = np.asarray(0)
ng_c = np.asarray(range(1,6))
ng_m = np.asarray(6)
ng_r = np.asarray(range(7,11))
ng_s = np.asarray(range(11,15))
ng_o = np.asarray(15)
ng_t = np.asarray(range(16,20))

synthetic_dim = 5

max_features=2000
pca_feats=20
#max_features=100
boston_num_feats = np.inf
concrete_num_feats = np.inf

boston_housing_raw_data_file = 'boston_housing%s/raw_data.pkl'
ng_raw_data_file = '20ng-%d/raw_data.pkl' % max_features
synthetic_step_transfer_file = 'synthetic_step_transfer/raw_data.pkl'
synthetic_step_kd_transfer_file = 'synthetic_step_transfer_%d/raw_data.pkl'
synthetic_step_linear_transfer_file = 'synthetic_step_linear_transfer/raw_data.pkl'
synthetic_classification_file = 'synthetic_classification/raw_data.pkl'
synthetic_classification_local_file = 'synthetic_classification_local/raw_data.pkl'
concrete_file = 'concrete%s/raw_data.pkl'
bike_file = 'bike_sharing%s/raw_data.pkl'
wine_file = 'wine%s/raw_data.pkl'

def make_learner():
    from methods.method import NadarayaWatsonMethod
    from loss_functions.loss_function import MeanSquaredError
    learner = NadarayaWatsonMethod()
    learner.configs.cv_loss_function = MeanSquaredError()
    learner.configs.loss_function = MeanSquaredError()
    return learner



def load_csv(file, has_field_names=True, dtype='float', delim=',',converters=None):
    nrows = 1
    if not has_field_names:
        nrows = 0
        all_field_names = None
    else:
        all_field_names = pd.read_csv(file,nrows=nrows,dtype='string',sep=delim)
        all_field_names = np.asarray(all_field_names.keys())
    data = np.loadtxt(
        file,
        skiprows=nrows,
        delimiter=delim,
        usecols=None,
        dtype=dtype,
        converters=converters
    )
    return all_field_names, data

def create_uci_yeast():
    pass

def create_iris():
    pass

def create_forest_fires():
    months = {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12
    }
    days = {
        'sun': 1,
        'mon': 2,
        'tue': 3,
        'wed': 4,
        'thu': 5,
        'fri': 6,
        'sat': 7
    }
    #month_to_season = lambda x : (months[x]-1)/3
    month_to_season = lambda x : months[x]
    day_to_int = lambda x: days[x]
    file = 'forest_fires/forestfires.csv'
    converters = {
        2: month_to_season,
        3: day_to_int
    }
    field_names, forest_data = load_csv(file,dtype='float',converters=converters)
    x = forest_data
    y = forest_data[:,-1]
    i = field_names == 'month'
    domain_ids = forest_data[:,i]
    #months_to_use = np.asarray([6,7,8])
    months_to_use = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12])
    to_use = array_functions.find_set(domain_ids,months_to_use)
    x = x[to_use,:]
    y = y[to_use]
    domain_ids = domain_ids[to_use]
    x = x[:,4:]
    field_names = field_names[4:]
    I = (y > 0) & (y < 700)
    x = x[I,:]
    y = y[I]
    domain_ids = domain_ids[I]
    viz_features(x,y,domain_ids,field_names)
    pass

def create_mpg():
    file = 'mpg/auto-mpg.data.txt'
    #field_names, mpg_data = load_csv(file,has_field_names=False,dtype='string',delim=' ')
    data = pd.read_csv(file,skiprows=0,delim_whitespace=True,dtype='string')
    data = np.asarray(data)[:,0:-1]
    has_missing_values = (data == '?').any(1)
    data = data[~has_missing_values,:]
    data = data.astype('float')
    domain_ids = data[:,1]
    x = data
    y = data[:,0]
    viz_features(x,y,domain_ids)
    pass

def create_energy():
    file = 'energy/ENB2012_data.csv'
    field_names, energy_data = load_csv(file)
    domain_ids = energy_data[:,0]
    x = energy_data
    y = energy_data[:,-1]
    viz_features(x,y,domain_ids,field_names)
    pass

def create_concrete():
    file = 'concrete/Concrete_Data.csv'
    used_field_names, concrete_data = load_csv(file)

    feat_ind = 0

    domain_ind = (used_field_names == 'age').nonzero()[0][0]
    ages = concrete_data[:,domain_ind]
    domain_ids = np.zeros(ages.shape)
    domain_ids[ages < 10] = 1
    domain_ids[(ages >= 10) & (ages <= 28)] = 2
    domain_ids[ages > 75] = 3


    data = data_class.Data()
    data.x = concrete_data[:,0:(concrete_data.shape[1]-2)]
    #0,3,5
    #data.x = preprocessing.scale(data.x)
    if concrete_num_feats == 1:
        data.x = array_functions.vec_to_2d(data.x[:,feat_ind])
        t = '-feat=' + str(feat_ind)
    elif concrete_num_feats >= data.x.shape[1]:
        t = '-' + str(min(data.x.shape[1], concrete_num_feats))
    else:
        assert False

    data.y = concrete_data[:,-1]
    data.set_defaults()
    data.is_regression = True
    data.data_set_ids = domain_ids

    viz = False
    if viz:
        to_use = domain_ids > 0
        domain_ids = domain_ids[to_use]
        concrete_data = concrete_data[to_use,:]
        np.delete(concrete_data,domain_ind,1)
        viz_features(concrete_data,concrete_data[:,-1],domain_ids,used_field_names)

        return
    data.x = array_functions.standardize(data.x)
    #viz_features(data.x,data.y,data.data_set_ids)

    s = concrete_file % t
    helper_functions.save_object(s,data)

def create_school():
    school_data = PyMTL_data.load_school_data()
    pass

def create_survey():
    computer_data = PyMTL_data.load_computer_survey_data()
    pass


def create_pager2008():
    file = 'PAGER2008/PAGER_CAT_2008_06.1.csv'
    all_field_names = pd.read_csv(file,nrows=1,dtype='string')
    all_field_names = np.asarray(all_field_names.keys())
    columns = None
    #used_field_names = all_field_names[columns]
    #pager_data = np.loadtxt(file,skiprows=0,delimiter=',',usecols=columns,dtype='string')
    pager_data = pd.read_csv(
        file,
        skiprows=0,
        delimiter=',',
        engine='c',
        usecols=columns,
        dtype='string',
    )


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
    y = newsgroups_train.target
    l = [1,2,7,8,12,17]
    #l = [1,2,7,8,12,13]
    I = array_functions.false(len(newsgroups_train.target))
    for i in l:
        I = I | (y == i)
    #I = y == 1 | y == 2 | y == 7 | y == 7 | y == 11 | y == 16
    I = I.nonzero()[0]
    max_df = .95
    min_df = .001
    #max_df = .1
    #min_df = .01
    newsgroups_train.data = [newsgroups_train.data[i] for i in I]
    newsgroups_train.target = newsgroups_train.target[I]
    tf_idf = TfidfVectorizer(
        stop_words='english',
        max_df=max_df,
        min_df=min_df,
        max_features=max_features
    )
    vectors = tf_idf.fit_transform(newsgroups_train.data)
    feature_counts = (vectors > 0).sum(0)
    vocab = helper_functions.invert_dict(tf_idf.vocabulary_)
    num_feats = len(vocab)
    vocab = [vocab[i] for i in range(num_feats)]

    pca = PCA(n_components=pca_feats)
    v2 = pca.fit_transform(vectors.toarray())
    vectors = v2

    y = newsgroups_train.target.copy()
    '''
    y[y==7] = 1
    y[(y==2) | (y==8)] = 2
    y[(y==12) | (y==17)] = 3
    '''
    '''
    y[y == 2] = 1
    y[(y==7) | (y==8)] = 2
    y[(y==12) | (y==13)] = 3
    #I_f = (y==1) | (y==7) | (y==11) | (y==16)
    I_f = array_functions.true(vectors.shape[0])
    f = f_classif
    k_best = SelectKBest(score_func=f, k=pca_feats)
    v2 = k_best.fit_transform(vectors[I_f,:], y[I_f])
    k_best.transform(vectors)
    s = k_best.get_support()
    selected_vocab = [vocab[i] for i in s.nonzero()[0]]
    vocab = selected_vocab
    vectors = v2
    '''


    data.x = vectors
    data.y = newsgroups_train.target
    data.set_defaults()
    data.is_regression = False
    data.feature_names = vocab
    class_counts = array_functions.histogram_unique(data.y)
    s = ng_raw_data_file
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

def create_synthetic_classification(file_dir='',local=True):
    dim = 1
    n_target = 200
    n_source = 200
    n = n_target + n_source
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,dim))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.zeros(n)
    x, ids = data.x, data.data_set_ids
    I = array_functions.in_range(x,0,.25)
    I2 = array_functions.in_range(x,.25,.5)
    I3 = array_functions.in_range(x,.5,.75)
    I4 = array_functions.in_range(x,.75,1)
    id0 = ids == 0
    id1 = ids == 1
    data.y[I & id0] = 1
    data.y[I2 & id0] = 2
    data.y[I3 & id0] = 1
    data.y[I4 & id0] = 2

    data.y[I & id1] = 3
    data.y[I2 & id1] = 4
    data.y[I3 & id1] = 3
    data.y[I4 & id1] = 4
    if local:
        data.y[I3 & id1] = 4
        data.y[I4 & id1] = 3
    data.set_true_y()
    data.set_train()
    data.is_regression = False
    noise_rate = 0
    #data.add_noise(noise_rate)
    data.add_noise(noise_rate, id0, np.asarray([1,2]))
    data.add_noise(noise_rate, id1, np.asarray([3,4]))
    s = synthetic_classification_file
    if local:
        s = synthetic_classification_local_file
    i = id1
    array_functions.plot_2d(data.x[i,:],data.y[i])
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)

def create_synthetic_step_transfer(file_dir='', dim=1):
    n_target = 100
    n_source = 100
    n = n_target + n_source
    sigma = .5
    data = data_class.Data()
    data.x = np.random.uniform(0,1,(n,dim))
    data.data_set_ids = np.zeros(n)
    data.data_set_ids[n_target:] = 1
    data.y = np.zeros(n)
    data.y[(data.data_set_ids == 0) & (data.x[:,0] >= .5)] = 2
    data.y += np.random.normal(0,sigma,n)
    data.set_defaults()
    data.is_regression = True
    if dim == 1:
        array_functions.plot_2d(data.x,data.y,data.data_set_ids)
    s = synthetic_step_transfer_file
    if dim > 1:
        s = synthetic_step_kd_transfer_file % dim
    if file_dir != '':
        s = file_dir + '/' + s
    helper_functions.save_object(s,data)


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

#1,9
def create_wine():
    red_file = 'wine/winequality-red.csv'
    white_file = 'wine/winequality-white.csv'
    field_names, red_data = load_csv(red_file, delim=';')
    white_data = load_csv(white_file, delim=';')[1]
    red_ids = np.zeros((red_data.shape[0],1))
    white_ids = np.ones((white_data.shape[0],1))
    red_data = np.hstack((red_data, red_ids))
    white_data = np.hstack((white_data, white_ids))
    wine_data = np.vstack((red_data,white_data))
    y = wine_data[:,-2]
    ids = wine_data[:,-1]
    x = wine_data[:,:-2]
    used_field_names = field_names[:-1]
    viz = False
    if viz:
        learner = make_learner()
        viz_features(x,y,ids,used_field_names,alpha=.01,learner=learner)

    feat_idx = 1
    data = data_class.Data()
    data.x = data.x = array_functions.standardize(x)
    #data.x = array_functions.vec_to_2d(x[:,feat_idx])

    data.y = y
    data.set_defaults()
    data.data_set_ids = ids
    data = data.rand_sample(.25, data.data_set_ids == 0)
    data = data.rand_sample(.1, data.data_set_ids == 1)
    data.is_regression = True
    #s = wine_file % ('-small-feat=' + str(feat_idx))
    s = wine_file % ('-small-' + str(data.p))
    helper_functions.save_object(s,data)



def create_boston_housing(file_dir=''):
    x_ind = 5
    domain_ind = 12
    boston_data = datasets.load_boston()
    data = data_class.Data()
    data.x = boston_data.data
    data.y = boston_data.target
    data.feature_names = list(boston_data.feature_names)
    data.set_defaults()
    data.is_regression = True
    s = boston_housing_raw_data_file
    x = data.x
    #y = data.y
    domain_ids = np.ones(x.shape[0])
    domain_ids = array_functions.bin_data(x[:,domain_ind],num_bins=4)
    x = np.delete(x,domain_ind,1)
    #viz_features(x,y,domain_ids,boston_data.feature_names)
    data.data_set_ids = domain_ids

    if boston_num_feats == 1:
        data.x = data.x[:,x_ind]
        data.x = array_functions.vec_to_2d(data.x)
        s = s % ''
    elif boston_num_feats >= data.x.shape[1]:
        data.x = array_functions.standardize(data.x)
        p = min(boston_num_feats,data.x.shape[1])
        s = s % ('-' + str(p))
    else:
        assert False
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
    domain_ids = np.squeeze(bike_data[:,domain_ind])
    #inds_to_keep = (used_field_names == 'temp') | (used_field_names == 'atemp')
    #bike_data = bike_data[:,inds_to_keep]
    #used_field_names = used_field_names[inds_to_keep]

    viz = True
    to_use = np.asarray([8,9,10,11])
    x = bike_data[:,to_use]
    used_field_names = used_field_names[to_use]
    y = bike_data[:,-1]
    if viz:
        learner = make_learner()
        viz_features(x,y,domain_ids,used_field_names,learner=learner)
    field_to_use = 1
    x = x[:,field_to_use]

    data = data_class.Data()
    data.is_regression = True
    data.x = array_functions.vec_to_2d(x)
    data.x = array_functions.standardize(data.x)
    data.y = y
    data.y = array_functions.normalize(data.y)
    data.set_defaults()
    data.data_set_ids = domain_ids

    s = bike_file % ('-feat=' + str(field_to_use))
    helper_functions.save_object(s,data)

    pass

if __name__ == "__main__":
    #create_boston_housing()
    #create_concrete()
    #create_synthetic_classification(local=True)
    #create_boston_housing()
    #create_bike_sharing()
    create_wine()
    from data_sets import create_data_split
    create_data_split.run_main()
