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
from copy import deepcopy
from create_synthetic_data import *
import itertools

ng_a = np.asarray(0)
ng_c = np.asarray(range(1,6))
ng_m = np.asarray(6)
ng_r = np.asarray(range(7,11))
ng_s = np.asarray(range(11,15))
ng_o = np.asarray(15)
ng_t = np.asarray(range(16,20))


#Abalone data - looks promising
#pair_target = 5
#pair_source = 6

pair_target = 56
pair_source = 59

max_features=2000
pca_feats=20
#max_features=100
boston_num_feats = np.inf
concrete_num_feats = np.inf

boston_housing_raw_data_file = 'boston_housing%s/raw_data.pkl'
ng_raw_data_file = '20ng-%d/raw_data.pkl' % max_features
concrete_file = 'concrete%s/raw_data.pkl'
bike_file = 'bike_sharing%s/raw_data.pkl'
wine_file = 'wine%s/raw_data.pkl'
adience_aligned_cnn_file = 'adience_aligned_cnn/raw_data.pkl'
adience_aligned_cnn_1_per_instance_id_file = 'adience_aligned_cnn_1_per_instance_id/raw_data.pkl'
drosophila_file = 'drosophilia/raw_data.pkl'
kc_housing_file = 'kc_housing/raw_data.pkl'
pollution_file = 'pollution/raw_data.pkl'

def pair_file(i,j):
    s = 'pair_data_' + str(i) + '_' + str(j) + '/raw_data.pkl'
    return s

def make_learner():
    from methods.method import NadarayaWatsonMethod
    from loss_functions.loss_function import MeanSquaredError
    learner = NadarayaWatsonMethod()
    learner.configs.cv_loss_function = MeanSquaredError()
    learner.configs.loss_function = MeanSquaredError()
    return learner

def create_and_save_data(x,y,domain_ids,file):
    data = data_class.Data()
    data.x = array_functions.vec_to_2d(x)
    data.y = y
    data.set_train()
    data.set_target()
    data.set_true_y()
    data.is_regression = True
    data.data_set_ids = domain_ids
    helper_functions.save_object(file,data)

import datetime
from utility import array_functions
from utility.array_functions import find_first_element

def to_date(date_str):
    a = date_str.split(' ')[0]
    a = a.split('/')
    month, day, year = [int(s) for s in a]
    d = datetime.date(year, month, day)
    return d

def quantize_loc(locs, num_bins):
    bins = np.linspace(locs.min(), locs.max(), num_bins)
    idx = np.digitize(locs, bins)
    idx -= 1
    return idx


def bin_to_idx(bin_loc, resolution):
    return np.ravel_multi_index((bin_loc[0], bin_loc[1]), resolution)

def load_trip_data(file_names, y_names, time_name, loc_names, resolution=np.asarray([20,20]), plot_data=True):
    resolution = np.asarray(resolution)
    feat_names = None
    data = None
    for file_name in file_names:
        curr_feat_names, curr_data = load_csv(
            file_name,
            True,
            dtype='str',
            delim=',',
            num_rows=1000000000
        )
        if feat_names is None:
            feat_names = curr_feat_names
            data = curr_data
            continue
        assert (feat_names == curr_feat_names).all()
        data = np.vstack((data, curr_data))
    locs = data[:, array_functions.find_set(feat_names, loc_names)]
    y_inds = None
    if y_names is not None:
        y_inds = array_functions.find_set(feat_names, y_names).nonzero()[0]
        y = data[:, y_inds].astype(np.float)
    else:
        y = np.ones(data.shape[0])
    date_strs = data[:, find_first_element(feat_names, time_name)]
    date_str_to_idx = dict()
    date_ids = np.zeros(data.shape[0])
    for i, date_str in enumerate(date_strs):
        date_obj = to_date(date_str)
        date_str_to_idx[date_str] = date_obj.toordinal()
        date_ids[i] = date_obj.toordinal()
    date_ids = date_ids.astype(np.int)



    min_date_id = date_ids.min()
    max_date_id = date_ids.max()
    num_days = max_date_id - min_date_id + 1
    dates_idx = date_ids - min_date_id
    num_locations = np.prod(resolution)
    trip_counts = 0 * np.ones((num_days, num_locations))
    locs = locs.astype(np.float)
    p_min = .3
    p_max = .7
    is_in_range = array_functions.is_in_percentile(locs[:, 0], p_min, p_max) & array_functions.is_in_percentile(locs[:, 1], p_min, p_max)
    locs = locs[is_in_range, :]
    dates_idx = dates_idx[is_in_range]
    x_bins = quantize_loc(locs[:, 0], resolution[0])
    y_bins = quantize_loc(locs[:, 1], resolution[1])
    #array_functions.plot_2d(locs[I,0],locs[I,1])
    xy_bins = list(itertools.product(range(resolution[0]), range(resolution[1])))
    for x_idx,y_idx in xy_bins:
        is_in_cell = (x_bins == x_idx) & (y_bins == y_idx)
        trips_in_cell = dates_idx[is_in_cell]
        trip_dates, trips_per_date = np.unique(trips_in_cell, return_counts=True)
        bin_idx = bin_to_idx([x_idx, y_idx], resolution)
        trip_counts[trip_dates, bin_idx] = trips_per_date
    #y = trip_counts[[0, 3], :].T
    tuesday_saturday_idx = np.asarray([0, 4])
    first_tuesday_idx = np.asarray([0, 154])

    #y = trip_counts[first_tuesday_idx + 0, :].T
    '''
    y1 = trip_counts[:30,:].sum(0)
    y2 = trip_counts[154:, :].sum(0)
    '''
    y1 = trip_counts[:30:7, :].mean(0)
    y2 = trip_counts[4:30:7, :].mean(0)
    y = np.stack((y1,y2), 1)

    #y[y > 100] = 0
    #y[y > 5000] = 0
    #y[y == y.max()] == 0
    y = np.log(y)
    if plot_data:
        array_functions.plot_heatmap(np.asarray(xy_bins), y, sizes=50)
    return np.asarray(xy_bins, dtype=np.float), y, np.asarray([str(xy) for xy in xy_bins])

import csv
def load_csv(file, has_field_names=True, dtype='float', delim=',',converters=None,
             num_rows=None, return_data_frame=False):
    nrows = 1
    if not has_field_names:
        nrows = 0
        all_field_names = None
    else:
        all_field_names = pd.read_csv(file,nrows=nrows,dtype='string',sep=delim)
        all_field_names = np.asarray(all_field_names.keys())
    if num_rows is not None or return_data_frame:
        assert converters is None
        data = pd.read_csv(
            file,
            skiprows=0,
            dtype='string',
            sep=delim,
            nrows=num_rows
        )
        if not return_data_frame:
            data = data.values
    else:
        data = np.loadtxt(
            file,
            skiprows=nrows,
            delimiter=delim,
            usecols=None,
            dtype=dtype,
            converters=converters,
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
    months_to_use = np.asarray([6,7,8])
    #months_to_use = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12])
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

    from methods import method
    learner = method.NadarayaWatsonMethod()
    viz_features(x,y,domain_ids,field_names,learner=learner)
    pass

def load_pair(num):
    file_base = 'pair_data/pair%s.txt'
    file = file_base % str(num).zfill(4)
    data = pd.read_csv(file,skiprows=0,delim_whitespace=True,dtype='float')
    return np.asarray(data)[:,0:2]

def create_pair_82_83():
    create_pair(82,83,y_col=1)

def create_pair(i,j,y_col):
    file = pair_file(i,j)
    data_i = load_pair(i)
    data_j = load_pair(j)
    data_all = np.vstack((data_i,data_j))
    x = data_all[:,1-y_col]
    y = data_all[:,y_col]
    domain_ids = np.zeros(data_i.shape[0] + data_j.shape[0])
    domain_ids[data_i.shape[0]:] = 1
    #viz_features(x,y,domain_ids,learner=make_learner())
    viz_features(x,y,domain_ids,learner=None)
    create_and_save_data(x,y,domain_ids,file)

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

#4-7 for domains?
def create_energy():
    file = 'energy/ENB2012_data.csv'
    field_names, energy_data = load_csv(file)
    domain_ids = energy_data[:,4]
    x = energy_data
    y = energy_data[:,-2]
    from methods import method
    #learner = method.NadarayaWatsonMethod()
    learner = None
    viz_features(x,y,domain_ids,field_names, learner=learner)
    pass

def create_drosophila():
    data = helper_functions.load_object('drosophilia/processed_data.pkl')
    x, y = data
    y = np.reshape(y,y.shape[0])
    I = np.random.choice(x.shape[0], size=500, replace=False)
    x = x[I,:]
    y = y[I]
    data = data_class.Data()
    data.x = x
    data.y = y
    data.set_train()
    data.set_target()
    data.set_true_y()
    data.is_regression = True

    helper_functions.save_object(drosophila_file, data)

#0 - 1 to 3
#3 - 1 to 3
#4 - 1 to 3
#5 -
def create_concrete(transfer = False):
    file = 'concrete/Concrete_Data.csv'
    used_field_names, concrete_data = load_csv(file)

    data = data_class.Data()
    t = ''
    if transfer:
        feat_ind = 0
        domain_ind = (used_field_names == 'age').nonzero()[0][0]
        ages = concrete_data[:,domain_ind]
        domain_ids = np.zeros(ages.shape)
        domain_ids[ages < 10] = 1
        domain_ids[(ages >= 10) & (ages <= 28)] = 2
        domain_ids[ages > 75] = 3
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
        data.data_set_ids = domain_ids
    else:
        data.x = concrete_data[:,0:-1]

    data.y = concrete_data[:,-1]
    data.set_train()
    data.set_target()
    data.set_true_y()
    data.is_regression = True

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

#1,9
WINE_TRANSFER = 1
WINE_RED = 2
WINE_WHITE = 3
def create_wine(data_to_create=WINE_RED):
    red_file = 'wine/winequality-red.csv'
    white_file = 'wine/winequality-white.csv'
    field_names, red_data = load_csv(red_file, delim=';')
    white_data = load_csv(white_file, delim=';')[1]

    if data_to_create == WINE_TRANSFER:
        red_ids = np.zeros((red_data.shape[0],1))
        white_ids = np.ones((white_data.shape[0],1))
        red_data = np.hstack((red_data, red_ids))
        white_data = np.hstack((white_data, white_ids))
        wine_data = np.vstack((red_data,white_data))

        ids = wine_data[:,-1]
        x = wine_data[:,:-2]
        y = wine_data[:,-2]
        used_field_names = field_names[:-1]
        viz = True
        if viz:
            learner = make_learner()
            #learner = None
            viz_features(x,y,ids,used_field_names,alpha=.01,learner=learner)
        suffix = 'transfer'
    else:
        if data_to_create == WINE_RED:
            wine_data = red_data
            suffix = 'red'
        elif data_to_create == WINE_WHITE:
            wine_data = white_data
            suffix = 'white'
        else:
            assert False

        ids = None
        x = wine_data[:,:-1]
        y = wine_data[:,-1]
        used_field_names = field_names[:-1]
    data = data_class.Data()
    data.x = data.x = array_functions.standardize(x)
    if data_to_create == WINE_TRANSFER:
        pass
        #feat_idx = 1
        #data.x = array_functions.vec_to_2d(x[:,feat_idx])

    data.y = y
    data.set_train()
    data.set_target()
    data.set_true_y()
    data.data_set_ids = ids
    data.is_regression = True
    '''
    data = data.rand_sample(.25, data.data_set_ids == 0)
    data = data.rand_sample(.1, data.data_set_ids == 1)
    s = wine_file % ('-small-' + str(data.p))
    '''
    s = wine_file % ('-' + suffix)
    helper_functions.save_object(s,data)



def create_boston_housing(file_dir=''):
    boston_data = datasets.load_boston()
    data = data_class.Data()
    data.x = boston_data.data
    data.y = boston_data.target
    data.feature_names = list(boston_data.feature_names)

    data.set_train()
    data.set_target()
    data.set_true_y()
    data.is_regression = True
    s = boston_housing_raw_data_file
    x = data.x
    y = data.y
    create_transfer_data = False
    create_y_split = True
    if create_y_split:
        from base import transfer_project_configs as configs_lib
        pc = configs_lib.ProjectConfigs()
        main_configs = configs_lib.MainConfigs(pc)
        learner = main_configs.learner
        learner.quiet = True
        learner.target_learner[0].quiet = True
        learner.source_learner.quiet = True
        learner.g_learner.quiet = False
        domain_ids = array_functions.bin_data(data.y, num_bins=2)
        data.data_set_ids = domain_ids
        data.is_train[:] = True
        corrs = []
        for i in range(x.shape[1]):
            corrs.append(scipy.stats.pearsonr(x[:,i], y)[0])
        learner.train_and_test(data)
        print 'Just playing with data - not meant to save it'
        for i, name in enumerate(data.feature_names):
            v = learner.g_learner.g[i]
            if abs(v) < 1e-6:
                v = 0
            print name + ': ' + str(v)
        exit()
    elif create_transfer_data:
        x_ind = 5
        domain_ind = 12
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
    else:
        s %= ''
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
        #learner = make_learner()
        learner = None
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

def create_kc_housing():
    file = 'kc_housing/processed_data.pkl'
    x, y = helper_functions.load_object(file)
    data = data_class.Data(x, y)
    data.is_regression = True
    s = kc_housing_file
    helper_functions.save_object(s, data)

def create_time_series(label_to_use=0, series_to_use=0, num_instances=None, normalize_x=False, save_data=True, name='CO2_emissions'):
    file = name + '/processed_data.pkl'
    all_data = []
    for i in series_to_use:
        y, ids = helper_functions.load_object(file)
        y_to_use = y[:, i, :]
        print str(i) + ': ' + ids[i]
        data = data_class.TimeSeriesData(y_to_use, np.asarray([ids[i]]))
        data.is_regression = True
        data.keep_series(label_to_use)
        data = data.get_min_range()
        data.smooth_missing()
        data = data.get_nth(7)
        data.reset_x()
        data.x = data.x.astype(np.float)
        if num_instances is not None:
            data = data.get_range([0, num_instances])
        data = data.get_range([1000, 1500])
        if normalize_x:
            data.x -= data.x.min()
            data.x /= data.x.max()
        data = data.create_data_instance()
        try:
            if len(series_to_use) > 1:
                data.data_set_ids[:] = i
        except:
            pass
        all_data.append(data)
        # perc_used = data.get_perc_used()
    data = all_data[0]
    del all_data[0]
    for di in all_data:
        data.combine(di)
    if num_instances is not None:
        pass
        s = name + '-%s-%d' % (str(series_to_use), num_instances)
    else:
        s = name + '-%s' % str(series_to_use)
    if normalize_x:
        s += '-norm'
    s += '/raw_data.pkl'
    # array_functions.plot_2d_sub_multiple_y(data.x, data.y, title=None, sizes=10)
    array_functions.plot_2d_sub(data.x, data.y, data_set_ids=data.data_set_ids, title=None, sizes=10)
    if save_data:
        helper_functions.save_object(s, data)

def create_drought(label_to_use=0, series_to_use=0, num_instances=None, normalize_x=False, save_data=True):
    file = 'drought/processed_data.pkl'
    y, ids = helper_functions.load_object(file)
    y_to_use = y[:, series_to_use, :]
    print str(series_to_use) + ': ' + ids[series_to_use]
    data = data_class.TimeSeriesData(y_to_use, np.asarray([ids[series_to_use]]))
    data.is_regression = True
    data.keep_series(label_to_use)
    data = data.get_min_range()
    data.smooth_missing()
    data.x = data.x.astype(np.float)
    if num_instances is not None:
        data = data.get_range([0, num_instances])
    if normalize_x:
        data.x -= data.x.min()
        data.x /= data.x.max()
    data = data.create_data_instance()
    # perc_used = data.get_perc_used()
    if num_instances is not None:
        pass
        s = 'drought-%d-%d' % (series_to_use, num_instances)
    else:
        s = 'drought-%d' % series_to_use
    if normalize_x:
        s += '-norm'
    s += '/raw_data.pkl'
    # array_functions.plot_2d_sub_multiple_y(data.x, data.y, title=None, sizes=10)
    array_functions.plot_2d_sub(data.x, data.y, data_set_ids=data.data_set_ids, title=None, sizes=10)
    if save_data:
        helper_functions.save_object(s, data)


def create_pollution(labels_to_use=np.arange(2), series_to_use=[0], num_instances=None, normalize_xy=True, save_data=True):
    #series_to_use = 1
    file = 'pollution/processed_data.pkl'
    y, ids = helper_functions.load_object(file)
    data = None
    label_names = []
    for i in range(y.shape[1]):
        print str(i) + '-' + ids[i] + ': ' + str(y[0,i,:])
    for idx, s in enumerate(series_to_use):
        for label in labels_to_use:
            label_names.append(str(label) + '-' + ids[s])
        y_to_use = y[:, s, :]
        print str(s) + ': ' + ids[s]
        time_series_data = data_class.TimeSeriesData(y_to_use, np.asarray([ids[s]]))
        time_series_data.is_regression = True
        time_series_data.keep_series(labels_to_use)
        time_series_data = time_series_data.get_min_range()
        time_series_data.smooth_missing()
        time_series_data.x = time_series_data.x.astype(np.float)
        if num_instances is not None:
            time_series_data = time_series_data.get_range([0, num_instances])

        if normalize_xy:
            time_series_data.reset_x()
            #time_series_data.normalize_y()

        curr_data = time_series_data.create_data_instance()
        curr_data.data_set_ids += idx * labels_to_use.size
        if data is None:
            data = curr_data
        else:
            data.combine(curr_data)
    data.label_names = label_names
    #perc_used = data.get_perc_used()
    if num_instances is not None:
        s = 'pollution-%s-%s' % (str(series_to_use), str(num_instances))
    else:
        s = 'pollution-%d' % series_to_use
    if normalize_xy:
        s+= '-norm'
    s += '/raw_data.pkl'
    #array_functions.plot_2d_sub_multiple_y(data.x, data.y, title=None, sizes=10)
    array_functions.plot_2d_sub(data.x, data.y, data_set_ids=data.data_set_ids, title=None, sizes=10)
    if save_data:
        helper_functions.save_object(s, data)

def create_spatial_data(dir='climate-month'):
    file = dir + '/processed_data.pkl'
    locs, y, ids = helper_functions.load_object(file)
    y = y.T
    is_missing_loc = (~np.isfinite(locs)).any(1)
    locs = locs[~is_missing_loc,:]
    y = y[~is_missing_loc,:]
    ids = ids[~is_missing_loc]
    data = data_class.Data(locs, y)
    data.multilabel_to_multisource()
    s = dir + '/raw_data.pkl'
    helper_functions.save_object(s, data)

if __name__ == "__main__":
    #create_boston_housing()
    #create_concrete()
    #create_bike_sharing()
    #create_wine()
    #create_energy()
    #create_forest_fires()
    #create_synthetic_step_linear_transfer()
    #create_synthetic_delta_linear_transfer()
    #create_synthetic_cross_transfer()
    #create_synthetic_curve_transfer()
    #create_pair_82_83()
    #create_pair(pair_target,pair_source,0)
    #create_wine()
    #create_drosophila()
    #create_kc_housing()
    create_pollution(series_to_use=np.asarray([60, 71]), num_instances=500, save_data=True, normalize_xy=True)
    #create_spatial_data()
    #create_spatial_data('uber')
    #create_spatial_data('irs-income')
    #create_concrete(True)
    #create_boston_housing()
    '''
    for i in range(200):
        create_pollution(labels_to_use=[0, 1], series_to_use=i, num_instances=500, save_data=False)
    '''
    #create_time_series(label_to_use=0, series_to_use=range(8), save_data=False)
    #create_time_series(label_to_use=0, series_to_use=[0,3], save_data=False)
    from data_sets import create_data_split
    create_data_split.run_main()
