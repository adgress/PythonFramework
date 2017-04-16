import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from datetime import date
from matplotlib import pyplot as pl
from data import data as data_lib

def keep_subset(I, num_to_keep):
    inds = I.nonzero()[0]
    if num_to_keep > inds.size:
        return I
    inds_to_keep = np.random.choice(inds, num_to_keep, replace=False)
    v = array_functions.false(I.size)
    v[inds_to_keep] = True
    return v

def get_date(s):
    s = s[0]
    year = s[:4]
    month = s[4:6]
    day = s[6:8]
    d = date(int(year), int(month), int(day))
    return d

create_geospatial_data = True
split_date = False
file_name = 'kc_house_data.csv'
save_data = True
sampled_size = 1000

feat_names, data = create_data_set.load_csv(file_name, True, dtype='str', delim=',')
y_name = 'price'
y_ind = array_functions.find_first_element(feat_names, y_name)
y = data[:, y_ind].astype(np.float)
y /= 100000
suffix = ''
if create_geospatial_data:
    x_feats = ['long', 'lat']
    x_feat_inds = array_functions.find_set(feat_names, x_feats)
    x = data[:, x_feat_inds]
    x = array_functions.remove_quotes(x)
    x = x.astype(np.float)

    x[:, 0] = array_functions.normalize(x[:, 0])
    x[:, 1] = array_functions.normalize(x[:, 1])
    I = array_functions.is_in_percentile(x[:, 0], .01, .99)
    I &= array_functions.is_in_percentile(x[:, 1], .01, .99)
    x = x[I, :]
    y = y[I]
    data = data[I, :]

    if split_date:
        dates = array_functions.remove_quotes(data[:, feat_names == 'date'])
        date_objs = []
        for d in dates:
            date_obj = get_date(d)
            date_objs.append(date_obj)
        min_date = min(date_objs)
        day_deltas = np.zeros(len(date_objs))
        months = np.zeros(len(date_objs))
        for i, d in enumerate(date_objs):
            day_deltas[i] = (d - min_date).days
            months[i] = d.month
        ids = day_deltas
        #I1 = day_deltas < 30
        #I2 = array_functions.in_range(day_deltas, 120, 150)
        I1 = months == 7
        I2 = months == 11
        suffix = 'date'
    else:
        #id_feat = 'bedrooms'
        #id_feat = 'yr_built'
        #id_feat = 'bathrooms'
        id_feat = 'floors'
        #id_feat = 'sqft_living'
        #id_feat = 'waterfront'
        #id_feat = 'condition'
        ids = array_functions.remove_quotes(data[:, feat_names == id_feat])
        ids = ids.astype(np.float)
        ids = np.squeeze(ids)
        I1 = ids < ids.mean()
        I2 = ids > ids.mean()
        suffix = id_feat


    I = np.isfinite(y)
    if sampled_size is not None:
        I1 = keep_subset(I1, sampled_size)
        I2 = keep_subset(I2, sampled_size)
        '''
        sampled = np.random.choice(I.size, sampled_size, replace=False)
        sampled_boolean = array_functions.false(I.size)
        sampled_boolean[sampled] = True
        I &= sampled_boolean
        I1 &= I
        I2 &= I
        '''

    print 'n1: ' + str(I1.sum())
    print 'n2: ' + str(I2.sum())

    fig1 = pl.figure(3)

    dot_size = 30
    array_functions.plot_heatmap(x[I1, :], y[I1], sizes=dot_size, alpha=1, subtract_min=False, fig=fig1)
    pl.title('Values 1')
    fig2 = pl.figure(4)
    array_functions.plot_heatmap(x[I2, :], y[I2], sizes=dot_size, alpha=1, subtract_min=False,fig=fig2)
    pl.title('Values 2')
    array_functions.move_fig(fig1, 500, 500, 2000, 100)
    array_functions.move_fig(fig2, 500, 500, 2600, 100)
    pl.show(block=True)

    data = (x,y)
    x = np.vstack((x[I1, :], x[I2, :]))
    data_set_ids = np.hstack((np.zeros(I1.sum()), np.ones(I2.sum())))

    y = np.hstack((y[I1], y[I2]))


    data = data_lib.Data(x, y)
    data.x[:, 0] = array_functions.normalize(data.x[:, 0])
    data.x[:, 1] = array_functions.normalize(data.x[:, 1])
    data.data_set_ids = data_set_ids
    print 'n-all: ' + str(data.y.size)
    if save_data:
        s = '../kc-housing-spatial'
        if suffix != '':
            s += '-' + suffix
        helper_functions.save_object(s + '/raw_data.pkl', data)

else:
    feats_to_clear = ['id', 'date', 'yr_renovated', 'zipcode', 'lat', 'long']
    clear_idx = array_functions.find_set(feat_names, feats_to_clear + [y_name])
    x = data[:, ~clear_idx]
    x = array_functions.remove_quotes(x)
    x = x.astype(np.float)
    data = (x,y)
    helper_functions.save_object('processed_data.pkl', data)

