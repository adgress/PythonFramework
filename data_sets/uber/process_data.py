import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from utility.array_functions import find_first_element
import datetime
import math

file_name_apr = 'uber-raw-data-apr14.csv'
file_name_sep = 'uber-raw-data-sep14.csv'

def to_date(date_str):
    a = date_str.split(' ')[0]
    a = a.split('/')
    month, day, year = [int(s) for s in a]
    d = datetime.date(year, month, day)
    return d


locs, y, ids = create_data_set.load_trip_data([file_name_apr, file_name_sep], None, 'Date/Time', np.asarray(['Lon', 'Lat']), [100, 100], plot_data=True)
y[:,0] /= y[:,0].max()
y[:,1] /= y[:,1].max()
locs[:,0] = array_functions.normalize(locs[:,0])
locs[:,1] = array_functions.normalize(locs[:,1])

I = (y.sum(1) > 0)
locs = locs[I,:]
y = y[I, :]
ids = ids[I]

data = (locs, y, ids)
helper_functions.save_object('processed_data.pkl', data)

pass
