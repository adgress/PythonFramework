import os
from os import path
from data_sets.create_data_set import load_csv, quantize_loc
data_dir = 'C:/Users/adgress/Desktop/cabspottingdata'
import numpy as np
from utility import array_functions
from utility.array_functions import normalize, in_range
from utility import helper_functions
from datetime import datetime
import matplotlib.pylab as pl
from data import data as data_lib
from data_sets import create_data_set

def replace_invalid_strings(x):
    try:
        float(x)
    except:
        return 'nan'
    return x

def remove_quotations(x):
    return x[1:-1]

vec_remove_quotations = np.vectorize(remove_quotations)
vec_replace = np.vectorize(replace_invalid_strings)

def get_zipcode_locations():
    file = '../zipcodes/zipcodes.txt'
    fields, zipcode_data = create_data_set.load_csv(file, has_field_names=True, dtype=np.float)
    locs = zipcode_data[:, [2,1]]
    zip_codes = zipcode_data[:,0].astype(np.int)
    zipcode_location_map = dict()
    for z, loc in zip(zip_codes, locs):
        zipcode_location_map[z] = loc
    return zipcode_location_map


file = 'Zip_Zhvi_AllHomes.csv'
data_fields, string_data = create_data_set.load_csv(file, has_field_names=True,dtype='string')
zip_code = vec_remove_quotations(string_data[:, 1]).astype(np.int)
state = vec_remove_quotations(string_data[:,3])
year1_idx = array_functions.find_first_element(data_fields, '1996-04')
year2_idx = array_functions.find_first_element(data_fields, '2017-02')
pricing_data =  string_data[:, [year1_idx, year2_idx]]
pricing_data = vec_replace(pricing_data).astype(np.float)
zipcode_location_map = get_zipcode_locations()
locations = np.zeros((zip_code.size,2))
for i, z in enumerate(zip_code):
    if z not in zipcode_location_map:
        print 'missing zipcode: ' + str(z)
        locations[i,:] = np.nan
        continue
    locations[i,:] = zipcode_location_map[z]

I = np.isfinite(year1_idx) & np.isfinite(year2_idx) & np.isfinite(locations[:,0])
#I = I & (state == 'CA')
viz = True
print 'n: ' + str(I.sum())
pricing_data[:] = 1
if viz:
    fig1 = pl.figure(3)
    array_functions.plot_heatmap(locations[I,:], pricing_data[I,0], sizes=30, alpha=1, subtract_min=False, fig=fig1)
    fig2 = pl.figure(4)
    array_functions.plot_heatmap(locations[I,:], pricing_data[I,1], sizes=30, alpha=1, subtract_min=False, fig=fig2)
    pl.show(block=True)

print ''
