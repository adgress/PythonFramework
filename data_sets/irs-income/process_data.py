import numpy as np
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from utility.array_functions import find_first_element
import datetime
import math

file_name_house_size = 'house-size-2010.csv'
file_name_income = 'income-tax-2014.csv'
file_name_zip_lat_long = 'zipcode-lat-long.csv'

def get_zipcode_locs():
    loc_fields, loc_data = create_data_set.load_csv(file_name_zip_lat_long, dtype='string', return_data_frame=True)

    zipcode = loc_data.Zipcode.values.astype(np.int)
    zip_lat = loc_data.Lat.values.astype(np.float)
    zip_lon = loc_data.Long.values.astype(np.float)
    zip_loc = np.stack((zip_lon, zip_lat), 1)
    has_loc = np.isfinite(zip_loc.sum(1))
    d = dict(zip(zipcode[has_loc], zip_loc[has_loc, :]))
    return d

def get_zipcode_wages():
    income_fields, income_data = create_data_set.load_csv(file_name_income, dtype='string', return_data_frame=True)

    zipcode = income_data.ZipCode.values.astype(np.float)
    agi = income_data.AdjustedGrossIncome.values.astype('string')
    num_returns = income_data.NumberOfReturns.values.astype('string')
    i = find_first_element(zipcode, 90001)
    I = np.arange(i, zipcode.shape[0], 8)

    zipcode = zipcode[I].astype(np.int)
    agi = agi[I].astype(np.float)
    num_returns = num_returns[I].astype(np.float)
    '''
    I = agi < 5000000
    zipcode = zipcode[I]
    agi = agi[I]
    num_returns = num_returns[I]
    '''

    mean_income = agi / num_returns
    I = (num_returns > 50) & (mean_income < np.percentile(mean_income, 99.6))
    d = dict(zip(zipcode[I], mean_income[I]))
    return d

def get_zipcode_housing():
    housing_fields, housing_data = create_data_set.load_csv(file_name_house_size, dtype='string', return_data_frame=True)
    zipcodes = housing_data.ZIP.values.astype(np.float)
    totals = housing_data.Total.values.astype(np.float)
    households = housing_data.values[:,4:].astype(np.float)
    weight_vec = np.arange(1,8)
    sums = households.dot(weight_vec)
    mean_househoulds = sums / totals
    I = np.isfinite(mean_househoulds) & (totals > 100)
    d = dict(zip(zipcodes[I], mean_househoulds[I]))
    return d

zipcode_housing = get_zipcode_housing()
zipcode_locs = get_zipcode_locs()
zipcode_income = get_zipcode_wages()

zipcodes = set(zipcode_income.keys())
zipcodes.intersection_update(zipcode_locs.keys())
zipcodes.intersection_update(zipcode_housing.keys())

zipcode_array = np.zeros(len(zipcodes))
income_array = np.zeros(len(zipcodes))
locs = np.zeros((len(zipcodes), 2))
households = np.zeros(len(zipcodes))

for i, key in enumerate(zipcodes):
    zipcode_array[i] = key
    income_array[i] = zipcode_income[key]
    locs[i] = zipcode_locs[key]
    households[i] = zipcode_housing[key]

income_array = np.log(income_array)
income_array = array_functions.normalize(income_array)

households = array_functions.normalize(households)

locs[:,0] = array_functions.normalize(locs[:,0])
locs[:,1] = array_functions.normalize(locs[:,1])

#array_functions.plot_heatmap(locs, 10*income_array, sizes=50)
#array_functions.plot_heatmap(locs, households, sizes=50)
y = np.stack((income_array, households), 1)
print 'Num Used: ' + str(y.shape[0])
array_functions.plot_heatmap(
    locs,
    y,
    sizes=100,
    share_axis=True
)
I = np.random.choice(y.shape[0], 400, replace=False)
data = (locs[I,:], y[I], zipcode_array[I])
helper_functions.save_object('processed_data.pkl', data)

pass
