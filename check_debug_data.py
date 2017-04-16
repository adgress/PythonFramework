import numpy as np
from utility import helper_functions

data = helper_functions.load_object('debug_data.pkl')

A = data['A']
S = data['S']
y = data['y']

v = S.dot(y)
try:
    np.linalg.lstsq(A, v)
    print 'it worked!'
except:
    print 'error caught'
pass