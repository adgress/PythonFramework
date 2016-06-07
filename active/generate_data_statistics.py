from data import data as data_lib
from utility import helper_functions
from matplotlib import pyplot as plt
import math

adience_dir = 'data_sets/adience_aligned_cnn_1_per_instance_id'
wine_dir =  'data_sets/wine-red'
synthetic_dir = 'data_sets/synthetic_linear_reg500-50-1'
boston_housing_dir = 'data_sets/boston_housing'

class data_statistics(object):
    def __init__(self, mean=None, std=None, range=None):
        self.mean = mean
        self.std = std
        self.range = range

def gen_stats(data):
    mean = data.true_y.mean()
    std = data.true_y.std()
    range = [data.true_y.min(), data.true_y.max()]
    stats = data_statistics(mean, std, range)
    return stats

def plot_labels(data):
    n, bins, patches = plt.hist(data.true_y, 50, normed=1, facecolor='green', alpha=0.75)

if __name__ == '__main__':
    dirs = [
        synthetic_dir, boston_housing_dir, adience_dir, wine_dir
    ]
    titles = [
        'synthetic', 'housing', 'adience', 'wine'
    ]
    num_rows = 2
    num_cols = math.ceil(len(dirs)/float(num_rows))
    stats = []
    for i, d in enumerate(dirs):
        plt.subplot(2,2,i)
        data = helper_functions.load_object('../' + d + '/split_data.pkl').data
        stats.append(gen_stats(data))
        plot_labels(data)
        plt.title(titles[i])
    plt.show()
    pass