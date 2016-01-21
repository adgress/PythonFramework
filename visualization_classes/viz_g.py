from utility import helper_functions
from utility import array_functions
import numpy as np

def aggregate_g(results):
    x = results.results_list[0].prediction.linspace_x
    g = np.zeros(results.results_list[0].prediction.linspace_g.shape)
    for r in results.results_list:
        gi =  1 / (1+r.prediction.linspace_g)
        g = g + gi
    g /= len(results.results_list)
    return x, g


def run_main(results_file):
    results = helper_functions.load_object(results_file)
    x_list = []
    g_list = []
    y_axes = [-4,4]
    for i, r  in enumerate(results.results_list):
        x, g = aggregate_g(r)
        x_list.append(x)
        g_list.append(g)
    array_functions.plot_line_sub(x_list, g_list, title=results_file, y_axes=y_axes)
    x = 1

if __name__ == '__main__':
    dir = '../base/synthetic_step_linear_transfer/'
    #s = 'LocalTransfer-NonParaHypTrans-l1-reg2.pkl'
    #s = 'LocalTransfer-no_reg-NonParaHypTrans-reg2.pkl'
    s = 'LocalTransferDelta.pkl'
    run_main(dir + s)