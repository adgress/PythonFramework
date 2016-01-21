from utility import helper_functions
from utility import array_functions
import numpy as np

def aggregate_g(results):
    x = results.results_list[0].prediction.x
    g = np.zeros(results.results_list[0].prediction.g.shape)
    for r in results.results_list:
        gi =  1 / (1+r.prediction.g)
        g = g + gi
    g /= len(results.results_list)
    return x,g


def run_main(results_file):
    results = helper_functions.load_object(results_file)
    for i, r  in enumerate(results.results_list):
        x,g = aggregate_g(r)
    pass

if __name__ == '__main__':
    results_file = '../base/synthetic_step_linear_transfer/LocalTransfer-NonParaHypTrans-l1-reg2.pkl'
    run_main(results_file)