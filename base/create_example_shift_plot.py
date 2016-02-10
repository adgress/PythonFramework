from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import matplotlib.pyplot as plt
from data import data as data_lib
from base import transfer_project_configs as configs_lib
import math

def step(x):
    y = 2*x + 1
    I = x > .5
    y[I] += 2
    return y

def run_main():
    target_func = lambda x: x
    source_funcs = [
        lambda x: x + 1,
        lambda x: 3*x + 1,
        lambda x: x**3 + 1,
        step
    ]
    titles = [
        'Constant Shift',
        'Linear Shift',
        'Nonlinear Shift',
        'Step Shift'
    ]
    #x = np.linspace(0,1,10)
    x = np.asarray([0,.1,.2,.3,.4,.5,.51,.6,.7,.8,.9,1.0])
    #num_rows = len(source_funcs)
    num_rows = 2
    num_cols = 2
    fig = plt.figure()
    for i, source_f in enumerate(source_funcs):
        subplot_idx = i + 1
        plt.subplot(num_rows,num_cols,subplot_idx)
        plt.title(titles[i])
        y = source_f(x)
        array_functions.plot_line_sub([x,x],[x,y],title=None,y_axes=None,fig=fig,show=False)
        a = plt.gca()
    left,right,top,bottom = (.05,.95,.95,.05)
    fig.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
    plt.show()
    pass



if __name__ == '__main__':
    run_main()