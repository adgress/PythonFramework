from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import matplotlib.pyplot as plt
from data import data as data_lib
from base import transfer_project_configs as configs_lib
import math
from scipy.special import expit as sigmoid

def step(x):
    y = 2*x + 1
    I = x > .5
    y[I] += 2
    return y

def run_main():
    f_ground_truth = lambda x: -np.cos(5*x) + x**2 + 3
    f_target = lambda x: 2.5*x + 2
    f_source = lambda x: -np.cos(5*x) + 2 + -x
    f_adapt = lambda x: f_ground_truth(x) - f_source(x)
    f_mixture = lambda x: sigmoid(10*(x - .5))
    f_final = lambda x: (1-f_mixture(x))*f_target(x) + f_mixture(x)*(f_source(x) + f_adapt(x))
    source_funcs = [
        f_target,
        f_source,
        f_adapt,
        f_mixture,
        f_final,
    ]
    titles = [
        r'Target: $f_T(x)$',
        r'Source: $f_S(x)$',
        r'Adaptation: $b(f_S(x), x)$',
        r'Mixture: $\alpha(x)$',
        r'Final: $ (1- \alpha(x))f_T(x) + \alpha(x) b(f_S(x), x)$'
    ]
    plot_ground_truth = [
        True,
        False,
        False,
        False,
        True
    ]
    x = np.linspace(0,1,100)
    #x = np.asarray([0,.1,.2,.3,.4,.5,.51,.6,.7,.8,.9,1.0])
    #num_rows = len(source_funcs)
    num_rows = 1
    num_cols = len(titles)
    fig = plt.figure()
    for i, source_f in enumerate(source_funcs):
        subplot_idx = i + 1
        plt.subplot(num_rows,num_cols,subplot_idx)
        plt.title(titles[i])
        y = source_f(x)
        x_all = [x]
        y_all = [y]
        if plot_ground_truth[i]:
            x_all.append(x)
            y_all.append(f_ground_truth(x))
        array_functions.plot_line_sub(x_all,y_all,title=None,y_axes=None,fig=fig,show=False)
        a = plt.gca()
    left,right,top,bottom = (.05,.95,.9,.1)
    fig.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
    array_functions.move_fig(fig, 1500, 300)
    plt.show()
    pass



if __name__ == '__main__':
    run_main()