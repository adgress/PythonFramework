import numpy as np
import matplotlib.pyplot as plt
from utility import array_functions


def f_line(x):
    return x

def covariate_shift_data(x):
    n_sub = int(x.size*.6)
    x_source = x[:n_sub,:]
    x_target = x[-n_sub:, :]
    y_target = np.sin(8*x_target)
    y_source = np.sin(8*x_source)+.05
    #y_target = f_line(x_target)
    #y_source = f_line(x_source)
    return x_source, y_source, x_target, y_target

def domain_adaptation_data(x):
    x_source = x.copy()
    x_target = x.copy()
    y_target = (.5*x_target+1)*np.sin(8 * x_target)
    y_source = np.sin(8 * x_source) + .5
    # y_target = f_line(x_target)
    # y_source = f_line(x_source)
    return x_source, y_source, x_target, y_target

def model_shift_data(x):
    x_source = x
    x_target = x
    #y_source = (10*x)**2 - x
    #y_target = (10*x)**2 - 20*x - 4
    y_source = 4*np.sin(5*x)
    y_target = 4*np.sin(5*x) + x + 5*x**2 + 2
    return x_source, y_source, x_target, y_target

def smoothness_shift_data(x):
    x_source = x
    x_target = x
    y_source = np.sin(5*x)
    y_source[x_source > .3] += 10
    y_source[x_source > .6] -= 5
    y_target = -10*x**2- 1
    y_target[x_target > .3] -= 20
    y_target[x_target > .6] -= 10
    return x_source, y_source, x_target, y_target

def plot_shift(func, x, legend_labels=None):
    x_source, y_source, x_target, y_target = func(x)
    line1 = plt.plot(x_source, y_source, c='b')[0]
    line2 = plt.plot(x_target, y_target, c='r')[0]
    if legend_labels is not None:
        plt.legend([line1, line2], legend_labels)
    plt.ylabel('f(x)')

fig = plt.figure()
x = np.linspace(0, 1, 100)
x = np.expand_dims(x, 1)
ax1 = plt.subplot(2, 2, 1)
plot_shift(covariate_shift_data, x, ['Source', 'Target'])
plt.title('Covariate Shift')
plt.xticks([], [])
ax1 = plt.subplot(2, 2, 2)
plot_shift(domain_adaptation_data, x)
plt.title('Hypothesis Transfer')
plt.xticks([], [])
ax1 = plt.subplot(2, 2, 3)
plt.title('Location-Scale')
plot_shift(model_shift_data, x)
plt.xlabel('x')
ax1 = plt.subplot(2, 2, 4)
plt.title('Regional')
plot_shift(smoothness_shift_data, x)
plt.xlabel('x')
#plt.axis([0, 1, 0, 1])
gap = .1
plt.subplots_adjust(left=gap+.05, bottom=gap, right=1-gap, top=1-gap, wspace=0, hspace=.2)
array_functions.move_fig(fig, 500, 500)
plt.tight_layout()
plt.show(block=True)




