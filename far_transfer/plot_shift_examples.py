import numpy as np
import matplotlib.pyplot as plt
from utility import array_functions

def f_line(x):
    return x

def covariate_shift_data(x):
    n_sub = int(x.size*.4)
    x_source = x[:n_sub,:]
    x_target = x[-n_sub:, :]
    y_target = f_line(x_target)
    y_source = f_line(x_source)
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

def plot_shift(func, x):
    x_source, y_source, x_target, y_target = func(x)
    plt.plot(x_source, y_source, c='b')
    plt.plot(x_target, y_target, c='r')

fig = plt.figure()
x = np.linspace(0, 1, 100)
x = np.expand_dims(x, 1)
ax1 = plt.subplot(3, 1, 1)
plot_shift(covariate_shift_data, x)
plt.title('Covariate Shift')
ax1 = plt.subplot(3, 1, 2)
plt.title('Location-Scale Shift')
plot_shift(model_shift_data, x)
ax1 = plt.subplot(3, 1, 3)
plt.title('Smoothness Shift')
plot_shift(smoothness_shift_data, x)
plt.show(block=True)




