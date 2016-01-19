
from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import pandas as pd

def train_on_data(x,y,domain_ids,learner):
    data = data_class.Data()
    data.is_regression = True
    data.x = array_functions.vec_to_2d(x)
    data.y = y
    data.set_defaults()
    x_plot = np.zeros((0,1))
    y_plot = np.zeros(0)
    ids_plot = np.zeros(0)
    density_plot = np.zeros(0)
    x_test = scipy.linspace(x.min(),x.max(),100)
    x_test = array_functions.vec_to_2d(x_test)
    data_test = data_class.Data()
    data_test.is_regression = True
    data_test.x = x_test
    data_test.y = np.zeros(x_test.shape[0])
    data_test.y[:] = np.nan

    from methods import density
    kde = density.KDE()

    max_n = 200.0
    for i in np.unique(domain_ids):
        I = domain_ids == i
        data_i = data.get_subset(I)
        if data_i.n > max_n:
            data_i = data_i.rand_sample(max_n/data_i.n)
        learner.train_and_test(data_i)
        o = learner.predict(data_test)
        x_plot = np.vstack((x_plot,x_test))
        y_plot = np.hstack((y_plot,o.y))
        ids_plot = np.hstack((ids_plot,np.ones(100)*i))

        kde.train_and_test(data_i)
        dens = kde.predict(data_test)
        dens.y = dens.y / dens.y.max()
        density_plot = np.hstack((density_plot,dens.y))
    return x_plot,y_plot,ids_plot,density_plot


def viz_features(x,y,domain_ids,feature_names=None,alpha=.1,learner=None):
    #y = array_functions.normalize(y)
    for i in range(x.shape[1]):
        xi = x[:,i]
        yi = y
        ids_i = domain_ids
        title = str(i)
        density = None
        if feature_names is not None:
            title = str(i) + ': ' + feature_names[i]
        if learner is not None:
            xi,yi,ids_i,density = train_on_data(xi,yi,domain_ids,learner)
            density = density*100 + 1
            I = array_functions.is_invalid(density)
            density[I] = 200
            alpha = 1
        array_functions.plot_2d_sub(xi,yi,alpha=alpha,title=title,data_set_ids=ids_i,sizes=density)
        pass


def run_main():
    import create_data_set
    from methods import method
    learner = method.NadarayaWatsonMethod()
    #s = create_data_set.synthetic_step_transfer_file
    #s = create_data_set.boston_housing_raw_data_file % '-13'
    #s = create_data_set.concrete_file % '-7'
    s = create_data_set.synthetic_classification_local_file
    learner = None
    data = helper_functions.load_object(s)
    viz_features(data.x,data.y,data.data_set_ids,learner=learner)

if __name__ == "__main__":
    run_main()