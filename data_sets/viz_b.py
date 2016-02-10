from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
import matplotlib.pyplot as plt
from data import data as data_lib
from base import transfer_project_configs as configs_lib
import math

def viz(pc, fig=None, show_histogram=False, show=True):
    import create_data_set
    from methods import method
    source_learner = method.NadarayaWatsonMethod()
    target_learner = method.NadarayaWatsonMethod()
    #pc = configs_lib.ProjectConfigs()
    data = helper_functions.load_object('../' + pc.data_file).data
    data.set_train()
    source_data = data.get_transfer_subset(pc.source_labels)
    source_data.set_target()
    target_data= data.get_transfer_subset(pc.target_labels)
    target_data.set_target()
    source_learner.train_and_test(source_data)
    target_learner.train_and_test(target_data)
    source_learner.sigma = 10
    target_learner.sigma = 10
    x = array_functions.vec_to_2d(np.linspace(data.x.min(), data.x.max(), 100))
    test_data = data_lib.Data()
    test_data.x = x
    test_data.is_regression = True
    y_s = source_learner.predict(test_data).fu
    y_t = target_learner.predict(test_data).fu

    #array_functions.plot_line(x,y_t-y_s,pc.data_set,y_axes=np.asarray([-5,5]))
    y = y_t-y_s
    #y = y - y.mean()
    array_functions.plot_line(x,y, title=None ,fig=fig,show=show)
    if show_histogram:
        array_functions.plot_histogram(data.x,20)
    x=1
    #viz_features(data.x,data.y,data.data_set_ids,learner=learner)

def all_same_sign(a):
    return np.abs((np.sign(a)).sum()) == a.size

def viz_all():
    vis_configs = configs_lib.VisualizationConfigs()
    data_sets = configs_lib.data_sets_for_exps
    max_rows = 3
    n = len(data_sets)
    #fig = plt.figure(len(data_sets))

    if getattr(vis_configs, 'figsize', None):
        fig = plt.figure(figsize=vis_configs.figsize)
    else:
        fig = plt.figure()
    num_rows = min(n, max_rows)
    num_cols = math.ceil(float(n) / num_rows)
    for i, data_set_id in enumerate(data_sets):
        subplot_idx = i + 1
        plt.subplot(num_rows,num_cols,subplot_idx)
        pc = configs_lib.ProjectConfigs(data_set_id)
        vis_configs = configs_lib.VisualizationConfigs(data_set_id)
        viz(pc,fig=fig,show=False)
        plt.title(vis_configs.title)
        a = plt.gca()
        ylim = np.asarray(a.get_ylim())
        if all_same_sign(ylim):
            I = (np.absolute(ylim)).argmin()
            ylim[I] = 0
            a.set_ylim(ylim)
        pass
    if getattr(vis_configs,'borders',None):
        left,right,top,bottom = vis_configs.borders
        fig.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
    plt.show()
    x=1


if __name__ == "__main__":
    viz_all()