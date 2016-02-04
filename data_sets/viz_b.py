from data import data as data_class
from utility import helper_functions
from utility import array_functions
import scipy
import numpy as np
from data import data as data_lib
from base import transfer_project_configs

def run_main():
    import create_data_set
    from methods import method
    source_learner = method.NadarayaWatsonMethod()
    target_learner = method.NadarayaWatsonMethod()
    pc = transfer_project_configs.ProjectConfigs()
    data = helper_functions.load_object('../' + pc.data_file).data
    data.set_train()
    source_data = data.get_transfer_subset(pc.source_labels)
    source_data.set_target()
    target_data= data.get_transfer_subset(pc.target_labels)
    target_data.set_target()
    source_learner.train_and_test(source_data)
    target_learner.train_and_test(target_data)
    x = array_functions.vec_to_2d(np.linspace(data.x.min(), data.x.max(), 100))
    test_data = data_lib.Data()
    test_data.x = x
    test_data.is_regression = True
    y_s = source_learner.predict(test_data).fu
    y_t = target_learner.predict(test_data).fu

    #array_functions.plot_line(x,y_t-y_s,pc.data_set,y_axes=np.asarray([-5,5]))
    y = y_t-y_s
    y = y - y.mean()
    array_functions.plot_line(x,y,pc.data_set)
    array_functions.plot_histogram(data.x,20)
    x=1
    #viz_features(data.x,data.y,data.data_set_ids,learner=learner)

if __name__ == "__main__":
    run_main()