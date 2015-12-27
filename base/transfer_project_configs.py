import methods.local_transfer_methods

__author__ = 'Aubrey'

from configs import base_configs as bc
import numpy as np
from data_sets import create_data_set
from loss_functions import loss_function
CR = []
for i in range(0,4):
    a = [create_data_set.ng_c[i],create_data_set.ng_r[i]]
    CR.append(np.asarray(a))

ST = []
for i in range(0,4):
    a = [create_data_set.ng_s[i],create_data_set.ng_t[i]]
    ST.append(np.asarray(a))


def create_project_configs():
    return ProjectConfigs()

pc_fields_to_copy = bc.pc_fields_to_copy + [
    'target_labels',
    'source_labels',
    'oracle_labels',
    'use_pool'
]

#data_set_to_use = bc.DATA_BOSTONG_HOUSING
data_set_to_use = bc.DATA_NG
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_TRANSFER
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER

synthetic_dim = 1
use_pool = False
max_features = 100

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self):
        super(ProjectConfigs, self).__init__()
        self.target_labels = np.empty(0)
        self.source_labels = np.empty(0)
        self.project_dir = 'base'
        self.num_labels = range(40,201,40)
        self.num_splits = 10
        self.oracle_labels = []
        self.use_pool = use_pool
        #self.num_splits = 30


        if data_set_to_use == bc.DATA_NG:
            self.set_ng_transfer()
            #self.num_labels = range(20,61,20) + [120, 180]
            #self.num_labels = range(20,61,20)
            self.num_labels = [5,10,20]
        elif data_set_to_use == bc.DATA_BOSTONG_HOUSING:
            self.set_boston_housing()
            self.num_labels = range(20,61,20)
        elif data_set_to_use == bc.DATA_SYNTHETIC_STEP_TRANSFER:
            self.set_synthetic_step_transfer()
            self.num_labels = range(10,31,10)
            #self.num_labels = [50]
        elif data_set_to_use == bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER:
            self.set_synthetic_step_linear_transfer()
            #self.num_labels = [30]
            self.num_labels = range(10,31,10)
        else:
            assert False

        self.labels_to_not_sample = self.source_labels.ravel()
        self.labels_to_keep = np.concatenate((self.target_labels,self.source_labels.ravel()))


    def set_synthetic_step_linear_transfer(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/synthetic_step_linear_transfer'
        self.data_name = 'synthetic_step_linear_transfer'
        self.data_set_file_name = 'split_data.pkl'
        self.results_dir = 'synthetic_step_linear_transfer'
        self.target_labels = np.asarray([0])
        self.source_labels = np.asarray([1])

    def set_synthetic_step_transfer(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/synthetic_step_transfer'
        self.data_name = 'synthetic_step_transfer'
        self.results_dir = 'synthetic_step_transfer'
        if synthetic_dim > 1:
            s = '_' + str(synthetic_dim)
            self.data_dir += s
            self.data_name += s
            self.results_dir += s
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([0])
        self.source_labels = np.asarray([1])

    def set_ng(self):
        self.data_dir = 'data_sets/20ng-%d' % max_features
        self.data_name = 'ng_data-%d' % max_features
        self.data_set_file_name = 'split_data.pkl'
        self.results_dir = '20ng-%d' % max_features

    def set_ng_transfer(self):
        self.loss_function = loss_function.ZeroOneError()
        self.set_ng()
        self.target_labels = CR[0]
        #self.source_labels = CR[1]
        self.source_labels = np.vstack((CR[1], ST[1]))
        self.oracle_labels = CR[1]
        #self.source_labels = ST[1]
        #self.oracle_labels = np.empty(0)
        #self.cv_loss_function = loss_function.ZeroOneError()
        self.cv_loss_function = loss_function.LogLoss()

class MainConfigs(bc.MainConfigs):
    def __init__(self):
        super(MainConfigs, self).__init__()
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import transfer_methods
        from methods import method
        from methods import scipy_opt_methods
        method_configs = MethodConfigs()
        method_configs.metric = 'euclidean'
        method_configs.use_fused_lasso = True
        method_configs.no_reg = False
        method_configs.use_g_learner = True
        if data_set_to_use == bc.DATA_NG:
            method_configs.metric = 'cosine'
            method_configs.use_fused_lasso = False

        fuse_log_reg = transfer_methods.FuseTransfer(method_configs)
        fuse_nw = transfer_methods.FuseTransfer(method_configs)
        fuse_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        target_nw = transfer_methods.TargetTranfer(method_configs)
        target_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        nw = method.NadarayaWatsonMethod(method_configs)
        log_reg = method.SKLLogisticRegression(MethodConfigs())
        target_knn = transfer_methods.TargetTranfer(method_configs)
        target_knn.base_learner = method.SKLKNN(method_configs)
        local_transfer = methods.local_transfer_methods.LocalTransfer(method_configs)
        scipy_ridge_reg = scipy_opt_methods.ScipyOptRidgeRegression(method_configs)
        model_transfer = methods.transfer_methods.ModelSelectionTransfer(method_configs)
        hyp_transfer = methods.local_transfer_methods.HypothesisTransfer(method_configs)

        #self.learner = hyp_transfer
        #self.learner = model_transfer
        #self.learner = scipy_ridge_reg
        self.learner = local_transfer
        #self.learner = fuse_nw
        #self.learner = target_nw


class MethodConfigs(bc.MethodConfigs):
    def __init__(self):
        super(MethodConfigs, self).__init__()
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        self.target_labels = pc.target_labels
        self.source_labels = pc.source_labels

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self):
        super(VisualizationConfigs, self).__init__()
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        self.files = [
            'TargetTransfer+SKL-LogReg.pkl',
            'FuseTransfer+SKL-LogReg.pkl',
            'ModelSelTransfer.pkl',
            'HypothesisTransfer.pkl',
            'SKL-LogReg.pkl',
            'NW.pkl',
            'SKL-RidgeReg.pkl',
            'TargetTransfer+SKL-KNN.pkl',
            'LocalTransfer.pkl',
            'FuseTransfer+NW.pkl',
            'LocalTransfer-Parametric.pkl',
            'LocalTransfer-Parametric-max_value=0.5.pkl',
            'LocalTransfer-Oracle.pkl',
            'HypothesisTransfer-Oracle.pkl',
            'FuseTransfer+NW-Oracle.pkl',
            'LocalTransfer-Oracle-Parametric-max_value=0.5.pkl',
            'LocalTransfer-l1.pkl',
            'LocalTransfer-l2.pkl',
            'LocalTransfer-no_reg.pkl',
            'LocalTransfer-NonParaHypTrans.pkl',
            'TargetTransfer+NW.pkl',
            'FuseTransfer+NW-tws=0.5.pkl',
            'FuseTransfer+NW-tws=0.9.pkl',
        ]

class BatchConfigs(bc.BatchConfigs):
    def __init__(self):
        super(BatchConfigs, self).__init__()
        self.config_list = [MainConfigs()]