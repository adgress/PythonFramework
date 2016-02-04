import methods.local_transfer_methods

__author__ = 'Aubrey'

from configs import base_configs as bc
import numpy as np
from data_sets import create_data_set
from loss_functions import loss_function
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions

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
    'use_pool',
    'pool_size',
    'use_1d_data',
    'data_set'
]
data_set_to_use = None
#data_set_to_use = bc.DATA_SYNTHETIC_CLASSIFICATION
#data_set_to_use = bc.DATA_SYNTHETIC_CLASSIFICATION_LOCAL
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_TRANSFER
#data_set_to_use = bc.DATA_NG

data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_BIKE_SHARING
#data_set_to_use = bc.DATA_WINE

#data_set_to_use = bc.DATA_SYNTHETIC_SLANT
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER
#data_set_to_use = bc.DATA_SYNTHETIC_DELTA_LINEAR
#data_set_to_use = bc.DATA_SYNTHETIC_CROSS

run_batch_exps = True
use_1d_data = True
use_constraints = False

use_fused_lasso = True
no_C3 = True
use_radius = True
include_scale = True
constant_b = False
linear_b = True
use_validation = False

synthetic_data_sets = [
    bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER,
    bc.DATA_SYNTHETIC_DELTA_LINEAR,
    bc.DATA_SYNTHETIC_CROSS,
    bc.DATA_SYNTHETIC_SLANT,
]

real_data_sets_1d = [
    bc.DATA_BOSTON_HOUSING,
    bc.DATA_CONCRETE,
    bc.DATA_BIKE_SHARING,
    bc.DATA_WINE
]

real_data_sets_full = [
    bc.DATA_BOSTON_HOUSING,
    bc.DATA_CONCRETE,
    bc.DATA_WINE,
]

all_1d = synthetic_data_sets + real_data_sets_1d

data_sets_for_exps = [data_set_to_use]
if run_batch_exps:
    if use_1d_data:
        data_sets_for_exps = all_1d
    else:
        data_sets_for_exps = real_data_sets_full
#data_sets_for_exps = real_data_sets_full

synthetic_dim = 1
if helper_functions.is_laptop():
    use_pool = False
    pool_size = 4
else:
    use_pool = True
    pool_size = 24
max_features = create_data_set.max_features

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None):
        super(ProjectConfigs, self).__init__()
        self.target_labels = np.empty(0)
        self.source_labels = np.empty(0)
        self.project_dir = 'base'
        self.num_labels = range(40,201,40)
        self.oracle_labels = np.empty(0)
        self.use_pool = use_pool
        self.pool_size = pool_size
        self.num_splits = 10
        self.num_splits = 30
        if data_set is None:
            data_set = data_set_to_use
        self.set_data_set(data_set, use_1d_data)

    def set_data_set(self, data_set, use_1d):
        self.data_set = data_set
        self.use_1d_data = use_1d
        if data_set == bc.DATA_NG:
            self.set_ng_transfer()
            #self.num_labels = range(20,61,20) + [120, 180]
            #self.num_labels = range(20,61,20)
            self.num_labels = [5,10,20]
            #self.num_labels = [20]
        elif data_set == bc.DATA_BOSTON_HOUSING:
            self.set_boston_housing_transfer()
            #self.num_labels = [5,10,20,40]
            self.num_labels = [5,10,20]
            #self.set_boston_housing()
            #self.num_labels = range(20,61,20)
        elif data_set == bc.DATA_SYNTHETIC_STEP_TRANSFER:
            self.set_synthetic_step_transfer()
            #self.num_labels = range(10,31,10)
            self.num_labels = [20]
        elif data_set == bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER:
            self.set_synthetic_step_linear_transfer()
            #self.num_labels = [30]
            self.num_labels = range(10,31,10)
        elif data_set == bc.DATA_SYNTHETIC_CLASSIFICATION:
            self.set_synthetic_classification()
            self.num_labels = [4,8,16]
            #self.num_labels = [10]
            #self.num_labels = range(10,31,10)
            #self.num_labels = range(10,71,10)
        elif data_set == bc.DATA_SYNTHETIC_CLASSIFICATION_LOCAL:
            self.set_synthetic_classification_local()
            self.num_labels = [4,8,16]
        elif data_set == bc.DATA_CONCRETE:
            self.set_concrete_transfer()
            #self.num_labels = [5,10,20,40]
            self.num_labels = [5,10,20]
        elif data_set == bc.DATA_BIKE_SHARING:
            self.set_bike_sharing()
            self.num_labels = [5,10,20,40]
        elif data_set == bc.DATA_WINE:
            self.set_wine()
            self.num_labels = [5,10,20]
        elif data_set == bc.DATA_SYNTHETIC_DELTA_LINEAR:
            self.set_synthetic_regression('synthetic_delta_linear_transfer')
            self.num_labels = np.asarray([10,20,30])
        elif data_set == bc.DATA_SYNTHETIC_CROSS:
            self.set_synthetic_regression('synthetic_cross_transfer')
            self.num_labels = np.asarray([10,20,30])
        elif data_set == bc.DATA_SYNTHETIC_SLANT:
            self.set_synthetic_regression('synthetic_slant')
            self.num_labels = np.asarray([10,20,30])
        else:
            assert False
        assert self.source_labels.size > 0
        assert self.target_labels.size > 0
        self.labels_to_not_sample = self.source_labels.ravel()
        a = self.source_labels.ravel()
        self.labels_to_keep = np.concatenate((self.target_labels,a))

    def set_wine(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()

        if self.use_1d_data:
            self.data_dir = 'data_sets/wine-small-feat=1'
            self.data_name = 'wine-small-feat=1'
            self.results_dir = 'wine-small-feat=1'
        else:
            self.data_dir = 'data_sets/wine-small-11'
            self.data_name = 'wine-small-11'
            self.results_dir = 'wine-small-11'

        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([0])
        self.source_labels = np.asarray([1])

    def set_boston_housing_transfer(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()

        if self.use_1d_data:
            self.data_dir = 'data_sets/boston_housing'
            self.data_name = 'boston_housing'
            self.results_dir = 'boston_housing'
        else:
            self.data_dir = 'data_sets/boston_housing-13'
            self.data_name = 'boston_housing-13'
            self.results_dir = 'boston_housing-13'
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([0])
        self.source_labels = np.asarray([1])

    def set_bike_sharing(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        assert self.use_1d_data == True
        self.data_dir = 'data_sets/bike_sharing-feat=1'
        self.data_name = 'bike_sharing-feat=1'
        self.results_dir = 'bike_sharing-feat=1'
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([1])
        self.source_labels = np.asarray([0])


    def set_concrete_transfer(self):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()

        if self.use_1d_data:
            self.data_dir = 'data_sets/concrete-feat=0'
            self.data_name = 'concrete-feat=0'
            self.results_dir = 'concrete-feat=0'
        else:
            self.data_dir = 'data_sets/concrete-7'
            self.data_name = 'concrete-7'
            self.results_dir = 'concrete-7'
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([1])
        self.source_labels = np.asarray([3])

    def set_synthetic_classification_local(self):
        self.loss_function = loss_function.ZeroOneError()
        self.data_dir = 'data_sets/synthetic_classification_local'
        self.data_name = 'synthetic_classification_local'
        self.data_set_file_name = 'split_data.pkl'
        self.results_dir = 'synthetic_classification_local'
        self.target_labels = np.asarray([1,2])
        #self.target_labels = array_functions.vec_to_2d(self.target_labels).T
        self.source_labels = np.asarray([3,4])
        self.source_labels = array_functions.vec_to_2d(self.source_labels).T
        self.cv_loss_function = loss_function.LogLoss()
        #self.cv_loss_function = loss_function.ZeroOneError()

    def set_synthetic_classification(self):
        self.loss_function = loss_function.ZeroOneError()
        self.data_dir = 'data_sets/synthetic_classification'
        self.data_name = 'synthetic_classification'
        self.data_set_file_name = 'split_data.pkl'
        self.results_dir = 'synthetic_classification'
        self.target_labels = np.asarray([1,2])
        #self.target_labels = array_functions.vec_to_2d(self.target_labels).T
        self.source_labels = np.asarray([3,4])
        self.source_labels = array_functions.vec_to_2d(self.source_labels).T
        self.cv_loss_function = loss_function.LogLoss()
        #self.cv_loss_function = loss_function.ZeroOneError()

    def set_synthetic_regression(self, name):
        self.loss_function = loss_function.MeanSquaredError()
        self.target_labels = np.zeros(1)
        self.source_labels = np.ones(1)
        self.loss_function = loss_function.MeanSquaredError()
        self.data_dir = 'data_sets/' + name
        self.data_name = name
        self.results_dir = name
        self.data_set_file_name = 'split_data.pkl'


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
        self.cv_loss_function = loss_function.LogLoss()

    def set_ng_transfer(self):
        self.loss_function = loss_function.ZeroOneError()
        self.set_ng()
        '''
        self.target_labels = np.asarray([1,2])
        S1 = np.asarray([7,8])
        S2 = np.asarray([12,13])
        self.source_labels = np.vstack((S1,S2))
        '''

        self.target_labels = CR[0]
        #self.source_labels = CR[1]
        self.source_labels = np.vstack((CR[1], ST[1]))
        self.oracle_labels = CR[1]
        #self.source_labels = ST[1]

        #self.oracle_labels = np.empty(0)
        #self.cv_loss_function = loss_function.ZeroOneError()
        self.cv_loss_function = loss_function.LogLoss()

def nonpositive_constraint(g):
    return g <= 0

def nonnegative_constraint(g):
    return g >= 0

class MainConfigs(bc.MainConfigs):
    def __init__(self, pc):
        super(MainConfigs, self).__init__()
        #pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        from methods import transfer_methods
        from methods import method
        from methods import scipy_opt_methods
        method_configs = MethodConfigs(pc)
        method_configs.metric = 'euclidean'
        method_configs.no_reg = False
        method_configs.use_g_learner = True
        method_configs.use_validation = False
        method_configs.use_reg2 = True

        method_configs.use_fused_lasso = use_fused_lasso
        method_configs.no_C3 = no_C3
        method_configs.use_radius = use_radius
        method_configs.include_scale = include_scale
        method_configs.constant_b = constant_b
        method_configs.linear_b = linear_b

        assert not (constant_b and linear_b)

        if self.data_set == bc.DATA_NG:
            method_configs.metric = 'cosine'
            method_configs.use_fused_lasso = False

        method_configs.constraints = []
        if use_constraints:
            if self.data_set == bc.DATA_CONCRETE:
                method_configs.constraints.append(nonpositive_constraint)
            elif self.data_set == bc.DATA_BIKE_SHARING:
                method_configs.constraints.append(nonnegative_constraint)


        fuse_log_reg = transfer_methods.FuseTransfer(method_configs)
        fuse_nw = transfer_methods.FuseTransfer(method_configs)
        fuse_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        target_nw = transfer_methods.TargetTranfer(method_configs)
        target_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        target_ridge = transfer_methods.TargetTranfer(method_configs)
        target_ridge.base_learner = method.SKLRidgeRegression(method_configs)
        nw = method.NadarayaWatsonMethod(method_configs)
        log_reg = method.SKLLogisticRegression(method_configs)
        target_knn = transfer_methods.TargetTranfer(method_configs)
        target_knn.base_learner = method.SKLKNN(method_configs)
        local_transfer = methods.local_transfer_methods.LocalTransfer(method_configs)
        scipy_ridge_reg = scipy_opt_methods.ScipyOptRidgeRegression(method_configs)
        model_transfer = methods.transfer_methods.ModelSelectionTransfer(method_configs)
        hyp_transfer = methods.local_transfer_methods.HypothesisTransfer(method_configs)
        iwl_transfer = methods.local_transfer_methods.IWTLTransfer(method_configs)
        sms_transfer = methods.local_transfer_methods.SMSTransfer(method_configs)
        dt_local_transfer = methods.local_transfer_methods.LocalTransferDelta(method_configs)
        dt_sms = methods.local_transfer_methods.LocalTransferDeltaSMS(method_configs)

        #self.learner = target_nw
        #self.learner = hyp_transfer
        #self.learner = local_transfer
        #self.learner = iwl_transfer
        #self.learner = sms_transfer
        self.learner = dt_local_transfer
        #self.learner = dt_sms
        self.learner.configs.use_validation = use_validation


class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)
        self.target_labels = pc.target_labels
        self.source_labels = pc.source_labels

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self):
        super(VisualizationConfigs, self).__init__()
        pc = create_project_configs()
        self.copy_fields(pc,pc_fields_to_copy)
        '''
        self.files = [
            'TargetTransfer+NW.pkl',
            'HypothesisTransfer.pkl',
            'LocalTransferDelta_C3=0_radius.pkl',
            'LocalTransferDelta_C3=0.pkl',
            'LocalTransferDelta_radius.pkl',
            'LocalTransferDelta.pkl',
            'LocalTransferDeltaSMS.pkl',
        ]
        '''

        self.files = {
            'TargetTransfer+NW.pkl': 'Target Only',
            'LocalTransferDelta_C3=0.pkl': 'Our Method, alpha=0',
            'LocalTransferDelta.pkl': 'Our Method',
            'LocalTransferDelta_C3=0_cons.pkl': None,
            'LocalTransferDelta_cons.pkl': None,
            'LocalTransferDelta_C3=0_radius.pkl': 'Our Method, ball graph, alpha=0',
            'LocalTransferDelta_radius.pkl': 'Our Method, ball graph'
        }
        self.files = {}
        self.files['TargetTransfer+NW.pkl'] = 'Target Only'
        #self.files['LocalTransferDelta_radius.pkl'] = 'Our Method, ball graph'
        #self.files['LocalTransferDelta_radius_l2.pkl'] = 'Our Method, ball graph, l2 loss'
        #self.files['LocalTransferDelta_l2.pkl'] = 'Our Method, l2 loss'
        #self.files['LocalTransferDelta_C3=0_radius.pkl'] = 'Our Method, ball graph, alpha=0'
        #self.files['LocalTransferDeltaSMS.pkl'] = 'SMS no scale'
        #self.files['LocalTransferDeltaSMS_scale.pkl'] = 'SMS with scale'
        self.files['LocalTransferDelta_radius_l2_constant-b.pkl'] = 'Our Method, constant b'
        self.files['LocalTransferDelta_C3=0_radius_l2_linear-b.pkl'] = 'Our Method, linear b, alpha=0'
        self.files['LocalTransferDelta_C3=0_radius_l2_constant-b.pkl'] = 'Our Method, constant b, alpha=0'
        #self.files['LocalTransferDelta_radius_cons_l2.pkl'] = 'Our Method, ball graph, l2 loss, constrained'
        self.files['LocalTransferDelta_radius_l2_use-val.pkl'] = 'Our method, ball graph, l2 loss, used validation'
        #self.files['LocalTransferDelta_C3=0_radius_l2_lap-reg.pkl'] = 'Ours, ball, l2 loss, lap-reg'
        #self.files['LocalTransferDelta_C3=0_radius_l2_use-val_lap-reg.pkl'] = 'Ours, ball, alpha=0, lap-reg, use validation'
        self.data_set_to_use = data_set_to_use
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')

        if self.use_1d_data:
            self.title += ' 1D'
        self.x_axis_string = 'Number of labeled target instances'


class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        self.config_list = []
        for i in data_sets_for_exps:
            pc = ProjectConfigs(i)
            self.config_list += [MainConfigs(pc)]
        assert len(self.config_list) > 0