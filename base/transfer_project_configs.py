import methods.local_transfer_methods

__author__ = 'Aubrey'

from configs import base_configs as bc
import numpy as np
from data_sets import create_data_set
from loss_functions import loss_function
from data_sets import create_data_set
from utility import array_functions
from utility import helper_functions
from collections import OrderedDict
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

#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_WINE
#data_set_to_use = bc.DATA_BIKE_SHARING
#data_set_to_use = bc.DATA_PAIR_82_83

data_set_to_use = bc.DATA_SYNTHETIC_CURVE
#data_set_to_use = bc.DATA_SYNTHETIC_SLANT
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER
#data_set_to_use = bc.DATA_SYNTHETIC_DELTA_LINEAR
#data_set_to_use = bc.DATA_SYNTHETIC_CROSS
#data_set_to_use = bc.DATA_SYNTHETIC_SLANT_MULTITASK

PLOT_PARAMETRIC = 1
PLOT_VALIDATION = 2
PLOT_CONSTRAINED = 3
PLOT_SMS = 4
PLOT_TABLE_COMPETING_METHODS = 5
PLOT_TABLE_VAL = 6
PLOT_ALPHA = 7
PLOT_TABLE_OUR_METHODS = 8
PLOT_TABLE_CONSTRAINED = 9
PLOT_COMPETING_METHODS = 10


plot_idx = PLOT_COMPETING_METHODS
#plot_idx = PLOT_TABLE_COMPETING_METHODS
#plot_idx = PLOT_TABLE_CONSTRAINED

max_rows = 1
fontsize = 10

vis_table = plot_idx in {PLOT_TABLE_COMPETING_METHODS, PLOT_TABLE_VAL, PLOT_TABLE_OUR_METHODS, PLOT_TABLE_CONSTRAINED}
size_to_vis = 10
sizes_to_use = [5, 10, 20, 30]
data_set_sizes_to_use = {
    bc.DATA_TAXI: [10, 20, 40, 80, 160],
    bc.DATA_BIKE_SHARING: [5, 10, 20, 40],
}

run_experiments = True
#show_legend_on_all = False
show_legend_on_all = False

crash_on_missing_files = False


run_batch_exps = True
vis_batch = True
use_1d_data = False
use_all_data = True
use_sms_plot_data_sets = plot_idx == PLOT_SMS

use_validation = True

use_constraints = False
use_fused_lasso = False
no_C3 = False
use_radius = False
include_scale = True
constant_b = False
linear_b = True
clip_b = True
separate_target_domains = False
multitask = False

use_delta_new = True

synthetic_data_sets = [
    bc.DATA_SYNTHETIC_CURVE,
    #bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER,
    bc.DATA_SYNTHETIC_DELTA_LINEAR,
    #bc.DATA_SYNTHETIC_CROSS,
    #bc.DATA_SYNTHETIC_SLANT,
]


#bc.DATA_SYNTHETIC_SLANT_MULTITASK

real_data_sets_1d = [
    bc.DATA_BOSTON_HOUSING,
    bc.DATA_CONCRETE,
    bc.DATA_WINE,
    bc.DATA_BIKE_SHARING,
]

real_data_sets_full = [
    bc.DATA_BOSTON_HOUSING,
    bc.DATA_CONCRETE,
    bc.DATA_WINE,
    #bc.DATA_KC_HOUSING,
    bc.DATA_TAXI,
]

synthetic_names = [
    'Curve',
    'Delta'
]

real_1d_names = [
    'BH 1D',
    'Concrete 1D',
    'Wine 1D',
    'Bike'
]

real_full_names = [
    'BH',
    'Concrete',
    'Wine',
    'Taxi'
]

all_1d = synthetic_data_sets + real_data_sets_1d
all_1d_names = synthetic_names + real_1d_names
names_for_table = None

data_sets_for_exps = [data_set_to_use]
if run_batch_exps:
    if use_sms_plot_data_sets:
        data_sets_for_exps = [
            bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER,
            bc.DATA_SYNTHETIC_DELTA_LINEAR,
            bc.DATA_SYNTHETIC_CROSS,
        ]
    elif use_1d_data:
        data_sets_for_exps = all_1d
        #data_sets_for_exps = synthetic_data_sets
    else:
        data_sets_for_exps = real_data_sets_full
if plot_idx in {PLOT_TABLE_COMPETING_METHODS, PLOT_TABLE_OUR_METHODS, PLOT_TABLE_CONSTRAINED}:
    names_for_table = all_1d_names + real_full_names
#data_sets_for_exps = real_data_sets_full

synthetic_dim = 1
if helper_functions.is_laptop():
    use_pool = False
    pool_size = 4
else:
    use_pool = False
    pool_size = 24
max_features = create_data_set.max_features
arguments = None
def apply_arguments(configs):
    if arguments.num_labels is not None:
        configs.overwrite_num_labels = arguments.num_labels
    if arguments.split_idx is not None:
        configs.split_idx = arguments.split_idx

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None, use_arguments=True, **kwargs):
        super(ProjectConfigs, self).__init__()
        self.target_labels = np.empty(0)
        self.source_labels = np.empty(0)
        self.project_dir = 'base'
        self.num_labels = range(40,201,40)
        self.oracle_labels = np.empty(0)
        self.use_pool = use_pool
        self.pool_size = pool_size
        if data_set is None:
            data_set = data_set_to_use
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        if 'use_1d_data' not in kwargs:
            self.use_1d_data = use_1d_data
        self.set_data_set(data_set, self.use_1d_data)
        if use_arguments and arguments is not None:
            apply_arguments(self)

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
        elif data_set == bc.DATA_SYNTHETIC_CURVE:
            self.set_synthetic_regression('synthetic_curve')
            self.num_labels = np.asarray([10,20,30])
        elif data_set == bc.DATA_KC_HOUSING:
            self.set_data_set_defaults('kc-housing-spatial-floors', source_labels=[0], target_labels=[1], is_regression=True)
            #self.num_labels = np.asarray([20, 40, 80, 160])
            self.num_labels = np.asarray([20, 40, 80, 160])
        elif data_set == bc.DATA_TAXI:
            #self.set_data_set_defaults('taxi2-20', source_labels=[1], target_labels=[0], is_regression=True)
            #self.set_data_set_defaults('taxi2-50', source_labels=[1], target_labels=[0], is_regression=True)
            #self.set_data_set_defaults('taxi2', source_labels=[0], target_labels=[1], is_regression=True)
            self.set_data_set_defaults('taxi3', source_labels=[1], target_labels=[0], is_regression=True)
            #self.num_labels = np.asarray([5, 10, 20, 40, 100, 200, 400, 800])
            self.num_labels = np.asarray([10, 20, 40, 80])
            #self.num_labels = np.asarray([50, 100, 200, 400])
        elif data_set == bc.DATA_SYNTHETIC_SLANT_MULTITASK:
            self.set_synthetic_regression('synthetic_slant_multitask')
            self.target_labels = np.asarray([0,1])
            self.source_labels = np.asarray([2])
            self.num_labels = np.asarray([5, 10, 20])
        elif data_set == bc.DATA_PAIR_82_83:
            self.set_synthetic_regression('pair_data_82_83')
            self.num_labels = np.asarray([10,20,30])
        elif data_set == bc.DATA_PAIR_13_14:
            self.set_synthetic_regression('pair_data_13_14')
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
            self.data_dir = 'data_sets/boston_housing(transfer)'
            self.data_name = 'boston_housing'
            self.results_dir = 'boston_housing'
        else:
            self.data_dir = 'data_sets/boston_housing-13(transfer)'
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

    def set_regression(self, name):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        self.target_labels = np.zeros(1)
        self.source_labels = np.ones(1)
        self.data_dir = 'data_sets/' + name
        self.data_name = name
        self.results_dir = name
        self.data_set_file_name = 'split_data.pkl'

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

def nonpositive_constraint(g,):
    return g <= 0

def nonnegative_constraint(g,):
    return g >= 0

def bound_4(g,):
    return g <= 4

def nonpositive_constraint_linear(g,b,X):
    return X * g + b <= 0

def nonnegative_constraint_linear(g,b,X):
    return X * g + b >= 0

def bound_4_linear(g,b,X):
    return X * g + b <= 4

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
        method_configs.use_validation = use_validation
        method_configs.use_reg2 = True
        method_configs.joint_cv = True

        method_configs.use_fused_lasso = use_fused_lasso
        method_configs.no_C3 = no_C3
        method_configs.use_radius = use_radius
        method_configs.include_scale = include_scale
        method_configs.constant_b = constant_b
        method_configs.linear_b = linear_b
        method_configs.clip_b = clip_b
        method_configs.separate_target_domains = separate_target_domains
        method_configs.multitask = multitask
        if self.data_set != bc.DATA_SYNTHETIC_SLANT_MULTITASK:
            method_configs.separate_target_domains = False
            method_configs.multitask = False
        assert not (constant_b and linear_b)

        if self.data_set == bc.DATA_NG:
            method_configs.metric = 'cosine'
            method_configs.use_fused_lasso = False

        method_configs.constraints = []
        if use_constraints:
            if linear_b:
                if self.data_set == bc.DATA_CONCRETE:
                    method_configs.constraints.append(nonpositive_constraint_linear)
                elif self.data_set == bc.DATA_BIKE_SHARING:
                    method_configs.constraints.append(nonnegative_constraint_linear)
                elif self.data_set == bc.DATA_BOSTON_HOUSING:
                    method_configs.constraints.append(nonnegative_constraint_linear)
                elif self.data_set == bc.DATA_WINE:
                    method_configs.constraints.append(nonnegative_constraint_linear)
                elif self.data_set == bc.DATA_SYNTHETIC_CURVE:
                    method_configs.constraints.append(nonpositive_constraint_linear)
                elif self.data_set == bc.DATA_SYNTHETIC_SLANT:
                    method_configs.constraints.append(nonpositive_constraint_linear)
                elif self.data_set == bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER:
                    method_configs.constraints.append(nonpositive_constraint_linear)
                elif self.data_set == bc.DATA_SYNTHETIC_DELTA_LINEAR:
                    method_configs.constraints.append(nonpositive_constraint_linear)
                elif self.data_set == bc.DATA_SYNTHETIC_CROSS:
                    method_configs.constraints.append(bound_4_linear)
                else:
                    assert False
            else:
                if self.data_set == bc.DATA_CONCRETE:
                    method_configs.constraints.append(nonpositive_constraint)
                elif self.data_set == bc.DATA_BIKE_SHARING:
                    method_configs.constraints.append(nonnegative_constraint)
                elif self.data_set == bc.DATA_BOSTON_HOUSING:
                    method_configs.constraints.append(nonnegative_constraint)
                elif self.data_set == bc.DATA_WINE:
                    method_configs.constraints.append(nonnegative_constraint)
                elif self.data_set == bc.DATA_SYNTHETIC_CURVE:
                    method_configs.constraints.append(nonpositive_constraint)
                elif self.data_set == bc.DATA_SYNTHETIC_SLANT:
                    method_configs.constraints.append(nonpositive_constraint)
                elif self.data_set == bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER:
                    method_configs.constraints.append(nonpositive_constraint)
                elif self.data_set == bc.DATA_SYNTHETIC_DELTA_LINEAR:
                    method_configs.constraints.append(nonpositive_constraint)
                elif self.data_set == bc.DATA_SYNTHETIC_CROSS:
                    method_configs.constraints.append(bound_4)
                else:
                    assert False

        method_configs.use_validation = use_validation
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
        scipy_ridge_reg = scipy_opt_methods.ScipyOptRidgeRegression(method_configs)
        model_transfer = methods.transfer_methods.ModelSelectionTransfer(method_configs)
        hyp_transfer = methods.local_transfer_methods.HypothesisTransfer(method_configs)
        iwl_transfer = methods.local_transfer_methods.IWTLTransfer(method_configs)
        sms_transfer = methods.local_transfer_methods.SMSTransfer(method_configs)
        local_transfer_delta = methods.local_transfer_methods.LocalTransferDelta(method_configs)
        dt_sms = methods.local_transfer_methods.LocalTransferDeltaSMS(method_configs)
        cov_shift = transfer_methods.ReweightedTransfer(method_configs)
        offset_transfer = methods.local_transfer_methods.OffsetTransfer(method_configs)
        stacked_transfer = methods.transfer_methods.StackingTransfer(method_configs)

        gaussian_process = methods.method.SKLGaussianProcess(method_configs)

        from methods import semisupervised
        from methods import preprocessing
        ssl_regression = semisupervised.SemisupervisedRegressionMethod(method_configs)
        ssl_regression.preprocessor = preprocessing.TargetOnlyPreprocessor()

        if use_delta_new:
            self.learner = methods.local_transfer_methods.LocalTransferDeltaNew(method_configs)
        else:
            #self.learner = target_nw
            #self.learner = offset_transfer
            self.learner = stacked_transfer
            #self.learner = ssl_regression
            #self.learner = cov_shift
            #self.learner = dt_sms



            #self.learner = hyp_transfer
            #self.learner = local_transfer
            #self.learner = iwl_transfer


            #self.learner = local_transfer_delta
            #self.learner = sms_transfer
            #self.learner = gaussian_process


class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)
        self.target_labels = pc.target_labels
        self.source_labels = pc.source_labels

class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__()
        pc = ProjectConfigs(data_set, **kwargs)
        self.copy_fields(pc,pc_fields_to_copy)
        self.max_rows = 2
        self.vis_table = vis_table
        self.size_to_vis = size_to_vis
        self.sizes_to_use = sizes_to_use
        if pc.data_set in data_set_sizes_to_use:
            self.sizes_to_use = data_set_sizes_to_use[pc.data_set]
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        self.fontsize = fontsize
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

        self.files = OrderedDict()

        if plot_idx == PLOT_PARAMETRIC:
            self.files = OrderedDict()
            self.files['TargetTransfer+NW.pkl'] = 'Target Only'
            #self.files['StackTransfer+SKL-RidgeReg.pkl'] = 'Stacking'
            #self.files['SLL-NW.pkl'] = 'LLGC'
            #self.files['CovShift.pkl'] = 'Reweighting'
            #self.files['LocalTransferDeltaSMS_scale'] = 'SMS'
#
            #self.files['OffsetTransfer-jointCV.pkl'] = 'Offset Transfer'
            self.files['LocalTransferNew-grad-bounds.pkl'] = 'Ours'
            self.files['LocalTransferNew-grad-bounds-opt_ft.pkl'] = 'Our Method: Optimize Target'
            #self.files['LocalTransferNew-grad-bounds-loo.pkl'] = 'Local Transfer New: LOO'
            self.files['LocalTransferNew-grad-bounds-scaleB.pkl'] = 'Our Method: Scale'
            #self.files['LocalTransferNew-grad-bounds-loo-noTransform.pkl'] = 'Local Transfer New: LOO, no Transform'
            #self.files['LocalTransferNew-grad.pkl'] = 'Local Transfer New: No Bounds'
            self.files['LocalTransferNew-bounds-linearB.pkl'] = 'Our Method: Linear'
        elif plot_idx == PLOT_VALIDATION:
            self.files = OrderedDict()
            self.files['TargetTransfer+NW.pkl'] = 'Target Only'
            self.files['SLL-NW.pkl'] = 'LLGC'
            self.files['LocalTransferDelta_radius_l2_linear-b_clip-b_use-val.pkl'] = 'Ours: Linear, validation'
            self.files['LocalTransferDelta_radius_l2_use-val_lap-reg.pkl'] = 'Ours: Nonparametric, validation'
        elif plot_idx == PLOT_CONSTRAINED:
            self.files = OrderedDict()
            self.files['TargetTransfer+NW.pkl'] = 'Target Only'
            self.files['LocalTransferNew-grad-bounds.pkl'] = 'Our Method'
            #self.files['LocalTransferNew-grad-bounds-boundB'] = 'Ours Method: Bounds'
            #self.files['LocalTransferNew-grad-bounds-boundPerc=80'] = 'Our Method: Bounds 80%'
            #self.files['LocalTransferNew-grad-bounds-boundUpper=80'] = 'Our Method: Bound Upper 80%'
            self.files['LocalTransferNew-grad-bounds-boundPerc=[10, 90].pkl'] = 'Our Method: Bound Constraints'
            #self.files['LocalTransferNew-grad-bounds-boundPerc=[0, 100].pkl'] = 'Our Method: Bound Constraints [0, 100]'
        elif plot_idx == PLOT_SMS:
            self.files = OrderedDict()
            self.files['TargetTransfer+NW.pkl'] = 'Target Only'
            self.files['SLL-NW.pkl'] = 'LLGC'
            self.files['LocalTransferDeltaSMS_scale.pkl'] ='SMS scale'
            self.files['LocalTransferDeltaSMS.pkl'] = 'SMS no scale'
            #self.files['LocalTransferDelta_C3=0_radius_l2_constant-b.pkl'] = 'Constant b, alpha=0'
        elif plot_idx == PLOT_COMPETING_METHODS:
            self.files = OrderedDict()
            self.files['LocalTransferNew-grad-bounds-opt_ft.pkl'] = 'Ours'
            #self.files['TargetTransfer+NW.pkl'] = 'Target Only'
            self.files['StackTransfer+SKL-RidgeReg-jointCV.pkl'] = 'Stacking'
            #self.files['SLL-NW.pkl'] = 'LLGC'
            #self.files['CovShift.pkl'] = 'Reweighting'
            self.files['OffsetTransfer-jointCV.pkl'] = 'Offset'
            #self.files['LocalTransferDeltaSMS_scale'] = 'SMS'
        elif plot_idx == PLOT_TABLE_COMPETING_METHODS:
            self.baseline_idx = 1
            self.data_names_for_table = names_for_table
            self.method_names_for_table = [
                'Ours', 'Target Only', 'Stacking','LLGC', 'Reweighting', 'Offset', 'SMS'
            ]
            self.files = OrderedDict()

            #self.files['LocalTransferNew-grad-bounds.pkl'] = 'Local Transfer New'
            self.files['LocalTransferNew-grad-bounds-opt_ft.pkl'] = 'Local Transfer New: Opt f_t'
            #self.files['LocalTransferNew-grad-bounds-scaleB.pkl'] = 'Local Transfer New: Scale'
            #self.files['LocalTransferNew-bounds-linearB.pkl'] = 'Local Transfer New: Linear B'

            self.files['TargetTransfer+NW.pkl'] = 'Target Only'
            self.files['StackTransfer+SKL-RidgeReg-jointCV.pkl'] = 'Stacking'
            self.files['SLL-NW.pkl'] = 'LLGC'
            self.files['CovShift.pkl'] = 'Reweighting'
            self.files['OffsetTransfer-jointCV.pkl'] = 'Offset Transfer'
            self.files['LocalTransferDeltaSMS_scale'] = 'SMS'
        elif plot_idx == PLOT_TABLE_OUR_METHODS:
            self.baseline_idx = 0
            self.data_names_for_table = names_for_table
            self.method_names_for_table = [
                'Ours', 'Ours: Fixed $f_T$', 'Ours: Fixed $f_T$+Scaling', 'Ours: Fixed $f_T$+linear $b$', 'Ours: Bound Constraints'
            ]
            self.files = OrderedDict()

            self.files['LocalTransferNew-grad-bounds-opt_ft.pkl'] = 'Local Transfer New: Opt f_t'
            self.files['LocalTransferNew-grad-bounds.pkl'] = 'Local Transfer New'
            self.files['LocalTransferNew-grad-bounds-scaleB.pkl'] = 'Local Transfer New: Scale'
            self.files['LocalTransferNew-bounds-linearB.pkl'] = 'Local Transfer New: Linear B'
            self.files['LocalTransferNew-grad-bounds-boundPerc=[10, 90].pkl'] = 'Our Method: Bound Constraints'

        elif plot_idx == PLOT_TABLE_VAL:
            self.files = OrderedDict()
            self.files['LocalTransferDelta_radius_l2_linear-b_clip-b_use-val-stacking.pkl'] = 'Ours: Linear, Stacking, VAL'
            self.files['LocalTransferDelta_radius_l2_linear-b_clip-b_use-val-stacking-sourceLOO.pkl'] = 'Ours: Linear, Stacking, LOO, VAL'
            self.files['StackTransfer+SKL-RidgeReg-VAL.pkl'] = 'Stacking, VAL'
        elif plot_idx == PLOT_ALPHA:
            self.files = OrderedDict()
            self.files['LocalTransferDelta_radius_l2_linear-b_clip-b.pkl'] = 'Ours: Linear'
            self.files['LocalTransferDelta_C3=0_radius_l2_linear-b.pkl'] = 'Ours: Linear, alpha=0'
        elif plot_idx == PLOT_TABLE_CONSTRAINED:
            self.baseline_idx = 1
            self.data_names_for_table = names_for_table
            self.method_names_for_table = [
                'Ours: Fixed $f_T$', 'Ours: Fixed $F_T$, Constraints'
            ]
            self.files = OrderedDict()
            self.files['LocalTransferNew-grad-bounds.pkl'] = 'Our Method'
            self.files['LocalTransferNew-grad-bounds-boundPerc=[10, 90].pkl'] = 'Our Method: Bound Constraints'

        if use_validation:
            test_files = OrderedDict()
            for f, leg in self.files.iteritems():
                f = helper_functions.remove_suffix(f, '.pkl')
                if f == 'OffsetTransfer-jointCV' or f.find('LocalTransferNew') == 0:
                    f += '-VAL.pkl'
                elif f == 'LocalTransferDelta_l2_lap-reg':
                    f = 'LocalTransferDelta_l2_use-val_lap-reg.pkl'
                elif f == 'TargetTransfer+NW':
                    f += '.pkl'
                elif f == 'StackTransfer+SKL-RidgeReg':
                    f += '-VAL.pkl'
                else:
                    #f += '_use-val.pkl'
                    f += '-VAL.pkl'
                #leg = 'VALIDATION: ' + leg
                test_files[f] = leg
            self.files = test_files

        if use_sms_plot_data_sets:
            if max_rows == 3:
                self.figsize = (4,10)
                self.borders = (.15,.9,.95,.05)
            else:
                self.figsize = (6,6)
                self.borders = (.1,.95,.95,.1)

            if data_set == bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER:
                self.ylims = (0,10)
            elif data_set == bc.DATA_SYNTHETIC_DELTA_LINEAR:
                self.ylims = (0,200)
            elif data_set == bc.DATA_SYNTHETIC_CROSS:
                self.ylims = (0,20)
        elif use_1d_data:
            self.borders = (.05,.95,.95,.05)
            if plot_idx in {PLOT_CONSTRAINED, PLOT_VALIDATION}:
                self.borders = (.1,.95,.95,.05)
            if max_rows == 3:
                self.figsize = (12,10)
            else:
                self.figsize = (14,6)
                self.borders = (.05,.95,.95,.1)
        else:
            if max_rows == 3:
                self.figsize = (4,4)
                self.borders = (.15,.9,.95,.05)
            else:
                self.figsize = (14,6)
                self.borders = (.05,.95,.95,.08)

        self.data_set_to_use = pc.data_set
        self.title = bc.data_name_dict.get(self.data_set_to_use, 'Unknown Data Set')
        self.show_legend_on_all = show_legend_on_all
        self.show_legend_on_missing_files = False
        self.crash_on_missing_files = crash_on_missing_files
        if self.use_1d_data and self.data_set_to_use < bc.DATA_SYNTHETIC_START:
            self.title += ' 1D'
        self.x_axis_string = 'Number of labeled target instances'
    '''
    def _results_files(self):
        dir = self.results_directory

        dir_prefixes = [
            'base/old/',
            'base/standard_scaler/',
            'base/min_max_scaler/',
        ]
        dir_prefixes = 'base/old/'
        i = 0
        files = []
        for key, value in self.files:
            d = dir_prefixes
            #d = dir_prefixes[i]
            files.append((d + '/' + dir + '/' + key, value))
            i += 1
        return files
    '''
    '''
    def _results_files(self):
        dir = self.results_directory
        dir = 'base/standard_scaler/' + self.results_directory
        files = []
        for key, value in self.files:
            files.append((dir + '/' + key, value))

        dir = 'base/min_max_scaler/' + self.results_directory
        for key, value in self.files:
            files.append((dir + '/' + key, value))
        return files
    '''


class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        self.config_list = []
        if use_all_data:
            for i in all_1d:
                pc = ProjectConfigs(i, use_1d_data=True)
                self.config_list += [MainConfigs(pc)]
            for i in real_data_sets_full:
                pc = ProjectConfigs(i, use_1d_data=False)
                self.config_list += [MainConfigs(pc)]
        else:
            for i in data_sets_for_exps:
                pc = ProjectConfigs(i)
                self.config_list += [MainConfigs(pc)]
        assert len(self.config_list) > 0



#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_WINE
#data_set_to_use = bc.DATA_BIKE_SHARING
#data_set_to_use = bc.DATA_PAIR_82_83

#data_set_to_use = bc.DATA_SYNTHETIC_CURVE
#data_set_to_use = bc.DATA_SYNTHETIC_SLANT
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER
#data_set_to_use = bc.DATA_SYNTHETIC_DELTA_LINEAR
#data_set_to_use = bc.DATA_SYNTHETIC_CROSS

if vis_table:
    viz_params = [{'data_set': d, 'use_1d_data': True} for d in all_1d] + \
                 [{'data_set': d, 'use_1d_data': False} for d in real_data_sets_full]
elif vis_batch:
    viz_params = [{'data_set': d, 'use_1d_data': True} for d in all_1d] + \
                 [{'data_set': d, 'use_1d_data': False} for d in real_data_sets_full]
else:
    viz_params = [{
        'data_set': data_set_to_use,
        'use_1d_data': use_1d_data
    }]
'''
elif use_1d_data:
    viz_params = [{'data_set': d} for d in all_1d]
else:
    viz_params = [{'data_set': d} for d in real_data_sets_full]
'''