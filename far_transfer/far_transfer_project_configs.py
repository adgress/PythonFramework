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
from copy import deepcopy

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
    'data_set',
    'oracle_guidance',
    'use_oracle_graph',
    'nystrom_percentage',
    'joint_cv'
]
data_set_to_use = None
#data_set_to_use = bc.DATA_SYNTHETIC_CLASSIFICATION
#data_set_to_use = bc.DATA_SYNTHETIC_CLASSIFICATION_LOCAL
#data_set_to_use = bc.DATA_NG

#data_set_to_use = bc.DATA_BOSTON_HOUSING
#data_set_to_use = bc.DATA_CONCRETE
#data_set_to_use = bc.DATA_WINE
#data_set_to_use = bc.DATA_BIKE_SHARING

#data_set_to_use = bc.DATA_POLLUTION_2
data_set_to_use = bc.DATA_CLIMATE_MONTH
#data_set_to_use = bc.DATA_IRS
#data_set_to_use = bc.DATA_TAXI
#data_set_to_use = bc.DATA_SYNTHETIC_PIECEWISE
#data_set_to_use = bc.DATA_ZILLOW
#data_set_to_use = bc.DATA_UBER

#data_set_to_use = bc.DATA_SYNTHETIC_CURVE
#data_set_to_use = bc.DATA_SYNTHETIC_SLANT
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_LINEAR_TRANSFER
#data_set_to_use = bc.DATA_SYNTHETIC_DELTA_LINEAR
#data_set_to_use = bc.DATA_SYNTHETIC_CROSS
#data_set_to_use = bc.DATA_SYNTHETIC_STEP_TRANSFER
#data_set_to_use = bc.DATA_SYNTHETIC_FLIP
#

VIZ_EXPERIMENTS = 0
VIZ_PAPER_ERROR = 1
VIZ_PAPER_NYSTROM = 2
VIZ_PAPER_GUIDANCE = 3

viz_type = VIZ_PAPER_NYSTROM

use_1d_data = True

show_legend_on_all = False
arguments = None
use_validation = True

run_experiments = True
run_batch_graph = False
run_batch_graph_nw = True
run_batch_baseline = True
run_batch_target_only = False
run_batch_stacking = True
run_batch_dummy = False

oracle_guidance = None
use_oracle_graph = False
#nystrom_percentage = .005
nystrom_percentage = None
stacking_joint_cv = False

BATCH_DATA_NONE = 0
BATCH_DATA_POSITIVE = 1
BATCH_DATA_NEGATIVE = 2
BATCH_ALL = 3

run_batch_datasets = BATCH_DATA_NONE
#run_batch_datasets = BATCH_DATA_POSITIVE
#run_batch_datasets = BATCH_DATA_NEGATIVE
#run_batch_datasets = BATCH_ALL

all_data_sets = [data_set_to_use]
if run_batch_datasets > 0:
    all_data_sets = []
    if run_batch_datasets in {BATCH_DATA_POSITIVE, BATCH_ALL}:
        all_data_sets += [
            bc.DATA_SYNTHETIC_PIECEWISE,
            bc.DATA_IRS,
            bc.DATA_CLIMATE_MONTH
        ]
    if run_batch_datasets in {BATCH_DATA_NEGATIVE, BATCH_ALL}:
        all_data_sets += [
            bc.DATA_SYNTHETIC_CROSS,
            bc.DATA_SYNTHETIC_SLANT,
            bc.DATA_SYNTHETIC_CURVE,
            bc.DATA_BIKE_SHARING,
            bc.DATA_BOSTON_HOUSING,
            bc.DATA_CONCRETE,
            #bc.DATA_POLLUTION_2,
        ]




FT_METHOD_GRAPH = 0
FT_METHOD_GRAPH_NW = 1
FT_METHOD_STACKING = 2
FT_METHOD_LOCAL = 3
FT_METHOD_LOCAL_NONPARAMETRIC = 4
FT_METHOD_SMS_DELTA = 5
FT_METHOD_OFFSET = 6
FT_METHOD_DUMMY = 7

other_method_configs = {
    'ft_method': FT_METHOD_LOCAL_NONPARAMETRIC,
    'predict_sample': None,
}

def apply_arguments(configs):
    if arguments.num_labels is not None:
        configs.overwrite_num_labels = arguments.num_labels
    if arguments.split_idx is not None:
        configs.split_idx = arguments.split_idx

class ProjectConfigs(bc.ProjectConfigs):
    def __init__(self, data_set=None, use_arguments=True, **kwargs):
        super(ProjectConfigs, self).__init__()
        self.oracle_guidance = oracle_guidance
        self.use_oracle_graph = use_oracle_graph
        self.nystrom_percentage = nystrom_percentage
        self.target_labels = np.empty(0)
        self.source_labels = np.empty(0)
        self.project_dir = 'far_transfer'
        self.num_labels = range(40,201,40)
        self.oracle_labels = np.empty(0)
        self.use_1d_data = use_1d_data
        self.joint_cv = stacking_joint_cv
        if data_set is None:
            data_set = data_set_to_use
        for key, value in other_method_configs.items():
            setattr(self, key, value)
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
        self.set_data_set(data_set)
        if use_arguments and arguments is not None:
            apply_arguments(self)

    def set_data_set(self, data_set):
        self.data_set = data_set
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
            self.num_labels = range(10,31,10)
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
        elif data_set == bc.DATA_PAIR_82_83:
            self.set_synthetic_regression('pair_data_82_83')
            self.num_labels = np.asarray([10,20,30])
        elif data_set == bc.DATA_PAIR_13_14:
            self.set_synthetic_regression('pair_data_13_14')
            self.num_labels = np.asarray([10,20,30])
        elif data_set == bc.DATA_SYNTHETIC_FLIP:
            self.set_synthetic_regression('synthetic_flip')
            self.num_labels = np.asarray([10, 20, 30])
        elif data_set == bc.DATA_POLLUTION_2:
            self.set_pollution(2, 500)
            self.num_labels = np.asarray([20, 40, 80, 160])
        elif data_set == bc.DATA_CLIMATE_MONTH:
            self.set_data_set_defaults('climate-month', source_labels=[0], target_labels=[4], is_regression=True)
            self.num_labels = np.asarray([20, 40, 80, 160])
        elif data_set == bc.DATA_UBER:
            self.set_data_set_defaults('uber', source_labels=[0], target_labels=[1], is_regression=True)
            self.num_labels = np.asarray([20, 40, 80, 160])
        elif data_set == bc.DATA_IRS:
            self.set_data_set_defaults('irs-income', source_labels=[0], target_labels=[1], is_regression=True)
            self.num_labels = np.asarray([20, 40, 80, 160])
        elif data_set == bc.DATA_SYNTHETIC_PIECEWISE:
            self.set_synthetic_regression('synthetic_piecewise')
            self.num_labels = np.asarray([10, 20, 30, 40])
        elif data_set == bc.DATA_TAXI:
            self.set_data_set_defaults('taxi', source_labels=[0], target_labels=[1], is_regression=True)
            #self.set_data_set_defaults('taxi2', source_labels=[0], target_labels=[1], is_regression=True)
            #self.num_labels = np.asarray([5, 10, 20, 40, 100, 200, 400, 800])
            #self.num_labels = np.asarray([5, 10, 20, 40, 100])
            self.num_labels = np.asarray([100, 200, 400, 800, 1600])
        elif data_set == bc.DATA_ZILLOW:
            self.set_data_set_defaults('zillow', source_labels=[0], target_labels=[1], is_regression=True)
            self.num_labels = np.asarray([5, 10, 20, 40, 80, 160])
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

    def set_pollution(self, id, size):
        self.loss_function = loss_function.MeanSquaredError()
        self.cv_loss_function = loss_function.MeanSquaredError()
        #assert self.use_1d_data == True
        s = 'pollution-%d-%d-norm' % (id, size)
        self.data_dir = 'data_sets/' + s
        self.data_name = s
        self.results_dir = s
        self.data_set_file_name = 'split_data.pkl'
        self.target_labels = np.asarray([0])
        self.source_labels = np.asarray([1])

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
        synthetic_dim = 1
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
        method_configs.use_validation = use_validation

        if self.data_set == bc.DATA_NG:
            method_configs.metric = 'cosine'
            method_configs.use_fused_lasso = False

        method_configs.constraints = []


        from methods import far_transfer_methods

        fuse_nw = transfer_methods.FuseTransfer(method_configs)
        fuse_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        target_nw = transfer_methods.TargetTranfer(method_configs)
        target_nw.base_learner = method.NadarayaWatsonMethod(method_configs)
        target_ridge = transfer_methods.TargetTranfer(method_configs)
        target_ridge.base_learner = method.SKLRidgeRegression(method_configs)
        nw = method.NadarayaWatsonMethod(method_configs)
        graph_transfer = far_transfer_methods.GraphTransfer(method_configs)



        from methods import semisupervised
        from methods import preprocessing
        ssl_regression = semisupervised.SemisupervisedRegressionMethod(method_configs)
        ssl_regression.preprocessor = preprocessing.TargetOnlyPreprocessor()

        dummy = method.SKLMeanRegressor(method_configs)
        dummy.preprocessor = preprocessing.TargetOnlyPreprocessor()

        graph_transfer_nw = far_transfer_methods.GraphTransferNW(method_configs)
        graph_transfer_nw.predict_sample = pc.predict_sample
        graph_transfer_nw.quiet = False
        #self.learner = target_nw
        offset_transfer = methods.local_transfer_methods.OffsetTransfer(method_configs)
        stacked_transfer = methods.transfer_methods.StackingTransfer(method_configs)
        sms_delta_transfer = methods.local_transfer_methods.LocalTransferDeltaSMS(method_configs)

        method_configs.metric = 'euclidean'
        method_configs.no_reg = False
        method_configs.use_g_learner = True
        method_configs.use_reg2 = True
        method_configs.use_fused_lasso = False
        method_configs.no_C3 = False
        method_configs.use_radius = False
        method_configs.include_scale = False
        method_configs.constant_b = False
        method_configs.linear_b = True
        method_configs.clip_b = True
        if pc.ft_method == FT_METHOD_LOCAL_NONPARAMETRIC:
            method_configs.linear_b = False
            method_configs.clip_b = False
            method_configs.use_radius = True


        dt_local_transfer = methods.local_transfer_methods.LocalTransferDelta(method_configs)
        if pc.ft_method == FT_METHOD_GRAPH:
            self.learner = graph_transfer
        elif pc.ft_method == FT_METHOD_GRAPH_NW:
            self.learner = graph_transfer_nw
        elif pc.ft_method == FT_METHOD_STACKING:
            self.learner = stacked_transfer
        elif pc.ft_method in {FT_METHOD_LOCAL,FT_METHOD_LOCAL_NONPARAMETRIC}:
            self.learner = dt_local_transfer
        elif pc.ft_method == FT_METHOD_SMS_DELTA:
            self.learner = sms_delta_transfer
        elif pc.ft_method == FT_METHOD_OFFSET:
            self.learner = offset_transfer
        elif pc.ft_method == FT_METHOD_DUMMY:
            self.learner = dummy
        else:
            assert False, 'Unknown ft_method'
        #self.learner.configs.use_validation = use_validation
        #self.learner.use_validation = use_validation


class MethodConfigs(bc.MethodConfigs):
    def __init__(self, pc):
        super(MethodConfigs, self).__init__()
        self.copy_fields(pc,pc_fields_to_copy)
        self.target_labels = pc.target_labels
        self.source_labels = pc.source_labels


def append_suffix_to_files(dict, suffix, legend_suffix):
    d = OrderedDict()
    for key,value in dict.iteritems():
        f = helper_functions.remove_suffix(key, '.pkl')
        f += suffix + '.pkl'
        d[f] = value + legend_suffix
    return d



class VisualizationConfigs(bc.VisualizationConfigs):
    def __init__(self, data_set=None, **kwargs):
        super(VisualizationConfigs, self).__init__()
        pc = ProjectConfigs(data_set, **kwargs)
        self.copy_fields(pc,pc_fields_to_copy)
        self.max_rows = 1
        self.show_legend_on_all = False
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

        viz_just_stacking = False
        self.files = OrderedDict()


        if viz_type == VIZ_PAPER_ERROR:
            self.files['StackTransfer+SKL-RidgeReg-target.pkl'] = 'Target only'
            self.files['StackTransfer+SKL-RidgeReg.pkl'] = 'Stacked'
            self.files['OffsetTransfer-jointCV.pkl'] = 'Offset'
            self.files['GraphTransferNW-use_rbf.pkl'] = 'Our Method'
            if run_batch_datasets != BATCH_DATA_NEGATIVE:
                self.files['GraphTransferNW-use_rbf-guidance_binary=0.2.pkl'] = 'Our Method, 20% Guidance'
                self.files['GraphTransferNW-use_rbf-oracle_graph.pkl'] = 'Our Method, Oracle Graph'
                #self.files['GraphTransferNW-use_rbf-nystrom=0.2.pkl'] = 'Our Method, rbf, 20% Nystrom'
        elif viz_type == VIZ_PAPER_NYSTROM:
            self.files['GraphTransferNW-use_rbf-nystrom=0.01.pkl'] = 'Our Method, rbf, 1% Nystrom'
            self.files['GraphTransferNW-use_rbf-nystrom=0.05.pkl'] = 'Our Method, rbf, 5% Nystrom'
            self.files['GraphTransferNW-use_rbf-nystrom=0.1.pkl'] = 'Our Method, rbf, 10% Nystrom'
            self.files['GraphTransferNW-use_rbf-nystrom=0.2.pkl'] = 'Our Method, rbf, 20% Nystrom'
            self.files['GraphTransferNW-use_rbf.pkl'] = 'Our Method'
            self.files['StackTransfer+SKL-RidgeReg-target.pkl'] = 'Target only'
        elif viz_type == VIZ_PAPER_GUIDANCE:
            self.files['GraphTransferNW-use_rbf-guidance_binary=0.05.pkl'] = 'Our Method, 5% Guidance'
            self.files['GraphTransferNW-use_rbf-guidance_binary=0.1.pkl'] = 'Our Method, 10% Guidance'
            self.files['GraphTransferNW-use_rbf-guidance_binary=0.2.pkl'] = 'Our Method, 20% Guidance'
            self.files['GraphTransferNW-use_rbf.pkl'] = 'Our Method'
        else:
            self.files['StackTransfer+SKL-RidgeReg.pkl'] = 'Stacked'
            self.files['StackTransfer+SKL-RidgeReg-jointCV.pkl'] = 'Stacked: Joint CV'
            #self.files['StackTransfer+SKL-RidgeReg-source.pkl'] = 'Stacked: Source only'
            self.files['StackTransfer+SKL-RidgeReg-target.pkl'] = 'Stacked: Target only'
            #self.files['StackTransfer+SKL-RidgeReg-bias.pkl'] = 'Stacked: Source and Bias'

            if not viz_just_stacking:
                #self.files['LocalTransferDelta_radius_l2_lap-reg.pkl'] = 'Local Transfer: Nonparametric'
                #self.files['LocalTransferDelta_radius_l2_lap-reg_knn.pkl'] = 'Local Transfer Nonparametric KNN'
                self.files['GraphTransferNW-use_rbf.pkl'] = 'Graph Transfer NW, rbf'
                if run_batch_datasets != BATCH_DATA_NEGATIVE:
                    self.files['GraphTransferNW-use_rbf-oracle_graph.pkl'] = 'Graph Transfer NW, rbf, oracle graph'
                    self.files['GraphTransferNW-use_rbf-oracle_graph-nystrom=0.2.pkl'] = 'Graph Transfer NW, rbf, oracle graph, 20% Nystrom'
                    self.files['GraphTransferNW-use_rbf-guidance_binary=0.1.pkl'] = 'Graph Transfer NW, rbf, 10% oracle binary'
                    self.files['GraphTransferNW-use_rbf-guidance_binary=0.2.pkl'] = 'Graph Transfer NW, rbf, 20% oracle binary'
                    #self.files['GraphTransferNW-use_rbf-guidance=0.2-nystrom=0.5.pkl'] = 'Graph Transfer NW, rbf, 20% oracle binary, 50% Nystrom'
                    #self.files['GraphTransferNW-use_rbf-guidance_binary=0.2-nystrom=0.2.pkl'] = 'Graph Transfer NW, rbf, 20% oracle binary, 20% binary'
                    #self.files['GraphTransferNW-use_rbf-nystrom=0.1.pkl'] = 'Graph Transfer NW, rbf, 10% Nystrom'
                    self.files['GraphTransferNW-use_rbf-nystrom=0.2.pkl'] = 'Graph Transfer NW, 20% Nystrom'
                    self.files['GraphTransfer.pkl'] = 'Graph Transfer'
                    #self.files['GraphTransfer_tr.pkl'] = 'Graph Transfer: Just transfer'
                    #self.files['GraphTransfer_ta.pkl'] = 'Graph Transfer: Just target'
        if use_validation:
            if viz_type != VIZ_EXPERIMENTS:
                self.files = append_suffix_to_files(self.files, '-VAL', '')
            else:
                self.files = append_suffix_to_files(self.files, '-VAL', ', VAL')
            #self.files['LocalTransferDelta_l2_linear-b_clip-b_use-val.pkl'] = 'Local Transfer VAL'
            #self.files['LocalTransferDelta_radius_l2_use-val_lap-reg.pkl'] = 'Local Transfer Nonparametric VAL'
            #self.files['LocalTransferDelta_radius_l2_use-val_lap-reg_knn.pkl'] = 'Local Transfer Nonparametric KNN VAL'
        else:
            pass
            #self.files['LocalTransferDelta_l2_linear-b_clip-b.pkl'] = 'Local Transfer'
            #self.files['OffsetTransfer.pkl'] = 'Offset'
            #self.files['OffsetTransfer-jointCV.pkl'] = 'Offset Joint CV'
            #self.files['GraphTransferNW-transfer_sparse=10.pkl'] = ' Graph Transfer NW Sparse=10'
            #self.files['GraphTransferNW-use_rbf.pkl'] = 'Graph Transfer RBF'
            #self.files['LocalTransferDeltaSMS.pkl'] = 'SMS Delta'
        self.title = self.results_dir


class BatchConfigs(bc.BatchConfigs):
    def __init__(self, pc):
        super(BatchConfigs, self).__init__()
        self.config_list = []
        '''
        configs_updates = [
            {'just_transfer': False, 'just_target': False},
            {'just_transfer': True, 'just_target': False},
            {'just_transfer': False, 'just_target': True},
        ]
        '''
        for d in all_data_sets:
            if run_batch_target_only:
                pc2 = ProjectConfigs(d)
                pc2.ft_method = FT_METHOD_GRAPH
                m = MainConfigs(pc2)
                m.learner.just_transfer = False
                m.learner.just_target = True
                self.config_list.append(m)
            if run_batch_graph:
                pc2 = ProjectConfigs(d)
                pc2.ft_method = FT_METHOD_GRAPH
                m = MainConfigs(pc2)
                m.learner.just_transfer = False
                m.learner.just_target = False
                self.config_list.append(m)
                m = deepcopy(m)
                m.learner.just_transfer = True
                self.config_list.append(m)
                m = deepcopy(m)
                m.learner.just_transfer = False
                m.learner.just_target = True
                self.config_list.append(m)
                m = deepcopy(m)
                m.learner.just_transfer = True
                m.learner.just_target = False
                self.config_list.append(m)
            if run_batch_graph_nw:
                pc2 = ProjectConfigs(d)
                pc2.ft_method = FT_METHOD_GRAPH_NW
                m = MainConfigs(pc2)
                self.config_list.append(m)
            if run_batch_stacking:
                pc2 = ProjectConfigs(d)
                pc2.ft_method = FT_METHOD_STACKING
                m = MainConfigs(pc2)
                self.config_list.append(m)
            if run_batch_dummy:
                pc2 = ProjectConfigs(d)
                pc2.ft_method = FT_METHOD_DUMMY
                m = MainConfigs(pc2)
                self.config_list.append(m)
            if run_batch_baseline:
                pc2 = ProjectConfigs(d)
                '''
                pc2.ft_method = FT_METHOD_LOCAL_NONPARAMETRIC
                m = MainConfigs(pc2)
                self.config_list.append(m)
                pc2 = deepcopy(pc2)
                pc2.ft_method = FT_METHOD_SMS_DELTA
                m = MainConfigs(pc2)
                self.config_list.append(m)
                '''
                pc2 = deepcopy(pc2)
                pc2.ft_method = FT_METHOD_OFFSET
                m = MainConfigs(pc2)
                self.config_list.append(m)
                '''
                pc2.ft_method = FT_METHOD_LOCAL
                m = MainConfigs(pc2)
                self.config_list.append(m)
                if not use_validation:
                    pc2.ft_method = FT_METHOD_LOCAL_NONPARAMETRIC
                    m = MainConfigs(pc2)
                    self.config_list.append(m)
                '''

viz_params = [
    {'data_set': d} for d in all_data_sets
]