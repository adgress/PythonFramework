__author__ = 'Aubrey and Evan'
"""
This module contains estimator classes.
"""
import abc
import numpy as np
from sklearn.preprocessing import StandardScaler as sklearn_scaler
from utility.HelperFunctions import get_x_y_from_file, convert_to_float
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.dummy import DummyRegressor
import pyspark
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.tree import RandomForest #only works for spark 1.2.0 and later
from pyspark.mllib.feature import StandardScaler as mllib_scaler
from pyspark.mllib.linalg import Vectors
from loss_functions.LossFunction import generate_loss_function_object






class Estimator(object):
    """
    Base class for all estimators
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.training_size = None
        self.rdd_scaler = mllib_scaler(withMean=True, withStd=True)
        self.np_scaler = sklearn_scaler()
        self.y_predicted = None
        self.y_real = None
        self.train_targets = None
        self.test_targets = None
        self.time_to_fit = None


    def set_params(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def set_score_fxns(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


    def set_time_to_fit(self, time_to_fit):
        self.time_to_fit = time_to_fit

    def fit(self, data):
        x, y, target_ids = self.unpack_x_y(data, 'fit')
        self.train_targets = target_ids
        return self._fit(x, y)

    def predict(self, data):
        x, y, target_ids = self.unpack_x_y(data, 'predict')
        #self.test_targets = target_ids
        y_predicted = self._predict(x)

        return y_predicted

    def unpack_x_y(self, data, call=''):

        if isinstance(data, pyspark.rdd.RDD):
            data_rdd, target_ids = self.preprocess_rdd(data)
            x = data_rdd.map(lambda labeled_pt: labeled_pt.features)
            y = data_rdd.map(lambda labeled_pt: labeled_pt.label)

        elif isinstance(data, np.ndarray):
            data_matrix, target_ids = self.preprocess_np_array(data)
            x = data_matrix[:, :-1]
            y = data_matrix[:, -1]

            if call == 'fit':
                x = self.np_scaler.fit_transform(x)
            elif call == 'predict':
                x = self.np_scaler.transform(x)

        else:
            assert False, "Data not of correct type (must be rdd or numpy array!)"
        return x, y, target_ids

    def preprocess_rdd(self,data_rdd):
        train_targets = data_rdd.keys()
        data_rdd = data_rdd.map(lambda (name, labeled_pt): labeled_pt)
        #data_rdd = self.scale_data(data_rdd)
        return data_rdd, train_targets

    def preprocess_np_array(self, data_matrix):

        #labels not already removed
        target_ids = data_matrix[:, 0]
        data_matrix = data_matrix[:,1:]
        return data_matrix, target_ids


    def scale_rdd_data(self,data):
        #will not work until spark 1.3 when the rdd.zip bug is fixed
        labels = data.map(lambda x: x.label)
        features = data.map(lambda x: x.features)
        scaler = self.rdd_scaler.fit(features)
        scaled_features = scaler.transform(features.map(lambda x: Vectors.dense(x.toArray())))
        scaled_data = labels.zip(scaled_features)
        scaled_data = scaled_data.map(lambda (label, features): LabeledPoint(label, features))
        return scaled_data


    def cv_score(self, test_data):
        return self.score(test_data, self.cv_score_fxn)

    def score(self, test_data, *score_fxns):

        _, y_real, target_ids = self.unpack_x_y(test_data )
        y_predicted = self.predict(test_data)
        scores = []
        for score_fxn in score_fxns:
            scores.append(self.calculate_score(score_fxn, y_real, y_predicted, target_ids))
        if len(scores) == 1:
            scores = scores[0]
        return scores

    def results_scores(self, test_data):
        score_names = self.results_score_fxns.split(',')
        scores = self.score(test_data, *score_names)
        scores_dict = dict(zip(score_names, scores))
        return scores_dict


    def calculate_score(self, score_fxn_name, y_real, y_predicted, test_targets):
        scorer = generate_loss_function_object(score_fxn_name)
        return scorer.compute_score(y_real, y_predicted, test_targets)


    @abc.abstractmethod
    def _predict(self, x):
        return

    @abc.abstractmethod
    def _fit(self, x, y):
        return


    @abc.abstractmethod
    def get_name(self):
        """
        Returns a display name for the Estimator
        """
        return


    def get_short_name(self):
        """
        Returns a short name for the Estimator
        """
        return self.get_name()



    def get_params(self):
        return {}



class ScikitLearnEstimator(Estimator):
    """
    Estimator that uses one of the Scikit Learn estimators.
    """
    _short_name_dict = {
        'KNeighborsRegressor': 'KNR',
        'RandomForestRegressor': 'RFR',
        'AdaBoostRegressor': 'ABR',
        'ExtraTreesRegressor': 'ETR',
        'GradientBoostingRegressor': 'GBR',
        'RadiusNeighborsRegressor': 'RNR',
        'DecisionTreeRegressor': 'DTR',
        'LinearRegression': 'LNR',
        'DummyRegressor': 'MDR',
    }

    def __init__(self, skl_estimator=None):
        super(ScikitLearnEstimator, self).__init__()
        self.skl_estimator = skl_estimator
        if hasattr(skl_estimator, 'n_jobs'):
            self.set_params(n_jobs=-1)

    def _fit(self,x,y):
        x, y = self.skl_preprocess(x, y)
        self.skl_estimator.fit(x, y)
        return self

    def _predict(self, x):
        x = self.skl_preprocess(x)
        return self.skl_estimator.predict(x)

    def skl_preprocess(self, *args):
        ret = []
        for arg in args:
            if isinstance(arg, pyspark.rdd.RDD):
                ret.append(arg.collect())
            else:
                ret.append(arg)
        if len(ret) == 1:
            ret = ret[0]
        return ret


    def get_name(self):
        return 'SciKitLearn-' + self._get_skl_estimator_name()

    def get_short_name(self):
        return 'SKL-' + ScikitLearnEstimator._short_name_dict[self._get_skl_estimator_name()]

    def _get_skl_estimator_name(self):
        return repr(self.skl_estimator).split('(')[0]

    def set_params(self, **kwargs):
        self.skl_estimator.set_params(**kwargs)
        return self

    def get_params(self):
        return self.skl_estimator.get_params()


class SKL_KNeighbors(ScikitLearnEstimator):
    def __init__(self):
        super(SKL_KNeighbors, self).__init__(KNeighborsRegressor())


class SKL_RandomForest(ScikitLearnEstimator):
    def __init__(self):
        super(SKL_RandomForest, self).__init__(RandomForestRegressor())


class SKL_DecisionTree(ScikitLearnEstimator):
    def __init__(self):
        super(SKL_DecisionTree, self).__init__(DecisionTreeRegressor())

class SKL_AdaBoost(ScikitLearnEstimator):
    def __init__(self):
        super(SKL_AdaBoost, self).__init__(AdaBoostRegressor())

class SKL_LinearRegression(ScikitLearnEstimator):
    def __init__(self):
        super(SKL_LinearRegression, self).__init__(LinearRegression())

class SKL_Mean(ScikitLearnEstimator):
    def __init__(self):
        super(SKL_Mean, self).__init__(DummyRegressor())






class GuessEstimator(Estimator):
    """
    Estimator that always guesses a uniform random number in [0,1]
    """
    def __init__(self):
        super(GuessEstimator, self).__init__()

    def _fit(self, x,y):
        return self

    def _predict(self, x):
        x = self.guess_preprocess(x)
        pred = np.random.rand(x.shape[0], x.shape[1])
        return pred

    def guess_preprocess(self, x):
        if isinstance(x, pyspark.rdd.RDD):
            x = x.collect()
        return x

    def get_name(self):
        return 'Guess'



def get_unique_file_name(file_name, file_id):
    return file_name + str(file_id) + '.txt'





class MLLibRegressionEstimator(Estimator):
    def __init__(self,learner_module):
        super(MLLibRegressionEstimator, self).__init__()
        self.learner_module = learner_module
        self.estimator_args = {}



    def fit(self, data_rdd):
        labeled_data_rdd, target_ids = self.preprocess_rdd(data_rdd)
        self.train_targets = target_ids

        self.model = self.learner_module.trainRegressor(labeled_data_rdd, {}, **self.estimator_args)
        self.training_size = data_rdd.count()
        return self


    def predict(self, data_rdd):
        labeled_data_rdd, target_ids = self.preprocess_rdd(data_rdd)
        self.test_targets = target_ids
        x = labeled_data_rdd.map(lambda labeled_pt: labeled_pt.features)
        y_pred = self.model.predict(x)

        return y_pred

    def _predict(self, x):
        pass

    def _fit(self, x, y):
        pass



    def set_params(self,**kwargs):
        self.estimator_args = kwargs
        return self

    def get_params(self):
        return self.estimator_args

    def get_name(self):
        return 'MLLib_'

    def set_spark_context(self, sc):
        self.spark_context = sc


class MLLibDecisionTree(MLLibRegressionEstimator):
    def __init__(self):
        super(MLLibDecisionTree, self).__init__(DecisionTree)

    def get_name(self):
        return 'MLLib_' + 'DTR'



class MLLibRandomForest(MLLibRegressionEstimator):
    def __init__(self):
        super(MLLibRandomForest, self).__init__(RandomForest)

    def get_name(self):
        return 'MLLib_' + 'RFR'







