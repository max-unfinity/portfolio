import datetime
import json
import os

import numpy as np
from sklearn.feature_selection import f_regression, mutual_info_regression, f_classif, mutual_info_classif
from sklearn.feature_selection.from_model import _get_feature_importances
from sklearn.linear_model import Lasso

from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.abstract_importance import AbstractImportance
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.verbose import Verbose
from src.concore.tdata.targetinfo import TargetInfo


def EstimatorFromFunction(f, **f_args):
    def fit(self, X, y):
        score = f(X, y, **f_args)
        if isinstance(score, tuple):
            score = score[0]
        self.feature_importances_ = np.nan_to_num(score)
        return score

    return type('ESTIMATOR_' + f.__name__, (), {'fit': fit})


class ESTIMATORS:
    """You can use one of this estimators or any model from sklearn. For example Lasso"""
    f_regression = EstimatorFromFunction(f_regression)
    f_classif = EstimatorFromFunction(f_classif)
    mutual_info_regression = EstimatorFromFunction(mutual_info_regression)
    mutual_info_classif = EstimatorFromFunction(mutual_info_classif)


default_estimator_args = {
    Lasso: {'alpha': 0.005}
}


class SklearnImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


class SklearnImportance(AbstractImportance):
    def _get_estimator(self, name):
        if self._target_info.task == 'regression':
            estimators = {
                'f_score': ESTIMATORS.f_regression,
                'mutual_info': ESTIMATORS.mutual_info_regression
            }
        else:
            estimators = {
                'f_score': ESTIMATORS.f_classif,
                'mutual_info': ESTIMATORS.mutual_info_classif
            }

        if name not in estimators:
            raise SklearnImportanceException(f'Invalid sklearn method: {name}')
        return estimators[name]

    def __init__(self,
                 name: str,
                 folder_path: str,
                 target_info: TargetInfo,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 idata_name=None,
                 threshold=1e-5,
                 estimator=ESTIMATORS.f_regression,
                 **estimator_args
                 ):
        super(SklearnImportance, self).__init__(
            name,
            folder_path,
            target_info,
            targets_generator,
            features_generator,
            split_name,
            idata_name,
        )
        if isinstance(estimator, str):
            estimator = self._get_estimator(estimator)

        if (estimator_args == {}) and (estimator in default_estimator_args):
                estimator_args = default_estimator_args[estimator]
        self._estimator = estimator(**estimator_args)

        self._estimator_args = estimator_args
        self._threshold = threshold

    def _is_reverse_rank(self):
        return False

    def calculate_features_importance(self):
        x_train, y_train, x_test, y_test = self.create_train_test_data()

        Verbose.instance.print(1, f'Calculating importance for {len(self._features)} features')

        self._estimator.fit(x_train, y_train)
        ranks = np.nan_to_num(_get_feature_importances(self._estimator))

        self._ranks = list(map(lambda x: 0. if x < self._threshold else x, ranks))

        self.logger.log({
            'date': datetime.datetime.now(),
            'name': self.name,
            'estimator': self._estimator.__class__.__name__,
            'estimator_args': json.dumps(self._estimator_args),
            'all_features': json.dumps(self._features),
            'ranking': json.dumps(self.ranks)
        })

    @property
    def logger(self):
        if not self._logger:
            self._logger = MetricLogger(
                self.name,
                os.path.join(self._folder_path, self.name),
                delimiter=";",
                values=[
                    "date",
                    "name",
                    "estimator",
                    "estimator_args",
                    "all_features",
                    "ranking",
                ]
            )
        return self._logger

    def read_log(self):
        pass
