import datetime
import json
import os

import numpy as np
import sklearn.ensemble as ens

from contrib.boruta.boruta import BorutaPy
from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.abstract_importance import AbstractImportance
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.verbose import Verbose
from src.concore.tdata.targetinfo import TargetInfo


def _select_features_from_ranks(ranks: list, features: list, ranking_threshold: int):
    selected_features = [feature for feature, rank in zip(features, ranks) if rank <= ranking_threshold]
    return selected_features


def _select_features_by_method(res: dict, selection_method: str):
    output_features = []
    if selection_method == "unique":
        for key in res.keys():
            output_features += res[key]
        return list(np.unique(output_features))

    elif selection_method == "match":
        keys = list(res.keys())
        for i in range(len(keys) - 1):
            res[keys[i + 1]] = np.intersect1d(res[keys[i]], res[keys[i + 1]])
        if len(res[keys[-1]]) == 0:
            raise ValueError("results of feature importance estimation algorithms don't match")
        return res[keys[-1]]
    else:
        raise ValueError("selection method unknown")


BORUTA_CORES = {
    'random_forest': {
        'object': ens.RandomForestRegressor,
        'default_parameters': {
            'n_estimators': 50,
            'max_depth': 7,
        },
    },
    'extra_trees': {
        'object': ens.ExtraTreesRegressor,
        'default_parameters': {
            'n_estimators': 50,
            'max_depth': 7,
        },
    }
}


class BorutaImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


class BorutaImportance(AbstractImportance):
    def __init__(self,
                 name: str,
                 folder_path: str,
                 target_info: TargetInfo,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 idata_name=None,
                 core_type: str = 'random_forest',
                 core_parameters: dict = None,
                 boruta_n_estimators='auto',
                 boruta_max_iter=100,
                 include_tentative=False
                 ):
        if core_type not in BORUTA_CORES.keys():
            raise BorutaImportanceException(f'Invalid boruta core type: {core_type}')

        self._core_type = core_type
        self._core_parameters = core_parameters
        self._include_tentative = include_tentative
        self._boruta_n_estimators = boruta_n_estimators
        self._boruta_max_iter = boruta_max_iter

        super(BorutaImportance, self).__init__(
            name,
            folder_path,
            target_info,
            targets_generator,
            features_generator,
            split_name,
            idata_name,
        )

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
                    "all_features",
                    "algorithm_name",
                    "ranking"
                ]
            )
        return self._logger

    def calculate_features_importance(self):

        x_train, y_train, x_test, y_test = self.create_train_test_data()

        self._features = list(x_train.columns.values)

        Verbose.instance.print(1, f'Calculating importance for {len(self._features)} features')

        core = BORUTA_CORES[self._core_type]['object']
        if self._core_parameters is None:
            parameters = BORUTA_CORES[self._core_type]['default_parameters']
        else:
            parameters = self._core_parameters
        core = core(**parameters)

        feature_selector = BorutaPy(core, n_estimators=self._boruta_n_estimators, max_iter=self._boruta_max_iter,
                                    verbose=Verbose.instance.verbose_level, random_state=1)
        feature_selector.fit(x_train.values, y_train.values.ravel())
        self._ranks = feature_selector.ranking_.tolist()

        self.logger.log({
            'date': datetime.datetime.now(),
            'name': self.name,
            'algorithm_name': self._core_type,
            'all_features': json.dumps(self._features),
            'ranking': json.dumps(self.ranks)
        })

    def _is_reverse_rank(self):
        return True

    def get_norm_ranks(self):
        threshold = 2 if self._include_tentative else 1
        ranks = np.array(self.ranks)

        relevant = np.zeros(len(ranks), dtype=bool)
        relevant[ranks <= threshold] = True

        ranks = abs(ranks - 3)
        ranks[relevant == False] = 0.

        if ranks.sum() != 0:
            ranks = ranks / ranks.sum()
        return list(ranks)

    def read_log(self):
        pass
