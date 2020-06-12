import datetime
import os
import random

import json_tricks as json
import numpy as np
import pandas as pd
import shap
import xgboost

from src.concore.data_preparation.preprocessing.workflow import CreationWorkflow
from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.abstract_importance import AbstractImportanceException
from src.concore.grader.importances.random_subsampling_mixin import RandomSubsamplingMixin
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.verbose import Verbose
from src.concore.imodel.imodel_factory import create_imodel
from src.concore.tdata.split.split import TTV
from src.concore.tdata.targetinfo import TargetInfo

# TODO switch hardcoded SHAP methods to user's customized methods
MODELS = {
    'xgb': {
        'regression': {
            'class': xgboost.XGBRegressor,
            'params': {
                'eta': 1,
                'max_depth': 10,
                'n_estimators': 50
            },
        },
        'classification': {
            'class': xgboost.XGBClassifier,
            'params': {
                'max_depth': 10,
                'n_estimators': 50,
                'objective': 'binary:logistic'
            },
        }
    }
}

# TODO add all other explainers types
EXPLAINERS = {
    'xgb': shap.TreeExplainer,
    'rf': shap.TreeExplainer
}


class SHAPImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


class SHAPImportance(RandomSubsamplingMixin):
    def __init__(self,
                 name: str,
                 folder_path: str,
                 target_info: TargetInfo,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 idata_name=None,
                 model: str = 'xgb',
                 model_args: dict = None,
                 ):

        if model not in MODELS:
            raise SHAPImportanceException(f'Invalid SHAP method: {model}')

        self.model_class = MODELS[model][target_info.task]['class']
        self.model_class_args = model_args or MODELS[model][target_info.task]['params']
        self.explainer = EXPLAINERS[model]

        super(SHAPImportance, self).__init__(
            name,
            folder_path,
            target_info,
            targets_generator,
            features_generator,
            split_name,
            idata_name,
        )

    def _is_reverse_rank(self):
        return False

    def _step(self, seed: int):
        """
        perform one calculation of given random seed
        :param seed:
        :return:
        """
        random.seed(seed)
        Verbose.instance.print(2, f'Step with seed {seed}')
        workflows_subsample = random.sample(list(self._workflows), self._subsample_len)
        workflow = CreationWorkflow([block for wf in workflows_subsample for block in wf])
        subsample_feature_generator = FeatureGenerator(
            'subsample',
            self._features_generator.idata_container,
            self._features_generator.global_data_container,
        )
        subsample_feature_generator.workflow = workflow
        model = create_imodel(
            self._core_parameters,
            self._split_name,
            [self._target_info],
            self._targets_generator,
            subsample_feature_generator,
            '',
        )
        model.run()
        data = subsample_feature_generator.get_splitted_data(
            self._split_name,
        )[TTV.VALIDATION]
        feature_values = model.model.explain_shap(data, averaged_norm=True)

        self.logger.log({
            'date': datetime.datetime.now(),
            'name': 'shap_grader',
            'seed': seed,
            'algorithm_features': json.dumps(list(feature_values.keys())),
            'algorithm_ranking': json.dumps(list(feature_values.values()))
        })

        return model.model.feature_importance

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
                    "seed",
                    "algorithm_features",
                    "algorithm_ranking",
                ]
            )
        return self._logger

    def read_log(self):
        """
        reading log with results and calculation of average importance
        :return:
        """
        feature_scores = {}

        data = pd.read_csv(self._log_path, sep=';')

        for row in data.iterrows():
            features = json.loads(row[1].algorithm_features)
            ranking = json.loads(row[1].algorithm_ranking)

            for feature, rank in zip(features, ranking):
                if feature in feature_scores.keys():
                    feature_scores[feature].append(rank)
                else:
                    feature_scores[feature] = [rank]

        feature_mean_scores = {}
        for feature_name in feature_scores.keys():
            feature_mean_scores[feature_name] = np.array(feature_scores[feature_name]).mean()

        self._feature_scores = feature_mean_scores
