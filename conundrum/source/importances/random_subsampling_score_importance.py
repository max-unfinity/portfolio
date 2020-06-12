import datetime
import random

import json_tricks as json
import numpy as np
import pandas as pd

from src.concore.data_preparation.preprocessing.workflow import CreationWorkflow
from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.random_subsampling_mixin import RandomSubsamplingMixin
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.metrics.metrics import all_metrics
from src.concore.i_utils.verbose import Verbose
from src.concore.imodel.imodel_factory import create_imodel
from src.concore.model.parameters.abstract_parameters import AbstractParameters
from src.concore.tdata.targetinfo import TargetInfo


class RandomSubsamplingScoreImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_modifiers': lambda x: None,
}


class RandomSubsamplingScoreImportance(RandomSubsamplingMixin):
    def __init__(self,
                 name: str,
                 folder_path: str,
                 target_info: TargetInfo,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 core_parameters: AbstractParameters = None,
                 n_estimations=10,
                 subsample_len=5,
                 target_metric='mse',
                 metric_function=None,
                 ):
        """

        :param name: name of the importance
        :param folder_path: path to save log results
        :param target_info: target to work with
        :param features_generator: feature generator with features to analyze
        :param split_name: train test split name
        :param core_parameters: parameters of the given ml algorithm
        :param n_estimations: amount of estimations to perform
        :param subsample_len: length of the features subsample to run core
        :param target_metric: metric to select top features
        :param metric_function: in case multiple output (not tested)
        """
        self._target_metric = target_metric
        self._metric_function = metric_function

        RandomSubsamplingMixin.__init__(
            self,
            name,
            folder_path,
            target_info,
            targets_generator,
            features_generator,
            split_name,
            core_parameters,
            n_estimations,
            subsample_len=subsample_len
        )

    def _is_reverse_rank(self):
        if all_metrics[self._target_metric].minimize:
            reverse = True
        else:
            reverse = False
        return reverse

    def _step(self, seed):
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
        predict_way = 'default'
        if self._target_info.task == 'classification':
            predict_way = 'proba'
        model.run(predict_way=predict_way)
        if self._metric_function:
            result = self._metric_function(model.metrics)
        else:
            # TODO: target_metric may not be in target_info. What to do then?
            result = np.array([model.metrics[key] for key in model.metrics.keys()
                               if self._target_metric in key]).mean()

        self.logger.log({
            'date': datetime.datetime.now(),
            'seed': seed,
            'features': json.dumps(list(model.model.feature_importance.keys())),
            'score': result,
        })

        return model.model.feature_importance, result

    @property
    def logger(self):
        if not self._logger:
            self._logger = MetricLogger(self.name,
                                        self._log_path,
                                        delimiter=";",
                                        values=[
                                            'date',
                                            'seed',
                                            'features',
                                            'score'])
        return self._logger

    def read_log(self):
        """
        reading log with results and calculation of average score
        :return:
        """
        data = pd.read_csv(self._log_path, sep=';')

        feature_scores = {}

        for row in data.iterrows():
            features = json.loads(row[1].features)
            score = row[1].score

            for feature in features:
                if feature in feature_scores.keys():
                    feature_scores[feature].append(score)
                else:
                    feature_scores[feature] = [score]

        feature_mean_scores = {}
        for feature_name in feature_scores.keys():
            feature_mean_scores[feature_name] = np.array(feature_scores[feature_name]).mean()

        self._feature_scores = feature_mean_scores
