import datetime
import os
import random

import json_tricks as json
import numpy as np
import pandas as pd

from src.concore.data_preparation.preprocessing.workflow import CreationWorkflow
from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.random_subsampling_mixin import RandomSubsamplingMixin
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.verbose import Verbose
from src.concore.imodel.imodel_factory import create_imodel


class RandomSubsamplingAveragedImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_modifiers': lambda x: None,
}


class RandomSubsamplingAveragedImportance(RandomSubsamplingMixin):

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

        # CreationWorkflow.get_union_cwf(self._workflows)

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

        self.logger.log({
            'date': datetime.datetime.now(),
            'seed': seed,
            'algorithm_features': json.dumps(list(model.model.feature_importance.keys())),
            'algorithm_ranking': json.dumps(list(model.model.feature_importance.values()))
        })

        return model.model.feature_importance

    @property
    def logger(self):
        if not self._logger:
            self._logger = MetricLogger(self.name,
                                        os.path.join(self._folder_path, self.name),
                                        delimiter=";",
                                        values=[
                                            'date',
                                            'seed',
                                            'algorithm_features',
                                            'algorithm_ranking'])
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
