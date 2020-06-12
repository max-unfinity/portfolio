import datetime
import json
import os

import numpy as np
import pandas as pd
from tsfresh.feature_selection.relevance import calculate_relevance_table

from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.abstract_importance import AbstractImportance
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.verbose import Verbose
from src.concore.tdata.targetinfo import TargetInfo


class TsfreshImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


class TsfreshImportance(AbstractImportance):
    def __init__(self,
                 name: str,
                 folder_path: str,
                 target_info: TargetInfo,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 idata_name=None,
                 **tsfresh_args
                 ):

        super(TsfreshImportance, self).__init__(
            name,
            folder_path,
            target_info,
            targets_generator,
            features_generator,
            split_name,
            idata_name,
        )

        self._tsfresh_args = tsfresh_args

    def _is_reverse_rank(self):
        return True

    def calculate_features_importance(self):

        x_train, y_train, x_test, y_test = self.create_train_test_data()

        Verbose.instance.print(1, f'Calculating importance for {len(self._features)} features')

        # x and y must to be pd.DataFrame and pd.Series for tsfresh
        x = pd.DataFrame(x_train)
        y = pd.Series(np.array(y_train).reshape(len(y_train)), index=x.index)

        t = calculate_relevance_table(x, y, **self._tsfresh_args).reindex(index=self._features)
        t.loc[t['relevant'] == False, 'p_value'] = t['p_value'].max()

        self._ranks = t['p_value'].tolist()

        self.logger.log({
            'date': datetime.datetime.now(),
            'name': self.name,
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
                    "all_features",
                    "ranking"
                ]
            )
        return self._logger

    def read_log(self):
        pass
