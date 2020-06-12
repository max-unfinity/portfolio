import abc
import os
import warnings
from enum import Enum

import numpy as np

from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.i_utils.metadatakeeper import MetaDataKeeper
from src.concore.i_utils.metrics.abstractmetric import AbstractMetric
from src.concore.i_utils.verbose import Verbose
from src.concore.tdata.split.split import TTV
from src.concore.tdata.targetinfo import TargetInfo


class ImportanceType(Enum):
    MINIMIZE = 0
    MAXIMIZE = 1
    RANKING = 3

    @classmethod
    def from_metric(cls, metric: AbstractMetric):
        if metric.minimize:
            return cls.MINIMIZE
        else:
            return cls.MAXIMIZE


class AbstractImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_target_info': lambda x: x.uuid,
    '_targets_generator': lambda x: x.name,
    '_features_generator': lambda x: x.name,
    '_split': lambda x: x.name,
    '_logger': lambda x: None,
    '_core_parameters': lambda x: x.metadata
}


class AbstractImportance(abc.ABC, MetaDataKeeper):
    def __init__(self,
                 name: str,
                 folder_path: str,
                 target_info: TargetInfo,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 idata_name: str = None,
                 ):
        self._name = name
        self._folder_path = folder_path
        self._target_info = target_info
        self._targets_generator = targets_generator
        self._features_generator = features_generator
        self._split_name = split_name
        self._idata_name = idata_name
        self._logger = None
        self._log_path = os.path.join(self._folder_path, self.name)
        self._class_name = self.__class__.__name__
        self._features = features_generator.workflow.get_features_names()

        self._ranks = None

        os.makedirs(self._folder_path, exist_ok=True)

        MetaDataKeeper.__init__(self, EXCEPTIONAL_FIELDS)

    @classmethod
    def from_metadata(cls,
                      metadata: dict,
                      targets_info: list,
                      targets_generator: FeatureGenerator,
                      features_generator: FeatureGenerator,
                      ):
        sel_target_info = [target_info for target_info in targets_info if target_info.uuid == metadata['_target_info']]
        if len(sel_target_info) == 0:
            raise AbstractImportanceException('Required target_info is absent')
        elif len(sel_target_info) > 1:
            raise AbstractImportanceException('Identical targets info has been found')

        target_info = sel_target_info[0]
        if metadata['_features_generator'] != features_generator.name:
            raise AbstractImportanceException('feature generator name mismatch')

        importance = cls(
            metadata['_name'],
            metadata['_folder_path'],
            target_info,
            targets_generator,
            features_generator,
            metadata['_split_name'],
        )

        importance.fill_fields(metadata)
        importance.read_log()

        return importance

    @abc.abstractmethod
    def _is_reverse_rank(self):
        pass

    def get_norm_ranks(self):
        ranks = np.array(self.ranks)

        if self._is_reverse_rank():
            ranks = np.nanmax(ranks) - ranks
        ranks[np.isnan(ranks)] = 0

        # activation
        # ranks = ranks**0.3
        # activation = lambda x: 0.5*(np.tanh(4*(2*x-1))+1)
        # ranks = activation(ranks)
        # ranks[ranks <= activation(ranks)+1e-6] = 0.

        if ranks.sum() != 0:
            ranks = ranks / ranks.sum()
        return list(ranks)

    def get_relevant_features(self):
        ranks_normed = np.array(self.get_norm_ranks())
        sorti = ranks_normed.argsort()
        return [self._features[i] for i in sorti if ranks_normed[i] > 0.][::-1]

    @property
    def ranks(self):
        return self._ranks

    @property
    def name(self):
        return self._name

    @property
    @abc.abstractmethod
    def logger(self):
        return self._logger

    @abc.abstractmethod
    def calculate_features_importance(self):
        pass

    @abc.abstractmethod
    def read_log(self):
        pass

    def create_train_test_data(self):
        splitted_data_x = self._features_generator.get_splitted_data(
            self._split_name,
            idata_name=self._idata_name
        )
        splitted_data_y = self._targets_generator.get_splitted_data(
            self._split_name,
            idata_name=self._idata_name
        )

        if TTV.TRAIN not in splitted_data_x.keys():
            raise AbstractImportanceException('No train data provided.')

        x_train = splitted_data_x[TTV.TRAIN]
        y_train = splitted_data_y[TTV.TRAIN]

        if TTV.VALIDATION not in splitted_data_x.keys():
            warnings.warn('No Validation data provided. Running on Test data.')
            x_test = splitted_data_x[TTV.TEST]
            y_test = splitted_data_y[TTV.TEST]
        elif TTV.TEST not in splitted_data_x.keys():
            x_test = splitted_data_x[TTV.TRAIN]
            y_test = splitted_data_y[TTV.TRAIN]
        else:
            x_test = splitted_data_x[TTV.VALIDATION]
            y_test = splitted_data_y[TTV.VALIDATION]

        Verbose.instance.print(1, f'Train length={len(x_train)}, test length={len(x_test)}')

        return x_train, y_train, x_test, y_test
