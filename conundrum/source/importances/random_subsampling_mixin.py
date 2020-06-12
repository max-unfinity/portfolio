import abc

import numpy as np
from tqdm import tqdm

from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.abstract_importance import AbstractImportance, AbstractImportanceException
from src.concore.i_utils.logutils import Logger
from src.concore.i_utils.verbose import Verbose
from src.concore.model.parameters.abstract_parameters import AbstractParameters
from src.concore.model.parameters.sklearn_parameters import SKLearnParameters
from src.concore.tdata.targetinfo import TargetInfo

logger = Logger.instance(__name__)


class RandomSubsamplingAveragedImportanceException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_modifiers': lambda x: None,
}


class RandomSubsamplingMixin(AbstractImportance):
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
                 features_amount=5,
                 random_sample_len=None
                 ):
        """

        :param name: name of the importance
        :param folder_path: path lo save log results
        :param target_info: target to work with
        :param features_generator: feature generator with features to analyze
        :param split_name: name of the train test split
        :param core_parameters: parameters of the given ml algorithm
        :param n_estimations: amount of estimations to perform
        :param subsample_len: length of the features subsample to run core
        :param features_amount: amount of features to select
        :param random_sample_len: top size of subsample to select random features from
        """

        self._subsample_len = subsample_len
        self._features_amount = features_amount
        self._random_sample_len = random_sample_len
        if not self._random_sample_len:
            self._random_sample_len = self._features_amount
        if self._random_sample_len < self._features_amount:
            raise AbstractImportanceException('Random sample cannot be < than features amount')
        self._n_estimations = n_estimations
        self._core_parameters = core_parameters or SKLearnParameters(core_type='xgb')
        self._feature_scores = None
        self._workflows = None

        AbstractImportance.__init__(
            self,
            name,
            folder_path,
            target_info,
            targets_generator,
            features_generator,
            split_name,
        )
        self.add_fields(EXCEPTIONAL_FIELDS)

    @property
    def feature_scores(self):
        # TODO: ? May be all feats must has score. If score undefined then np.nan
        return self._feature_scores

    @abc.abstractmethod
    def read_log(self):
        pass

    @abc.abstractmethod
    def _step(self, seed: int):
        pass

    @abc.abstractmethod
    def _is_reverse_rank(self):
        pass

    def calculate_features_importance(self):
        """
        Calculating feature importances for subsamples
        :return:
        """
        individual_workflows = self._features_generator.workflow.get_individual_cwf()
        self._workflows = list(individual_workflows.values())

        Verbose.instance.print(1, f'Calculating importance for {len(self._workflows)} features')

        # TODO: seed may repeat
        # rnd_seed = np.random.randint(0, self._n_estimations, self._n_estimations)
        rnd_seed = range(self._n_estimations)

        for seed in tqdm(rnd_seed, desc=self.__class__.__name__, disable=not bool(Verbose.instance.verbose_level)):
            try:
                self._step(seed)
            except ValueError as e:
                logger.warning(f'error with seed {seed}: {e}')

        self.read_log()

        scores_all_features = dict.fromkeys(self._features, np.nan)
        scores_all_features.update(self.feature_scores)
        self._ranks = list(scores_all_features.values())

        Verbose.instance.print(1, 'Calculation complete')
