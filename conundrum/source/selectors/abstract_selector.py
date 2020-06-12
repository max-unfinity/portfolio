import abc

from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.i_utils.metadatakeeper import MetaDataKeeper


class SelectorException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_targets_info': lambda x: [info.uuid for info in x],
    '_targets_generator': lambda x: x.name,
    '_features_generator': lambda x: x.name,
    '_logger': lambda x: None,
    '_importances': lambda x: {name: x[name].metadata for name in x.keys()},
}


class AbstractSelector(abc.ABC, MetaDataKeeper):
    def __init__(self,
                 name:str,
                 targets_info: list,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 importances: dict,
                 features_amount,
                 random_sample_len=None,
                 seed=None
                 ):

        self._name = name
        self._targets_info = targets_info
        self._targets_generator = targets_generator
        self._features_generator = features_generator
        self._split_name = split_name
        self._importances = importances
        self._seed = seed

        if features_amount is None:
            features_amount = len(self._features_generator.workflow.get_features_names())

        if random_sample_len is None:
            random_sample_len = features_amount

        if random_sample_len < features_amount:
            raise SelectorException('Random sample cannot be < than features amount')

        self._features_amount = features_amount
        self._random_sample_len = random_sample_len

        self.selected_features = None

        MetaDataKeeper.__init__(self, EXCEPTIONAL_FIELDS)

    @property
    def name(self):
        return self._name

    @abc.abstractmethod
    def select_features(self):
        pass
