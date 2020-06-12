import random

from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.selectors.abstract_selector import AbstractSelector, SelectorException


class SimpleSelector(AbstractSelector):
    def __init__(self,
                 name: str,
                 targets_info: list,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 importances: dict,
                 method: str,
                 features_amount,
                 random_sample_len=None,
                 seed=None
                 ):
        AbstractSelector.__init__(self,
                                  name,
                                  targets_info,
                                  targets_generator,
                                  features_generator,
                                  split_name,
                                  importances,
                                  features_amount,
                                  random_sample_len,
                                  seed
                                  )
        self.method = method

    def select_features(self):
            features = set()
            random.seed(self._seed)
            for name, importance in self._importances.items():
                imp_features = importance.get_relevant_features()[:self._random_sample_len]
                imp_features = set(random.sample(imp_features, min(len(imp_features), self._features_amount)))
                if self.method == "unique":
                    features = features | imp_features
                elif self.method == "match":
                    if len(features) > 0:
                        features = features & imp_features
                    else:
                        features = imp_features
                else:
                    raise SelectorException(f'Unknown features selection method {self.method} given')

            self.selected_features = list(features)
            return self.selected_features
