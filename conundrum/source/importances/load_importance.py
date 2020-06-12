from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.importances.boruta_importance import BorutaImportance
from src.concore.grader.importances.random_subsampling_averaged_importance import RandomSubsamplingAveragedImportance
from src.concore.grader.importances.random_subsampling_score_importance import RandomSubsamplingScoreImportance
from src.concore.grader.importances.random_subsampling_shap_importance import SHAPImportance
from src.concore.grader.importances.sklearn_importance import SklearnImportance
from src.concore.grader.importances.tsfresh_importance import TsfreshImportance

IMPORTANCES = {
    'BorutaImportance': BorutaImportance,
    'RandomSubsamplingAveragedImportance': RandomSubsamplingAveragedImportance,
    'RandomSubsamplingScoreImportance': RandomSubsamplingScoreImportance,
    'TsfreshImportance': TsfreshImportance,
    'SklearnImportance': SklearnImportance,
    'SHAPImportance': SHAPImportance
}


def load_importance(
        metadata: dict,
        targets_info: list,
        targets_generator: FeatureGenerator,
        feature_generator: FeatureGenerator,
        ):
    cls = IMPORTANCES[metadata['_class_name']]
    return cls.from_metadata(metadata, targets_info, targets_generator, feature_generator)
