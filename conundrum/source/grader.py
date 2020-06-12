import datetime
import os

from src.concore.data_preparation.preprocessing.workflow import CreationWorkflow
from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.feature_generator.featuregeneratormanager import FeatureGeneratorManager
from src.concore.grader.importances.boruta_importance import BorutaImportance
from src.concore.grader.importances.load_importance import load_importance
from src.concore.grader.importances.random_subsampling_averaged_importance import RandomSubsamplingAveragedImportance
from src.concore.grader.importances.random_subsampling_score_importance import RandomSubsamplingScoreImportance
from src.concore.grader.importances.random_subsampling_shap_importance import SHAPImportance
from src.concore.grader.importances.sklearn_importance import SklearnImportance
from src.concore.grader.importances.tsfresh_importance import TsfreshImportance
from src.concore.grader.selectors.quality_selector import QualitySelector
from src.concore.grader.selectors.simple_selector import SimpleSelector
from src.concore.i_utils.metadatakeeper import MetaDataKeeper
from src.concore.i_utils.metriclogger import MetricLogger
from src.concore.i_utils.verbose import Verbose
from src.concore.tdata.targetinfo import TargetInfo


class GraderException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_targets_info': lambda x: [info.uuid for info in x],
    '_targets_generator': lambda x: x.name,
    '_features_generator': lambda x: x.name,
    '_logger': lambda x: None,
    '_importances': lambda x: {name: x[name].metadata for name in x.keys()},
    '_verbose': lambda x: None,
}


class Grader(MetaDataKeeper):

    IMPORTANCE_TYPES = {
        'boruta': BorutaImportance,
        'averaged_importance': RandomSubsamplingAveragedImportance,
        'averaged_score': RandomSubsamplingScoreImportance,
        'tsfresh': TsfreshImportance,
        'sklearn': SklearnImportance,
        'shap': SHAPImportance
    }

    def __init__(
            self,
            name: str,
            targets_info: list,
            targets_generator: FeatureGenerator,
            features_generator: FeatureGenerator,
            split_name: str,
            folder_path: str,
    ):
        self._name = name
        self._targets_info = targets_info
        self._targets_generator = targets_generator
        self._features_generator = features_generator
        self._split_name = split_name
        self._folder_path = folder_path
        self._importances = {}
        self._logger = None
        self._selected_features = None
        self._selectors = {}

        self._log_path = os.path.join(self._folder_path, 'grader.log')

        MetaDataKeeper.__init__(self, EXCEPTIONAL_FIELDS)

    @classmethod
    def from_metadata(
            cls,
            metadata: dict,
            targets_info: list,
            feature_generator_manager: FeatureGeneratorManager,
    ):
        for target_info, target_info_id in zip(targets_info, metadata['_targets_info']):
            if target_info.uuid != target_info_id:
                raise GraderException('TargetInfo ID mismatch')
        features_generator = feature_generator_manager.get(metadata['_features_generator'])

        grader = cls(
            metadata['_name'],
            targets_info,
            feature_generator_manager.targets_generator,
            features_generator,
            metadata['_split_name'],
            metadata['_folder_path']
        )
        grader.fill_fields(metadata)
        for importance in metadata['_importances'].values():
            grader._importances[importance['_name']] = load_importance(
                importance,
                targets_info,
                feature_generator_manager.targets_generator,
                features_generator,
            )

        return grader

    @property
    def name(self):
        return self._name

    @property
    def logger(self):
        if not self._logger:
            self._logger = MetricLogger(
                self.name,
                self._log_path,
                values=["date", "grader_name", "feature_generator_name",
                        "test_column_name",
                        # "selected_features"
                        ],
                mode='a'
            )
        return self._logger

    def run(self, importance_type: str, **kwargs):
        """
        Creating and running given importance type
        :param importance_type: key from IMPORTANCE_TYPES
        :param kwargs: arguments for selected importance
        :return:
        """
        for target_info in self._targets_info:
            importance_name = f'{self.name}_{importance_type}_{target_info.generated_name[0]}'
            self.add_importance(importance_name, importance_type, target_info, **kwargs)
            Verbose.instance.print(1, f'started importance {importance_name}')
            self.run_importance(importance_name)
            Verbose.instance.print(1, f'finished importance {importance_name}')

    def add_importance(
            self,
            importance_name: str,
            importance_type: str,
            target_info: TargetInfo,
            **kwargs
    ):
        """
        Creating importance by given type and adding it to _importances
        :param importance_name: name of importance to create
        :param importance_type: key from IMPORTANCE_TYPES
        :param target_info: target to run importance
        :param kwargs: arguments for selected importance
        :return:
        """

        if len(target_info.generated_name) > 1:
            raise GraderException('Multiple features are generated by target modifier.')

        if importance_name in self._importances.keys():
            raise GraderException(f'Given importance name {importance_name} '
                                  f'is already presented  in grander {self.name}')

        importance_class = Grader.IMPORTANCE_TYPES[importance_type]

        self._importances[importance_name] = importance_class(
            importance_name,
            os.path.join(self._folder_path, importance_name),
            target_info,
            self._targets_generator,
            self._features_generator,
            self._split_name,
            **kwargs
        )

    def run_importance(self, importance_name: str):
        """
        Running importance by given name
        :param importance_name:
        :return:
        """
        self._importances[importance_name].calculate_features_importance()
        self.logger.log({
            "date": datetime.datetime.now(),
            "grader_name": self._name,
            "feature_generator_name": self._features_generator.name,
            "test_column_name": self._split_name,
            # "selected_features": json.dumps(self._importances[importance_name].selected_features)
        })

    def run_all_importances(self):
        """
        running all importances in this object
        :return:
        """
        for importance_name in self._importances.keys():
            self.run_importance(importance_name)

    def get_selected_features(self,
                              importances_names: list = None,
                              method="unique",
                              features_amount=5,
                              random_sample_len=None,
                              volume_info=1.0,
                              selector_name=None,
                              **selector_args
                              ) -> FeatureGenerator:
        """

        :param importances_names:
        :param method:
        :param features_amount:
        :param random_sample_len:
        :param volume_info:
        :param selector_name:
        :return:
        """
        if not selector_name:
            selector_name = method+'_selector'

        if selector_name in self._selectors.keys():
            raise GraderException(f'Given selector name {selector_name} is already exists in grander {self.name}')

        Verbose.instance.print(1, f'Selecting {features_amount} features')
        if importances_names is None:
            importances_names = list(self._importances.keys())

        importances = dict.fromkeys(importances_names)
        for name in importances:
            importances[name] = self._importances[name]

        for name, imp in self._importances.items():
            Verbose.instance.print(1, f'{name} has {len(imp.get_relevant_features())} relevant features')

        if method == 'quality':
            self._selectors[selector_name] =\
                QualitySelector(selector_name, self._targets_info, self._targets_generator, self._features_generator,
                                self._split_name, importances, volume_info, features_amount, **selector_args)

        elif method in ['unique', 'match']:
            self._selectors[selector_name] =\
                SimpleSelector(selector_name, self._targets_info, self._targets_generator, self._features_generator,
                               self._split_name, importances, method, features_amount, random_sample_len, **selector_args)

        else:
            raise GraderException(f'Unknown features selection method {method} given')

        selected_features = self._selectors[selector_name].select_features()
        Verbose.instance.print(1, f'total selected: {len(selected_features)}')
        selected_workflow = self._create_workflow(selected_features)

        out = FeatureGenerator(
            f'{self._features_generator.name}_{self.name}_{random_sample_len}',
            self._features_generator.idata_container,
            self._features_generator.global_data_container,
        )
        out.workflow = selected_workflow

        return out

    def _create_workflow(self, selected_features) -> CreationWorkflow:
        """
        Splitting modifiers in _feature_generator into individual ones
        :param selected_features: list of feature names to select from _feature_generator
        :return: Workflow with selected features creation
        """
        blocks = []
        individual_workflows = self._features_generator.workflow.get_individual_cwf()
        for feature_name in selected_features:
            if feature_name in individual_workflows.keys():
                blocks.extend(individual_workflows[feature_name].workflow)

        workflow = CreationWorkflow(blocks)

        union_workflow = CreationWorkflow.get_union_cwf([workflow])

        return union_workflow
