import os

from src.concore.feature_generator.featuregeneratormanager import FeatureGeneratorManager
from src.concore.grader.grader import Grader
from src.concore.i_utils.metadatakeeper import MetaDataKeeper
from src.concore.i_utils.verbose import Verbose


class GraderManagerException(Exception):
    def __init__(self, error_string):
        Exception.__init__(self, error_string)


EXCEPTIONAL_FIELDS = {
    '_graders': lambda x: {name: x[name].metadata for name in x.keys()},
    '_feature_generator_manager': lambda x: None,
    '_targets_info': lambda x: [info.uuid for info in x],

}


class GraderManager(MetaDataKeeper):
    def __init__(self,
                 feature_generator_manager: FeatureGeneratorManager,
                 folder_path: str,
                 ):

        self._graders = {}
        self._folder_path = folder_path
        self._feature_generator_manager = feature_generator_manager
        self._targets_info = feature_generator_manager.targets_info

        MetaDataKeeper.__init__(self, EXCEPTIONAL_FIELDS)

    @classmethod
    def from_metadata(
            cls,
            metadata: dict,
            feature_generator_manager: FeatureGeneratorManager,
    ):
        """

        :param metadata: Grader manager metadata
        :param feature_generator_manager:
        :return:
        """
        grader_manager = cls(
            feature_generator_manager,
            metadata['_folder_path']
        )
        grader_manager.fill_fields(metadata)
        for grader_metadata in metadata['_graders'].values():
            grader_manager._graders[grader_metadata['_name']] = Grader.from_metadata(
                grader_metadata,
                feature_generator_manager.targets_info,
                feature_generator_manager,
            )
        return grader_manager

    @property
    def graders(self):
        return self._graders

    def create_grader(self, grader_name, feature_generator_name, split_name):
        """
            Add already created grader to storage. Raise exception if grader with the same name already in storage
            :param grader_name: name of the grader to create
            :param feature_generator_name: name of the feature generator to work with
            :param split_name: split name to run models
            :return:
        """
        if grader_name in self._graders.keys():
            raise GraderManagerException(f'Grader with name {grader_name} is already exist')

        targets_generator = self._feature_generator_manager.targets_generator
        features_generator = self._feature_generator_manager.get(feature_generator_name)

        grader_folder = os.path.join(self._folder_path, f'grader_{grader_name}')
        # TODO check if okay to exist
        os.makedirs(grader_folder, exist_ok=True)
        self._graders[grader_name] = Grader(
            grader_name,
            self._targets_info,
            targets_generator,
            features_generator,
            split_name,
            grader_folder,
        )
        Verbose.instance.print(1, f'Created Grader {grader_name}')
