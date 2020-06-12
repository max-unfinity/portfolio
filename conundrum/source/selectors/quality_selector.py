import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.concore.data_preparation.preprocessing.workflow import CreationWorkflow
from src.concore.feature_generator.featuregenerator import FeatureGenerator
from src.concore.grader.selectors.abstract_selector import SelectorException, AbstractSelector
from src.concore.i_utils.metrics.metrics import all_metrics
from src.concore.i_utils.verbose import Verbose
from src.concore.imodel.sklearn_imodel import SKlearnIModel, SKLearnParameters


class QualitySelector(AbstractSelector):

    def __init__(self,
                 name: str,
                 targets_info: list,
                 targets_generator: FeatureGenerator,
                 features_generator: FeatureGenerator,
                 split_name: str,
                 importances: dict,
                 volume_info=1.0,
                 max_features=None,
                 metric_name='rmse',
                 model=SKlearnIModel,
                 model_parameters=None,
                 ):

        AbstractSelector.__init__(self,
                                  name,
                                  targets_info,
                                  targets_generator,
                                  features_generator,
                                  split_name,
                                  importances,
                                  max_features
                                  )

        if model_parameters is None:
            if model is not SKlearnIModel:
                raise SelectorException(f'give model_parameters to selector')
            model_parameters = SKLearnParameters(core_type='xgb', initial_parameters={})

        if len(self._targets_info) > 1:
            raise SelectorException(f'Multitarget not supported')
        else:
            self._target_info = self._targets_info[0]

        self._model = model
        self._model_parameters = model_parameters
        self._features = self._features_generator.workflow.get_features_names()
        self._raw_score = None
        self._not_fitted_imps = dict(**self._importances)

        self.metric_name = metric_name
        self.volume_info = volume_info
        self.max_features = max_features or len(self._features)

        self.ranks_final = None
        self.qualities = {}

    def select_features(self):
        self.calculate_qualities()
        self.selected_features = self.select_by_ranks(self.ranks_final, self.volume_info, self.max_features)
        # print(f'{len(self.selected_features)} features: {self.test_selected_features(self.selected_features)} {self.metric_name}')
        # self.plot_ranks()
        return list(self.selected_features)

    def calculate_qualities(self):
        if self._raw_score is None:
            Verbose.instance.print(1, f'Fitting Raw data on {len(self._features)} features...')
            self._raw_score = self.test_selected_features(self._features)
            Verbose.instance.print(1, f'Raw score: {self._raw_score} {self.metric_name}\n')

        importance_scores = {}
        for name, importance in list(self._not_fitted_imps.items()):
            selected = self.select_by_ranks(
                importance.get_norm_ranks(), 1.0, min(self.max_features, self.volume_info*len(self._features)))
            Verbose.instance.print(1, f'Fitting {name} on {len(selected)} features...')
            importance_scores[name] = self.test_selected_features(selected)

            if all_metrics[self.metric_name].minimize:
                q = self._raw_score / importance_scores[name]
            else:
                q = importance_scores[name] / self._raw_score
            self.qualities[name] = q

        if len(self.qualities) > 1:
            # standardize qualities
            q = pd.Series(self.qualities)
            q -= 1
            q /= q.std()**0.5
            q += 1
            self.qualities = q.to_dict()

        importance_ranks_table = pd.DataFrame(index=self._features)
        for name, importance in list(self._not_fitted_imps.items()):
            importance_ranks_table[name] = np.array(importance.get_norm_ranks()) * self.qualities[name]
            self._not_fitted_imps.pop(name)
            Verbose.instance.print(1, f'{name} score: {importance_scores[name]} {self.metric_name}, quality: {self.qualities[name]}')

        self.ranks_final = importance_ranks_table.sum(axis=1)  # ranks_final is pd.Series
        self.ranks_final[self.ranks_final < 0] = 0
        self.ranks_final = self.ranks_final / self.ranks_final.sum()

    def select_by_ranks(self, ranks, volume_info, max_features=None, threshold=0.0):
        if volume_info >= 1.0:
            volume_info = 1.0 - 1e-6  # because of float precision

        ranks = pd.Series(ranks, self._features)
        ranks_sorted = ranks.sort_values(ascending=False)

        if max_features is None or max_features == -1:
            max_features = len(ranks_sorted)

        summ = 0
        feature = ranks_sorted.index[0]
        for feature, rank in ranks_sorted.iteritems():
            summ += rank
            if summ > volume_info or rank < threshold:
                break
        selected_features = ranks_sorted[:feature].index

        return list(selected_features[:int(max_features)])

    def test_selected_features(self, features, model_parameters=None):
        selected_fg = self._create_selected_fg(features)
        params = model_parameters or self._model_parameters
        core = self._model(
            name='tmp_quality_model',
            parameters=params,
            split_name=self._split_name,
            targets_info=self._targets_info,
            targets_generator=self._targets_generator,
            feature_generator=selected_fg,
            path_model_out='tmp'
        )
        core.init_model()
        core.run()  # TODO: need to pass any idata?
        # TODO: deal with metrics
        return core.metrics['test_'+self.metric_name+'_target_0']

    def plot_importance_ranks(self, n=20, figsize=(8, 5)):

        score_table = pd.DataFrame()
        for name, importance in list(self._importances.items()):
            score_table[name] = pd.Series(importance.get_norm_ranks(), self._features)

        score_table = score_table.sum(axis=1)
        feats = list(score_table.sort_values(ascending=False).index)[:n]

        i = 0
        for name, importance in list(self._importances.items()):
            plt.figure(figsize=figsize)
            ranks = pd.Series(importance.get_norm_ranks(), self._features)
            ranks = ranks.reindex(feats)
            plt.barh(feats, ranks.values[:n], color='C' + str(i))
            plt.title(name)
            plt.tight_layout()
            i += 1

        plt.figure(figsize=figsize)
        bot = pd.Series(0, feats)
        for name, importance in list(self._importances.items()):
            ranks = pd.Series(importance.get_norm_ranks(), self._features)
            ranks = ranks.reindex(feats)
            plt.barh(feats, ranks.values[:n], left=bot.values)
            bot += ranks
        plt.legend(self._importances.keys())
        plt.plot(bot.values, bot.index)
        plt.title('Ranks summary')
        plt.tight_layout()

        plt.show()

    def plot_ranks(self, figsize=(8, 5)):
        
        if self.ranks_final is None:
            Verbose.instance.print(1, f'Can not plot ranks. Calculate it before')
            return
            
        feats = list(self.ranks_final.sort_values(ascending=False).index)

        i = 0
        for name, importance in list(self._importances.items()):
            plt.figure(figsize=figsize)
            ranks = pd.Series(importance.get_norm_ranks(), self._features)
            ranks = ranks.reindex(feats)
            plt.bar(range(len(ranks)), ranks.values, 1.0, color='C'+str(i))
            plt.title(name)
            plt.tight_layout()
            i += 1

        # plt.grid(linestyle='--', c='0.7')
        table = pd.DataFrame()
        for name, importance in list(self._importances.items()):
            ranks = pd.Series(importance.get_norm_ranks(), self._features)
            ranks = ranks.reindex(feats) * self.qualities[name]
            table[name] = ranks
        table.plot.area(figsize=figsize, stacked=True)
        plt.title('Ranks summary')
        plt.tight_layout()

        plt.figure(figsize=figsize)
        plt.plot(self.ranks_final.sort_values(ascending=False).values)
        plt.title('Final ranks')
        plt.tight_layout()

        plt.show()

    def plot_top(self, n=20, figsize=(8, 5)):
        if self.ranks_final is None:
            Verbose.instance.print(1, f'Can not plot ranks. Calculate it before')
            return

        feats = list(self.ranks_final.sort_values(ascending=False).index)

        table = pd.DataFrame()
        for name, importance in list(self._importances.items()):
            ranks = pd.Series(importance.get_norm_ranks(), self._features)
            ranks = ranks.reindex(feats) * self.qualities[name]
            table[name] = ranks
        table.iloc[:n].plot.barh(figsize=figsize, stacked=True)
        plt.title('Ranks summary')
        plt.tight_layout()
            
    def _create_selected_fg(self, features):
        blocks = []
        individual_workflows = self._features_generator.workflow.get_individual_cwf()
        for feature_name in features:
            if feature_name in individual_workflows.keys():
                blocks.extend(individual_workflows[feature_name].workflow)
        workflow = CreationWorkflow(blocks)

        out = FeatureGenerator(
            f'{self._features_generator.name}_{self.__class__}_{len(features)}_features',
            self._features_generator.idata_container,
            self._features_generator.global_data_container,
        )
        out.workflow = workflow

        return out
