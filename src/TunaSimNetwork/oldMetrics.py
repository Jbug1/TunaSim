# old similarity measures
import scipy
import pandas as pd
import numpy as np
from numba import njit
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score


class oldMetricEvaluator:

    def __init__(self,
                 dataset,
                 groupby_columns,
                 intermediates_path,
                 performance_path):
        
        self.dataset = dataset
        self.groupby_columns = groupby_columns
        self.intermediates_path = intermediates_path
        self.performance_path = performance_path

        self.metrics = [oldMetricEvaluator.entropy_similarity, 
                    oldMetricEvaluator.dot_product_similarity,
                    oldMetricEvaluator.harmonic_mean_similarity,
                    oldMetricEvaluator.probabilistic_symmetric_chi_squared_similarity,
                    oldMetricEvaluator.lorentzian_similarity,
                    oldMetricEvaluator.matusita_similarity,
                    oldMetricEvaluator.harmonic_mean_similarity,
                    oldMetricEvaluator.fidelity_similarity]
        
        self.names = ['entropy',
                'dot_product',
                'harmonic_mean',
                'chi_squared',
                'lorentzian',
                'matusita',
                'harmonic_mean',
                'fidelity']

    @staticmethod
    @njit
    def _weight_intensity_by_entropy(x, entropy_x):

        WEIGHT_START = 0.25
        ENTROPY_CUTOFF = 3
        weight_slope = (1 - WEIGHT_START) / ENTROPY_CUTOFF

        weight = min(WEIGHT_START + weight_slope * entropy_x, 1)
        x = np.power(x, weight)
        x = x / np.sum(x)

        return x

    @staticmethod
    @njit
    def harmonic_mean_similarity(p, q):
        r"""
        Harmonic mean distance:

        .. math::

            1-2\sum(\frac{P_{i}Q_{i}}{P_{i}+Q_{i}})
        """

        return 1 - 2 * np.sum(p * q / (p + q))

    @staticmethod
    @njit
    def lorentzian_similarity(p, q):
        r"""
        Lorentzian distance:

        .. math::

            \sum{\ln(1+|P_i-Q_i|)}
        """

        return 1 - np.sum(np.log(1 + np.abs(p - q)))

    @staticmethod
    @njit
    def matusita_similarity(p, q):
        r"""
        Matusita distance:

        .. math::

            \sqrt{\sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2}
        """
        return 1 - np.sqrt(np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2))) / np.sqrt(2)

    @staticmethod
    @njit
    def probabilistic_symmetric_chi_squared_similarity(p, q):
        r"""
        Probabilistic symmetric χ2 distance:

        .. math::

            \frac{1}{2} \times \sum\frac{(P_{i}-Q_{i}\ )^2}{P_{i}+Q_{i}\ }
        """
    
        return 1 - (1 / 2 * np.sum(np.power(p - q, 2) / (p + q)))

    @staticmethod
    def entropy_similarity(p, q):
        r"""
        Unweighted entropy distance:

        .. math::

            -\frac{2\times S_{PQ}-S_P-S_Q} {ln(4)}, S_I=\sum_{i} {I_i ln(I_i)}
        """
    
        merged = p + q
        entropy_increase = 2 * \
                        scipy.stats.entropy(merged) - scipy.stats.entropy(p) - \
                        scipy.stats.entropy(q)
        
        return 1 - min(1, entropy_increase / np.log(4))

    @staticmethod
    @njit
    def dot_product_similarity(p, q):
        r"""
        Dot product distance:

        .. math::

            1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
        """
        
        score = np.power(np.sum(q * p), 2) / (
            np.sum(np.power(q, 2)) * np.sum(np.power(p, 2))
        )
        return 1 - np.sqrt(score)

    @staticmethod
    @njit
    def fidelity_similarity(p, q):
        r"""
        Fidelity distance:

        .. math::

            1-\sum\sqrt{P_{i}Q_{i}}
        """
        return np.sum(np.sqrt(p * q))

    def evaluate_and_write_results(self):
        """ 
        Evaluates this set of old measures for a collection of queries and targets
        """

        #get unweighted scores
        results, performance = self.get_evals()
        results.to_csv(f'{self.intermediates_path}/old_metrics_unweighted.csv')
        performance.to_csv(f'{self.performance_path}/old_metrics_unweighted.csv')

        #get unweighted scores
        results, performance = self.get_evals(reweighted = True)
        results.to_csv(f'{self.intermediates_path}/old_metrics_weighted.csv')
        performance.to_csv(f'{self.performance_path}/old_metrics_weighted.csv')

        
    def get_evals(self, reweighted = False):

        queries = self.dataset['query'].to_numpy()
        targets = self.dataset['target'].to_numpy()

        if reweighted:

            queries = [oldMetricEvaluator._weight_intensity_by_entropy(i, scipy.stats.entropy(i)) for i in queries]
            targets = [oldMetricEvaluator._weight_intensity_by_entropy(i, scipy.stats.entropy(i)) for i in targets]

        #get weighted scores
        results = list()

        for metric, name in zip(self.metrics, self.names):

            scores = list()
            for query, target in zip(queries, targets):

                scores.append(metric(query, target))
        
            self.dataset[name] = scores
        
            #get performance
            performance = self.dataset.groupby(self.groupby_columns).max()

        if reweighted:
            results.append((name, roc_auc_score(performance['score'], performance[name+'_reweighted'])))
        else:
            results.append((name, roc_auc_score(performance['score'], performance[name])))
        
        return pd.DataFrame(performance, columns = ['name', 'performance'])