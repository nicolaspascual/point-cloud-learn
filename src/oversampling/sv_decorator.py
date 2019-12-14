from imblearn.base import BaseSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring

from smote_variants import G_SMOTE


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class G_SMOTEDecorator(BaseSampler):
    """Perform over-sampling using SV sampling
    approach for imbalanced datasets.
    """
    _sampling_type = 'over-sampling'
    oversampler = G_SMOTE()

    def _fit_resample(self, X, y):
        return self.oversampler.sample(X, y)
