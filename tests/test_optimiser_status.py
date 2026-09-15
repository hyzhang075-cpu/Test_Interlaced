from types import MethodType

import numpy as np

from optimiser import Optimiser


def _optimizer_returning(loss):
    optim = Optimiser.__new__(Optimiser)
    optim.direct = 240
    optim.max_iter = 1
    optim.min_loss_diff = 1e-4
    optim.slope_upper_bound = 50.0
    optim.weights = {240: np.ones(2)}
    optim.slopes = {240: np.ones((2, 1))}

    def fake_iter(self, **kwargs):
        self.weights[self.direct] = np.full(2, 2.0)
        return loss, 1.0

    optim._optim_coeff_iter = MethodType(fake_iter, optim)
    return optim


def test_finite_worse_candidate_is_no_improvement_and_rolls_back():
    optim = _optimizer_returning(1.1)
    final_loss, status, start_loss = optim._optim_one_coeff_until_stop(
        'weights', None, options=None, verbose=False,
    )
    assert (final_loss, status, start_loss) == (1.0, 'no_improvement', 1.0)
    assert optim.weights[240].tolist() == [1.0, 1.0]


def test_nonfinite_candidate_is_divergence_and_rolls_back():
    optim = _optimizer_returning(np.nan)
    final_loss, status, start_loss = optim._optim_one_coeff_until_stop(
        'weights', None, options=None, verbose=False,
    )
    assert (final_loss, status, start_loss) == (1.0, 'diverge', 1.0)
    assert optim.weights[240].tolist() == [1.0, 1.0]
