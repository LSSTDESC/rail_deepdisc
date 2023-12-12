import numpy as np
from deepdisc.preprocessing.ground_truth_gen import mad_wavelet_own
from numpy.testing import assert_almost_equal


def test_deepdisc_installed():
    """Simple test that can be removed later. Just guarantees that DeepDISC
    and it's dependencies have been correctly installed."""

    np.random.seed(2023)
    image = np.random.random((100, 100))
    res = mad_wavelet_own(image)
    assert_almost_equal(res, 0.342808, decimal=5)
