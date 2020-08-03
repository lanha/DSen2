import pytest

import numpy as np

from utils import patches


@pytest.fixture()
def dset_10():
    return np.ones((10980, 10980, 4))


@pytest.fixture()
def dset_20():
    return np.ones((5490, 5490, 5))


@pytest.fixture()
def dset_60():
    return np.ones((1830, 1830, 3))


def test_get_test_patches(dset_10, dset_20):
    r = patches.get_test_patches(dset_10, dset_20, 128, 8)
    assert r


def test_get_test_patches60(dset_10, dset_20, dset_60):
    r = patches.get_test_patches60(dset_10, dset_20, dset_60, 192, 12)
    assert r


def test_save_random_patches():
    pass


def test_save_random_patches60():
    pass
