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
    assert len(r) == 2
    assert r[0].shape == (9801, 4, 128, 128)
    assert r[1].shape == (9801, 5, 128, 128)


def test_get_test_patches60(dset_10, dset_20, dset_60):
    r = patches.get_test_patches60(dset_10, dset_20, dset_60, 192, 12)
    assert len(r) == 3
    assert r[0].shape == (4356, 4, 192, 192)
    assert r[1].shape == (4356, 5, 192, 192)
    assert r[2].shape == (4356, 3, 192, 192)


def test_save_random_patches():
    pass


def test_save_random_patches60():
    pass
