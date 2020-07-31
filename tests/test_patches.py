import pytest

import numpy as np

from utils import patches


@pytest.fixture()
def dset_10():
    return np.random.randint(low=0, high=8, size=(4, 10980, 10980), dtype=np.int8)


@pytest.fixture()
def dset_20():
    return np.random.randint(low=0, high=8, size=(5, 5490, 5490), dtype=np.int8)


@pytest.fixture()
def dset_60():
    return np.random.randint(low=0, high=8, size=(3, 1830, 1830), dtype=np.int8)


def test_get_test_patches(dset_10, dset_20):
    r = patches.get_test_patches(dset_10, dset_20)
    print(r)
    assert False


def test_get_test_patches60(dset_10, dset_20, dset_60):
    r = patches.get_test_patches60(dset_10, dset_20, dset_60)
    assert r


def test_save_random_patches():
    pass


def test_save_random_patches60():
    pass
