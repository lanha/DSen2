from pathlib import Path
import tempfile
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


@pytest.fixture()
def scale_20():
    return 2


@pytest.fixture()
def scale_60():
    return 6


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


def save_test_patches(dset_10, dset_20):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_test_patches(dset_10, dset_20, tmpdir, 128, 8)
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 2


def save_test_patches60(dset_10, dset_20, dset_60):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_test_patches60(dset_10, dset_20, dset_60, tmpdir, 192, 12)
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 3


def test_save_random_patches(dset_10, dset_20, scale_20):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_random_patches(
            dset_20,
            patches.downPixelAggr(dset_10, scale_20),
            patches.downPixelAggr(dset_20, scale_20),
            tmpdir,
        )
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 2


def test_save_random_patches60(dset_10, dset_20, dset_60, scale_60):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_random_patches60(
            dset_60,
            patches.downPixelAggr(dset_10, scale_60),
            patches.downPixelAggr(dset_20, scale_60),
            patches.downPixelAggr(dset_60, scale_60),
            tmpdir,
        )
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 3


def test_downPixelAggr(dset_10, scale_20):
    r = patches.downPixelAggr(dset_10, scale_20)
    assert r.shape == (1, 1, 1)


def test_recompose_images(dset_10, dset_20):
    p = patches.get_test_patches(dset_10, dset_20, 128, 8)
    r_p = patches.recompose_images(p[0], 8)
    assert dset_10.shape == r_p.shape

def test_interp_patches(dset20, dset_10):
    patches.interp_patches(dset20, dset10.shape)
