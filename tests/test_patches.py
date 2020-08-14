from pathlib import Path
import tempfile
import pytest

import numpy as np

from context import patches


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


@pytest.mark.skip(reason="too long test")
def save_test_patches(dset_10, dset_20):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_test_patches(dset_10, dset_20, tmpdir + "/", 128, 8)
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 2


@pytest.mark.skip(reason="too long test")
def save_test_patches60(dset_10, dset_20, dset_60):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_test_patches60(dset_10, dset_20, dset_60, tmpdir + "/", 192, 12)
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 3


def test_get_random_patches(dset_10, dset_20, scale_20):
    data10_lr = patches.downPixelAggr(dset_10, SCALE=scale_20)
    data20_lr = patches.downPixelAggr(dset_20, SCALE=scale_20)
    r = patches.get_random_patches(dset_20, data10_lr, data20_lr, 8000)
    assert len(r) == 3
    assert r[0].shape == (8000, 4, 32, 32)
    assert r[1].shape == (8000, 5, 32, 32)
    assert r[2].shape == (8000, 5, 32, 32)


def test_get_random_patches60(dset_10, dset_20, dset_60, scale_60):
    data10_lr = patches.downPixelAggr(dset_10, SCALE=scale_60)
    data20_lr = patches.downPixelAggr(dset_20, SCALE=scale_60)
    data60_lr = patches.downPixelAggr(dset_60, SCALE=scale_60)
    r = patches.get_random_patches60(dset_60, data10_lr, data20_lr, data60_lr, 8000)
    assert len(r) == 4
    assert r[0].shape == (8000, 4, 96, 96)
    assert r[1].shape == (8000, 3, 96, 96)
    assert r[2].shape == (8000, 5, 96, 96)
    assert r[3].shape == (8000, 3, 96, 96)


def test_get_crop_window():
    w = patches.get_crop_window(100, 50, 25)
    assert w == [100, 50, 125, 75]
    w = patches.get_crop_window(100, 50, 25, 2)
    assert w == [200, 100, 250, 150]


def test_crop_array_to_window():
    ar = np.ones(shape=(100, 100, 4))
    w = patches.get_crop_window(50, 50, 25)
    assert patches.crop_array_to_window(ar, w).shape == (4, 25, 25)
    assert patches.crop_array_to_window(ar, w, False).shape == (25, 25, 4)


@pytest.mark.skip(reason="too long test")
def test_save_random_patches(dset_10, dset_20, scale_20):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_random_patches(
            dset_20,
            patches.downPixelAggr(dset_10, scale_20),
            patches.downPixelAggr(dset_20, scale_20),
            tmpdir + "/",
        )
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 2


@pytest.mark.skip(reason="too long test")
def test_save_random_patches60(dset_10, dset_20, dset_60, scale_60):
    with tempfile.TemporaryDirectory() as tmpdir:
        patches.save_random_patches60(
            dset_60,
            patches.downPixelAggr(dset_10, scale_60),
            patches.downPixelAggr(dset_20, scale_60),
            patches.downPixelAggr(dset_60, scale_60),
            tmpdir + "/",
        )
        f = Path(tmpdir).glob("*/**")
        assert len(list(f)) == 3


def test_downPixelAggr(dset_10, scale_20):
    r = patches.downPixelAggr(dset_10, scale_20)
    assert r.shape == (5490, 5490, 4)

    dset_20_w = np.ones((5489, 5489, 6))
    r = patches.downPixelAggr(dset_20_w, scale_20)
    assert r.shape == (2744, 2744, 6)


def test_recompose_images(dset_10, dset_20):
    p = patches.get_test_patches(dset_10, dset_20, 128, 8)
    r_p = patches.recompose_images(p[0], 8, dset_10.shape)
    assert dset_10.shape == r_p.shape


@pytest.mark.skip(reason="too long test")
def test_interp_patches(dset_20, dset_10):
    r = patches.interp_patches(dset_20, dset_10.shape)
    assert r
