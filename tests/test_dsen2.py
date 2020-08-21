import pytest

from keras.models import Model

from context import DSen2Net


@pytest.fixture
def input_shape_20():
    return ((4, None, None), (6, None, None))


@pytest.fixture
def input_shape_60():
    return ((4, None, None), (6, None, None), (2, None, None))


def test_s2model_20(input_shape_20):
    m = DSen2Net.s2model(input_shape_20)
    assert isinstance(m, Model)


def test_s2model_60(input_shape_60):
    m = DSen2Net.s2model(input_shape_60)
    assert isinstance(m, Model)


def test_aesrmodel_20(input_shape_20):
    m = DSen2Net.aesrmodel(input_shape_20)
    assert isinstance(m, Model)


def test_aesrmodel_60(input_shape_60):
    m = DSen2Net.aesrmodel(input_shape_60)
    assert isinstance(m, Model)
