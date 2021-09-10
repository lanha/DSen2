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
    assert m.layers[-1].output_shape[1:] == input_shape_20[-1]
    assert isinstance(m, Model)


def test_s2model_60(input_shape_60):
    m = DSen2Net.s2model(input_shape_60)
    assert m.layers[-1].output_shape[1:] == input_shape_60[-1]
    assert isinstance(m, Model)


def test_aesrmodel_20(input_shape_20):
    m = DSen2Net.aesrmodel(input_shape_20)
    assert m.layers[-1].output_shape[1:] == input_shape_20[-1]
    assert isinstance(m, Model)


def test_aesrmodel_60(input_shape_60):
    m = DSen2Net.aesrmodel(input_shape_60)
    assert m.layers[-1].output_shape[1:] == input_shape_60[-1]
    assert isinstance(m, Model)


def test_srcnn_20(input_shape_20):
    m = DSen2Net.srcnn(input_shape_20)
    assert m.layers[-1].output_shape[1:] == input_shape_20[-1]
    assert isinstance(m, Model)


def test_srcnn_60(input_shape_60):
    m = DSen2Net.srcnn(input_shape_60)
    assert m.layers[-1].output_shape[1:] == input_shape_60[-1]
    assert isinstance(m, Model)


def test_rednetsr_20(input_shape_20):
    m = DSen2Net.rednetsr(input_shape_20)
    assert m.layers[-1].output_shape[1:] == input_shape_20[-1]
    assert isinstance(m, Model)


def test_rednetsr_60(input_shape_60):
    m = DSen2Net.rednetsr(input_shape_60)
    assert m.layers[-1].output_shape[1:] == input_shape_60[-1]
    assert isinstance(m, Model)


def test_resnetsr_20(input_shape_20):
    m = DSen2Net.resnetsr(input_shape_20)
    assert m.layers[-1].output_shape[1:] == input_shape_20[-1]
    assert isinstance(m, Model)


def test_resnetsr_60(input_shape_60):
    m = DSen2Net.resnetsr(input_shape_60)
    assert m.layers[-1].output_shape[1:] == input_shape_60[-1]
    assert isinstance(m, Model)
