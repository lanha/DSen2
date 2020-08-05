import os
import numpy as np

import pytest
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

from blockutils.common import ensure_data_directories_exist
from context import Superresolution


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="session")
def fixture_superresolution_clip():
    ensure_data_directories_exist()
    return Superresolution(
        "a.SAFE", "50.550671,26.15174,50.596161,26.19195", True, "/tmp/output"
    )


def test_start(fixture_superresolution_clip, monkeypatch):
    _location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    data_10m = os.path.join(_location_, "mock_data/test_10m.tif")
    data_20m = os.path.join(_location_, "mock_data/test_20m.tif")
    data_60m = os.path.join(_location_, "mock_data/test_60m.tif")
    expected_final_dset = [data_10m, data_20m, data_60m]

    def _mock_getdata(self):
        return expected_final_dset

    monkeypatch.setattr(Superresolution, "get_data", _mock_getdata)
    (
        data10,
        data20,
        data60,
        [xmin, ymin, xmax, ymax],
        pr,
    ) = fixture_superresolution_clip.start()
    assert data10.shape == (444, 456, 4)
    assert data20.shape == (221, 227, 6)
    assert data60.shape == (73, 75, 2)
    assert [xmin, ymin, xmax, ymax] == [48, 174, 503, 617]
    assert pr == {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": None,
        "width": 1584,
        "height": 1762,
        "count": 4,
        "crs": CRS.from_epsg(32639),
        "transform": Affine(10.0, 0.0, 454590.0, 0.0, -10.0, 2898770.0),
        "blockxsize": 128,
        "blockysize": 128,
        "tiled": True,
        "interleave": "pixel",
    }


def test_inference(fixture_superresolution_clip):
    _location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    fixture_superresolution_clip.validated_10m_bands = ["B4", "B3", "B2", "B8"]
    fixture_superresolution_clip.validated_20m_bands = [
        "B5",
        "B6",
        "B7",
        "B8A",
        "B11",
        "B12",
    ]
    fixture_superresolution_clip.validated_60m_bands = ["B1", "B9"]
    fixture_superresolution_clip.validated_descriptions_all = {
        "B4": "B4 (665 nm)",
        "B3": "B3 (560 nm)",
        "B2": "B2 (490 nm)",
        "B8": "B8 (842 nm)",
        "B5": "B5 (705 nm)",
        "B6": "B6 (740 nm)",
        "B7": "B7 (783 nm)",
        "B8A": "B8A (865 nm)",
        "B11": "B11 (1610 nm)",
        "B12": "B12 (2190 nm)",
        "B1": "B1 (443 nm)",
        "B9": "B9 (945 nm)",
    }
    fixture_superresolution_clip.data_name = (
        "S2B_MSIL1C_20200708T070629_N0209_R106_T39RVJ_20200708T100303.SAFE"
    )

    data10 = np.load(os.path.join(_location_, "mock_data/data_10.npy"))
    data20 = np.load(os.path.join(_location_, "mock_data/data_20.npy"))
    data60 = np.load(os.path.join(_location_, "mock_data/data_60.npy"))

    coord = [48, 174, 503, 617]
    pr = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": None,
        "width": 1584,
        "height": 1762,
        "count": 4,
        "crs": CRS.from_epsg(32639),
        "transform": Affine(10.0, 0.0, 454590.0, 0.0, -10.0, 2898770.0),
        "blockxsize": 128,
        "blockysize": 128,
        "tiled": True,
        "interleave": "pixel",
    }

    fixture_superresolution_clip.inference(data10, data20, data60, coord, pr)
    result_path = os.path.join(
        "/tmp/output",
        fixture_superresolution_clip.data_name.split(".")[0] + "_superresolution.tif",
    )
    assert os.path.isfile(result_path)
    with rasterio.open(result_path) as src:
        assert src.count == 12
        assert src.transform == Affine(10.0, 0.0, 455070.0, 0.0, -10.0, 2897030.0)
        assert src.profile["driver"] == "GTiff"
