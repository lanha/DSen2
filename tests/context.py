"""
This module is used in test_s2_tiles_supres script.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils/"))
)


# pylint: disable=unused-import,wrong-import-position
from data_utils import DATA_UTILS, get_logger
