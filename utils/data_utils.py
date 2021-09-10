from __future__ import division
import os
import sys
import re
import glob
import logging

from collections import defaultdict
from typing import List, Tuple
import numpy as np
import imageio

import rasterio
from rasterio.windows import Window
import pyproj as proj

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    """
    Instantiate a logger with a given level and name.

    Example:
        ```python
        logger = get_logger(__name__)
        # __name__ is the name of the file where the logger is instantiated.
        ```

    Arguments:
        name: A name for the logger - is included in all the logging messages.
        level: A logging level (i.e. logging.DEBUG).

    Returns:
        A logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


LOGGER = get_logger(__name__)


class DATA_UTILS:
    def __init__(self, data_file_path):
        self.data_file_path = data_file_path

    # pylint: disable=attribute-defined-outside-init
    def get_data(self) -> list:
        """
        This method returns the raster data set of original image for
        all the available resolutions.
        """
        data_folder = "MTD*.xml"
        for file in glob.iglob(os.path.join(self.data_file_path, data_folder)):
            data_path = file

        LOGGER.info(f"Data path is {data_path}")

        raster_data = rasterio.open(data_path)
        datasets = raster_data.subdatasets

        return datasets

    @staticmethod
    def get_max_min(x_1: int, y_1: int, x_2: int, y_2: int, data) -> Tuple:
        """
        This method gets pixels' location for the region of interest on the 10m bands
        and returns the min/max in each direction and to nearby 60m pixel boundaries and the area
        associated to the region of interest.
        **Example**
        >>> get_max_min(0,0,400,400)
        (0, 0, 395, 395, 156816)

        """
        with rasterio.open(data) as d_s:
            d_width = d_s.width
            d_height = d_s.height

        tmxmin = max(min(x_1, x_2, d_width - 1), 0)
        tmxmax = min(max(x_1, x_2, 0), d_width - 1)
        tmymin = max(min(y_1, y_2, d_height - 1), 0)
        tmymax = min(max(y_1, y_2, 0), d_height - 1)
        # enlarge to the nearest 60 pixel boundary for the super-resolution
        tmxmin = int(tmxmin / 6) * 6
        tmxmax = int((tmxmax + 1) / 6) * 6 - 1
        tmymin = int(tmymin / 6) * 6
        tmymax = int((tmymax + 1) / 6) * 6 - 1
        area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
        return tmxmin, tmymin, tmxmax, tmymax, area

    # pylint: disable-msg=too-many-locals
    def to_xy(self, lon: float, lat: float, data) -> Tuple:
        """
        This method gets the longitude and the latitude of a given point and projects it
        into pixel location in the new coordinate system.
        :param lon: The longitude of a chosen point
        :param lat: The longitude of a chosen point
        :return: The pixel location in the coordinate system of the input image
        """
        # get the image's coordinate system.
        with rasterio.open(data) as d_s:
            coor = d_s.transform
        a_t, b_t, xoff, d_t, e_t, yoff = [coor[x] for x in range(6)]

        # transform the lat and lon into x and y position which are defined in
        # the world's coordinate system.
        local_crs = self.get_utm(data)
        crs_wgs = proj.Proj(init="epsg:4326")  # WGS 84 geographic coordinate system
        crs_bng = proj.Proj(init=local_crs)  # use a locally appropriate projected CRS
        x_p, y_p = proj.transform(crs_wgs, crs_bng, lon, lat)
        x_p -= xoff
        y_p -= yoff

        # matrix inversion
        # get the x and y position in image's coordinate system.
        det_inv = 1.0 / (a_t * e_t - d_t * b_t)
        x_n = (e_t * x_p - b_t * y_p) * det_inv
        y_n = (-d_t * x_p + a_t * y_p) * det_inv
        return int(x_n), int(y_n)

    @staticmethod
    def get_utm(data) -> str:
        """
        This method returns the utm of the input image.
        :param data: The raster file for a specific resolution.
        :return: UTM of the selected raster file.
        """
        with rasterio.open(data) as d_s:
            data_crs = d_s.crs.to_dict()
        utm = data_crs["init"]
        return utm

    # pylint: disable-msg=too-many-locals
    def area_of_interest(self, data, clip_to_aoi) -> Tuple:
        """
        This method returns the coordinates that define the desired area of interest.
        """
        if clip_to_aoi:
            roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [
                float(x) for x in re.split(",", clip_to_aoi)
            ]
            x_1, y_1 = self.to_xy(roi_lon1, roi_lat1, data)
            x_2, y_2 = self.to_xy(roi_lon2, roi_lat2, data)
        else:
            x_1, y_1, x_2, y_2 = 0, 0, 20000, 20000

        xmi, ymi, xma, yma, area = self.get_max_min(x_1, y_1, x_2, y_2, data)
        return xmi, ymi, xma, yma, area

    @staticmethod
    def validate_description(description: str) -> str:
        """
        This method rewrites the description of each band in the given data set.
        :param description: The actual description of a chosen band.

        **Example**
        >>> ds10.descriptions[0]
        'B4, central wavelength 665 nm'
        >>> validate_description(ds10.descriptions[0])
        'B4 (665 nm)'
        """
        m_re = re.match(r"(.*?), central wavelength (\d+) nm", description)
        if m_re:
            return m_re.group(1) + " (" + m_re.group(2) + " nm)"
        return description

    @staticmethod
    def get_band_short_name(description: str) -> str:
        """
        This method returns only the name of the bands at a chosen resolution.

        :param description: This is the output of the validate_description method.

        **Example**
        >>> desc = validate_description(ds10.descriptions[0])
        >>> desc
        'B4 (665 nm)'
        >>> get_band_short_name(desc)
        'B4'
        """
        if "," in description:
            return description[: description.find(",")]
        if " " in description:
            return description[: description.find(" ")]
        return description[:3]

    def validate(self, data) -> Tuple:
        """
        This method takes the short name of the bands for each
        separate resolution and returns three lists. The validated_
        bands and validated_indices contain the name of the bands and
        the indices related to them respectively.
        The validated_descriptions is a list of descriptions for each band
        obtained from the validate_description method.
        :param data: The raster file for a specific resolution.
        **Example**
        >>> validated_10m_bands, validated_10m_indices, \
        >>> dic_10m = validate(ds10)
        >>> validated_10m_bands
        ['B4', 'B3', 'B2', 'B8']
        >>> validated_10m_indices
        [0, 1, 2, 3]
        >>> dic_10m
        defaultdict(<class 'str'>, {'B4': 'B4 (665 nm)',
         'B3': 'B3 (560 nm)', 'B2': 'B2 (490 nm)', 'B8': 'B8 (842 nm)'})
        """
        input_select_bands = "B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12"  # type: str
        select_bands = re.split(",", input_select_bands)  # type: List[str]
        validated_bands = []  # type: list
        validated_indices = []  # type: list
        validated_descriptions = defaultdict(str)  # type: defaultdict
        with rasterio.open(data) as d_s:
            for i in range(0, d_s.count):
                desc = self.validate_description(d_s.descriptions[i])
                name = self.get_band_short_name(desc)
                if name in select_bands:
                    select_bands.remove(name)
                    validated_bands += [name]
                    validated_indices += [i]
                    validated_descriptions[name] = desc
        return validated_bands, validated_indices, validated_descriptions

    # pylint: disable-msg=too-many-arguments
    @staticmethod
    def data_final(
        data, term: List, x_mi: int, y_mi: int, x_ma: int, y_ma: int, n_res, scale
    ) -> np.ndarray:
        """
        This method takes the raster file at a specific
        resolution and uses the output of get_max_min
        to specify the area of interest.
        Then it returns an numpy array of values
        for all the pixels inside the area of interest.
        :param data: The raster file for a specific resolution.
        :param term: The validate indices of the
        bands obtained from the validate method.
        :return: The numpy array of pixels' value.
        """
        if term:
            LOGGER.info(term)
            with rasterio.open(data) as d_s:
                d_final = np.rollaxis(
                    d_s.read(
                        window=Window(
                            col_off=x_mi // scale,
                            row_off=y_mi // scale,
                            width=(x_ma - x_mi + n_res) // scale,
                            height=(y_ma - y_mi + n_res) // scale,
                        )
                    ),
                    0,
                    3,
                )[:, :, term]
        return d_final

    @staticmethod
    def save_band(save_prefix: str, data: np.ndarray, name: str, percentile_data=None):
        # The percentile_data argument is used to plot superresolved and original data
        # with a comparable black/white scale
        if percentile_data is None:
            percentile_data = data
        mi, ma = np.percentile(percentile_data, (1, 99))
        band_data = np.clip(data, mi, ma)
        band_data = (band_data - mi) / (ma - mi)
        imageio.imsave(save_prefix + name + ".png", band_data)

    @staticmethod
    def check_size(dims):
        xmin, ymin, xmax, ymax = dims
        if xmax < xmin or ymax < ymin:
            LOGGER.error("Invalid region of interest / UTM Zone combination")
            sys.exit(1)

        if (xmax - xmin) < 192 or (ymax - ymin) < 192:
            LOGGER.error(
                "AOI too small. Try again with a larger AOI (minimum pixel width or heigh of 192)"
            )
            # sys.exit(1)
