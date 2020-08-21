from __future__ import division
import os
import sys
import gc
from typing import Tuple
import argparse

import rasterio
from rasterio import Affine as A

import numpy as np

from utils.data_utils import DATA_UTILS, get_logger
from supres import dsen2_20, dsen2_60

LOGGER = get_logger(__name__)

# pylint: disable-msg=too-many-arguments
def save_result(
    model_output, output_bands, valid_desc, output_profile, image_name,
):
    """
    This method saves the feature collection meta data and the
    image with high resolution for desired bands to the provided location.
    :param model_output: The high resolution image.
    :param output_bands: The associated bands for the output image.
    :param valid_desc: The valid description of the existing bands.
    :param output_profile: The georeferencing for the output image.
    :param output_features: The meta data for the output image.
    :param image_name: The name of the output image.

    """

    with rasterio.open(image_name, "w", **output_profile) as d_s:
        for b_i, b_n in enumerate(output_bands):
            d_s.write(model_output[:, :, b_i], indexes=b_i + 1)
            d_s.set_band_description(b_i + 1, "SR " + valid_desc[b_n])


# pylint: disable-msg=too-many-arguments
def update(pr_10m, size_10m: Tuple, model_output: np.ndarray, xmi: int, ymi: int):
    """
    This method creates the proper georeferencing for the output image.
    :param data: The raster file for 10m resolution.

    """
    # Here based on the params.json file, the output image dimension will be calculated.
    out_dims = model_output.shape[2]

    new_transform = pr_10m["transform"] * A.translation(xmi, ymi)
    pr_10m.update(dtype=rasterio.float32)
    pr_10m.update(driver="GTiff")
    pr_10m.update(width=size_10m[1])
    pr_10m.update(height=size_10m[0])
    pr_10m.update(count=out_dims)
    pr_10m.update(transform=new_transform)
    return pr_10m


class Superresolution(DATA_UTILS):
    def __init__(self, data_file_path, clip_to_aoi, copy_original_bands, output_dir):
        self.data_file_path = data_file_path
        self.clip_to_aoi = clip_to_aoi
        self.copy_original_bands = copy_original_bands
        self.output_dir = output_dir
        self.data_name = os.path.basename(data_file_path)

        super().__init__(data_file_path)

    # pylint: disable=attribute-defined-outside-init
    def start(self):
        data_list = self.get_data()

        for dsdesc in data_list:
            if "10m" in dsdesc:
                if self.clip_to_aoi:
                    xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(
                        dsdesc, self.clip_to_aoi
                    )
                else:
                    # Get the pixel bounds of the full scene
                    xmin, ymin, xmax, ymax, interest_area = self.get_max_min(
                        0, 0, 20000, 20000, dsdesc
                    )
                LOGGER.info("Selected pixel region:")
                LOGGER.info("xmin = %s", xmin)
                LOGGER.info("ymin = %s", ymin)
                LOGGER.info("xmax = %s", xmax)
                LOGGER.info("ymax = %s", ymax)
                LOGGER.info("The area of selected region = %s", interest_area)
            self.check_size(dims=(xmin, ymin, xmax, ymax))

        for dsdesc in data_list:
            if "10m" in dsdesc:
                LOGGER.info("Selected 10m bands:")
                (
                    self.validated_10m_bands,
                    validated_10m_indices,
                    dic_10m,
                ) = self.validate(dsdesc)
                data10 = self.data_final(
                    dsdesc, validated_10m_indices, xmin, ymin, xmax, ymax, 1
                )
                with rasterio.open(dsdesc) as d_s:
                    pr_10m = d_s.profile

            if "20m" in dsdesc:
                LOGGER.info("Selected 20m bands:")
                (
                    self.validated_20m_bands,
                    validated_20m_indices,
                    dic_20m,
                ) = self.validate(dsdesc)
                data20 = self.data_final(
                    dsdesc,
                    validated_20m_indices,
                    xmin // 2,
                    ymin // 2,
                    xmax // 2,
                    ymax // 2,
                    1 // 2,
                )
            if "60m" in dsdesc:
                LOGGER.info("Selected 60m bands:")
                (
                    self.validated_60m_bands,
                    validated_60m_indices,
                    dic_60m,
                ) = self.validate(dsdesc)
                data60 = self.data_final(
                    dsdesc,
                    validated_60m_indices,
                    xmin // 6,
                    ymin // 6,
                    xmax // 6,
                    ymax // 6,
                    1 // 6,
                )

        self.validated_descriptions_all = {**dic_10m, **dic_20m, **dic_60m}
        return data10, data20, data60, [xmin, ymin, xmax, ymax], pr_10m

    def inference(self, data10, data20, data60, coord, pr_10m):

        if (
            self.validated_60m_bands
            and self.validated_20m_bands
            and self.validated_10m_bands
        ):
            LOGGER.info("Super-resolving the 60m data into 10m bands")
            sr60 = dsen2_60(data10, data20, data60)
            LOGGER.info("Super-resolving the 20m data into 10m bands")
            sr20 = dsen2_20(data10, data20)
        else:
            LOGGER.info("No super-resolution performed, exiting")
            sys.exit(0)

        if self.copy_original_bands:
            sr_final = np.concatenate((data10, sr20, sr60), axis=2)
            validated_sr_final_bands = (
                self.validated_10m_bands
                + self.validated_20m_bands
                + self.validated_60m_bands
            )
        else:
            sr_final = np.concatenate((sr20, sr60), axis=2)
            validated_sr_final_bands = (
                self.validated_20m_bands + self.validated_60m_bands
            )

        pr_10m_updated = update(pr_10m, data10.shape, sr_final, coord[0], coord[1])

        path_to_output_img = self.data_name.split(".")[0] + "_superresolution.tif"
        filename = os.path.join(self.output_dir, path_to_output_img)

        LOGGER.info("Now writing the super-resolved bands")
        save_result(
            sr_final,
            validated_sr_final_bands,
            self.validated_descriptions_all,
            pr_10m_updated,
            filename,
        )
        del sr_final
        LOGGER.info("This is for releasing memory: %s", gc.collect())
        LOGGER.info("Writing the super-resolved bands is finished.")

    def process(self):
        data10, data20, data60, coord, pr_10m = self.start()
        self.inference(data10, data20, data60, coord, pr_10m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform super-resolution on Sentinel-2 with DSen2. Code based on superres"
        " by Nicolas Brodu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file_path",
        help="An input sentinel-2 data file. This can be either the original ZIP file, or the S2A[...].xml "
        "file in a SAFE directory extracted from that ZIP.",
    )
    parser.add_argument(
        "--clip_to_aoi",
        default="",
        help=(
            "Sets the region of interest to extract as pixels locations on the 10m"
            'bands. Use this syntax: x_1,y_1,x_2,y_2. E.g. --roi_x_y "2000,2000,3200,3200"'
        ),
    )
    parser.add_argument(
        "--copy_original_bands",
        action="store_true",
        help="The default is not to copy the original selected 10m bands into the output file in addition "
        "to the super-resolved bands. If this flag is used, the output file may be used as a 10m "
        "version of the original Sentinel-2 file.",
    )
    parser.add_argument(
        "--output_dir", default="", help="Directory to the final output",
    )
    args = parser.parse_args()
    Superresolution(
        args.data_file_path, args.clip_to_aoi, args.copy_original_bands, args.output_dir
    ).process()
