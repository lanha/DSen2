from __future__ import division
import os
import sys
import argparse
import json

from typing import Tuple
import numpy as np

from utils.data_utils import DATA_UTILS, get_logger


sys.path.append("../")
from utils.patches import (
    downPixelAggr,
    save_test_patches,
    save_random_patches,
    save_random_patches60,
    save_test_patches60,
)

LOGGER = get_logger(__name__)


class readS2fromFile(DATA_UTILS):
    def __init__(
        self,
        data_file_path,
        clip_to_aoi=None,
        save_prefix="../data/",
        rgb_images=False,
        run_60=False,
        true_data=False,
        test_data=False,
        train_data=False,
    ):
        self.data_file_path = data_file_path
        self.test_data = test_data
        self.clip_to_aoi = clip_to_aoi
        self.save_prefix = save_prefix
        self.rgb_images = rgb_images
        self.run_60 = run_60
        self.true_data = true_data
        self.test_data = test_data
        self.train_data = train_data

        super().__init__(data_file_path)

    def get_original_image(self) -> Tuple:

        data_list = self.get_data()
        for dsdesc in data_list:
            if "10m" in dsdesc:
                xmin, ymin, xmax, ymax, interest_area = self.area_of_interest(
                    dsdesc, self.clip_to_aoi
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
                _, validated_10m_indices, _ = self.validate(dsdesc)
                data10 = self.data_final(
                    dsdesc, validated_10m_indices, xmin, ymin, xmax, ymax, 1
                )
            if "20m" in dsdesc:
                LOGGER.info("Selected 20m bands:")
                _, validated_20m_indices, _ = self.validate(dsdesc)
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
                _, validated_60m_indices, _ = self.validate(dsdesc)
                data60 = self.data_final(
                    dsdesc,
                    validated_60m_indices,
                    xmin // 6,
                    ymin // 6,
                    xmax // 6,
                    ymax // 6,
                    1 // 6,
                )

        return data10, data20, data60, xmin, ymin, xmax, ymax

    def get_downsampled_images(self, data10, data20, data60) -> Tuple:
        if self.run_60:
            data10_lr = downPixelAggr(data10, SCALE=6)
            data20_lr = downPixelAggr(data20, SCALE=6)
            data60_lr = downPixelAggr(data60, SCALE=6)
            return data10_lr, data20_lr, data60_lr
        else:
            data10_lr = downPixelAggr(data10, SCALE=2)
            data20_lr = downPixelAggr(data20, SCALE=2)

            return data10_lr, data20_lr

    def process_patches(self):
        if self.run_60:
            scale = 6
        else:
            scale = 2

        # self.name = self.data_name.split(".")[0]

        data10, data20, data60, xmin, ymin, xmax, ymax = self.get_original_image()

        if self.test_data:
            out_per_image = self.saving_test_data(data10, data20, data60)
            with open(out_per_image + "roi.json", "w") as f:
                json.dump(
                    [
                        xmin // scale,
                        ymin // scale,
                        (xmax + 1) // scale,
                        (ymax + 1) // scale,
                    ],
                    f,
                )

        if self.rgb_images:
            self.create_rgb_images(data10, data20, data60)

        if self.true_data:
            out_per_image = self.saving_true_data(data10, data20, data60)
            with open(out_per_image + "roi.json", "w") as f:
                json.dump(
                    [
                        xmin // scale,
                        ymin // scale,
                        (xmax + 1) // scale,
                        (ymax + 1) // scale,
                    ],
                    f,
                )

        if self.train_data:
            self.saving_train_data(data10, data20, data60)

        LOGGER.info("Success.")

    def saving_test_data(self, data10, data20, data60):
        # if test_data:
        if self.run_60:
            data10_lr, data20_lr, data60_lr = self.get_downsampled_images(
                data10, data20, data60
            )
            out_per_image0 = self.save_prefix + "test60/"
            out_per_image = self.save_prefix + "test60/" + self.data_name + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)

            LOGGER.info(f"Writing files for testing to:{out_per_image}")
            save_test_patches60(data10_lr, data20_lr, data60_lr, out_per_image)

        else:
            data10_lr, data20_lr = self.get_downsampled_images(data10, data20, data60)
            out_per_image0 = self.save_prefix + "test/"
            out_per_image = self.save_prefix + "test/" + self.data_name + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)

            LOGGER.info(
                f"Writing files for testing to:{out_per_image}"
            )  # pylint: disable=logging-fstring-interpolation
            save_test_patches(data10_lr, data20_lr, out_per_image)

        if not os.path.isdir(out_per_image + "no_tiling/"):
            os.mkdir(out_per_image + "no_tiling/")

        LOGGER.info("Now saving the whole image without tiling...")
        if self.run_60:
            np.save(
                out_per_image + "no_tiling/" + "data60_gt", data60.astype(np.float32)
            )
            np.save(
                out_per_image + "no_tiling/" + "data60", data60_lr.astype(np.float32)
            )
        else:
            np.save(
                out_per_image + "no_tiling/" + "data20_gt", data20.astype(np.float32)
            )
            self.save_band(
                self.save_prefix,
                data10_lr[:, :, 0:3],
                "/test/" + self.data_name + "/RGB",
            )
        np.save(out_per_image + "no_tiling/" + "data10", data10_lr.astype(np.float32))
        np.save(out_per_image + "no_tiling/" + "data20", data20_lr.astype(np.float32))
        return out_per_image

    def create_rgb_images(self, data10, data20, data60):
        # elif write_images
        data10_lr, data20_lr = self.get_downsampled_images(data10, data20, data60)
        LOGGER.info("Creating RGB images...")
        self.save_band(
            self.save_prefix,
            data10_lr[:, :, 0:3],
            "/raw/rgbs/" + self.data_name + "RGB",
        )
        self.save_band(
            self.save_prefix,
            data20_lr[:, :, 0:3],
            "/raw/rgbs/" + self.data_name + "RGB20",
        )

    def saving_true_data(self, data10, data20, data60):
        # elif true_data:
        out_per_image0 = self.save_prefix + "true/"
        out_per_image = self.save_prefix + "true/" + self.data_name + "/"
        if not os.path.isdir(out_per_image0):
            os.mkdir(out_per_image0)
        if not os.path.isdir(out_per_image):
            os.mkdir(out_per_image)

        LOGGER.info(
            f"Writing files for testing to:{out_per_image}"
        )  # pylint: disable=logging-fstring-interpolation
        save_test_patches60(
            data10, data20, data60, out_per_image, patchSize=384, border=12
        )

        if not os.path.isdir(out_per_image + "no_tiling/"):
            os.mkdir(out_per_image + "no_tiling/")

        LOGGER.info("Now saving the whole image without tiling...")
        np.save(out_per_image + "no_tiling/" + "data10", data10.astype(np.float32))
        np.save(out_per_image + "no_tiling/" + "data20", data20.astype(np.float32))
        np.save(out_per_image + "no_tiling/" + "data60", data60.astype(np.float32))
        return out_per_image

    def saving_train_data(self, data10, data20, data60):
        # if train_data
        if self.run_60:
            out_per_image0 = self.save_prefix + "train60/"
            out_per_image = self.save_prefix + "train60/" + self.data_name + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)
            LOGGER.info(
                f"Writing files for training to:{out_per_image}"
            )  # pylint: disable=logging-fstring-interpolation
            data10_lr, data20_lr, data60_lr = self.get_downsampled_images(
                data10, data20, data60
            )
            save_random_patches60(
                data60, data10_lr, data20_lr, data60_lr, out_per_image
            )
        else:
            out_per_image0 = self.save_prefix + "train/"
            out_per_image = self.save_prefix + "train/" + self.data_name + "/"
            if not os.path.isdir(out_per_image0):
                os.mkdir(out_per_image0)
            if not os.path.isdir(out_per_image):
                os.mkdir(out_per_image)
            LOGGER.info(
                f"Writing files for training to:{out_per_image}"
            )  # pylint: disable=logging-fstring-interpolation
            data10_lr, data20_lr = self.get_downsampled_images(data10, data20, data60)
            save_random_patches(data20, data10_lr, data20_lr, out_per_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read Sentinel-2 data. The code was adapted from N. Brodu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file_path",
        help=(
            "An input Sentinel-2 data file. This can be either the original ZIP file,"
            " or the S2A[...].xml file in a SAFE directory extracted from that ZIP."
        ),
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
        "--test_data",
        default=False,
        action="store_true",
        help="Store test patches in a separate dir.",
    )
    parser.add_argument(
        "--rgb_images",
        default=False,
        action="store_true",
        help=(
            "If set, write PNG images for the original and the superresolved bands,"
            " together with a composite rgb image (first three 10m bands), all with a "
            "quick and dirty clipping to 99%% of the original bands dynamic range and "
            "a quantization of the values to 256 levels."
        ),
    )
    parser.add_argument(
        "--save_prefix",
        default="../data/",
        help=(
            "If set, speficies the name of a prefix for all output files. "
            "Use a trailing / to save into a directory. The default of no prefix will "
            "save into the current directory. Example: --save_prefix result/"
        ),
    )
    parser.add_argument(
        "--run_60",
        default=False,
        action="store_true",
        help="If set, it will create patches also from the 60m channels.",
    )
    parser.add_argument(
        "--true_data",
        default=False,
        action="store_true",
        help=(
            "If set, it will create patches for S2 without GT. This option is not "
            "really useful here, please check the testing folder for predicting S2 images."
        ),
    )
    parser.add_argument(
        "--train_data",
        default=False,
        action="store_true",
        help="Store train patches in a separate dir",
    )

    args = parser.parse_args()

    LOGGER.info(
        f"I will proceed with file {args.data_file}"
    )  # pylint: disable=logging-fstring-interpolation
    readS2fromFile(
        args.data_file_path,
        args.clip_to_aoi,
        args.save_prefix,
        args.rgb_images,
        args.run_60,
        args.true_data,
        args.test_data,
        args.train_data,
    ).process_patches()
