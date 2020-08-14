import sys
import argparse
from pathlib import Path

sys.path.append("..")

from create_patches import readS2fromFile
from create_patches import parser_common


from utils.data_utils import get_logger

LOGGER = get_logger(__name__)


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Read Sentinel-2 data. The code was adapted from N. Brodu.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_folder_path", help=("Path to folder with S2 SAFE files."),
    )
    parser = parser_common(parser)
    args = parser.parse_args()
    return args


def main(args):
    # pylint: disable=logging-fstring-interpolation
    LOGGER.info(f"I will proceed with file {args.data_folder_path}")

    for file_path in Path(args.data_folder_path).glob("S2*"):
        LOGGER.info(f"Processing {file_path}")
        readS2fromFile(
            str(file_path),
            "",
            args.save_prefix,
            args.rgb_images,
            args.run_60,
            args.true_data,
            args.test_data,
            args.train_data,
        ).process_patches()


if __name__ == "__main__":
    main(arg_parse())
