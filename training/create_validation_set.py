import os
import glob
import argparse

from random import randrange
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="Create train validation index split file"
    )
    parser.add_argument("--path", help="Path of data. Only relevant if set.")
    parser.add_argument(
        "--run_60", action="store_true", help="Generate val_index for 60m patches."
    )
    args = parser.parse_args()
    return args


def main(args):
    # The `val_index.npy` must be created every time the number of training patches changes. It defines (and keeps set)
    # which of the patches will be used for validation.

    # This file must be changed if the DSen2_60 net is trained! (change the `path` and size of patches)

    # Size: number of S2 tiles (times) patches per tile
    n_scenes = len(
        [os.path.basename(x) for x in sorted(glob.glob(args.path + "*SAFE"))]
    )
    n_patches = 8000
    if args.run_60:
        n_patches = 500
    size = n_scenes * n_patches
    ratio = 0.1
    nb = int(size * ratio)

    index = np.zeros(size).astype(np.bool)
    i = 0
    while np.sum(index.astype(np.int)) < nb:
        x = randrange(0, size)
        index[x] = True
        i += 1

    np.save(args.path + "val_index.npy", index)

    print("Full no of samples: {}".format(size))
    print("Validation samples: {}".format(np.sum(index.astype(np.int))))

    print("Number of iterations: {}".format(i))


if __name__ == "__main__":
    args = get_args()
    main(args)
