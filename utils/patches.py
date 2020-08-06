from __future__ import division
import os
import glob
import json
from math import ceil
from random import randrange

from typing import Tuple, List

import numpy as np
from skimage.transform import resize
import skimage.measure
from scipy.ndimage.filters import gaussian_filter


def interp_patches(
    image_20: np.ndarray, image_10_shape: Tuple[int, int, int, int]
) -> np.ndarray:
    """Upsample patches to shape of higher resolution"""
    data20_interp = np.zeros((image_20.shape[0:2] + image_10_shape[2:4])).astype(
        np.float32
    )
    for k in range(image_20.shape[0]):
        for w in range(image_20.shape[1]):
            data20_interp[k, w] = (
                resize(image_20[k, w] / 30000, image_10_shape[2:4], mode="reflect")
                * 30000
            )  # bilinear
    return data20_interp


def get_patches(
    dset: np.ndarray,
    patch_size: int,
    border: int,
    patches_along_i: int,
    patches_along_j: int,
) -> np.ndarray:
    n_bands = dset.shape[2]

    # array index
    nr_patches = (patches_along_i + 1) * (patches_along_j + 1)
    range_i = np.arange(0, patches_along_i) * (patch_size - 2 * border)
    range_j = np.arange(0, patches_along_j) * (patch_size - 2 * border)

    patches = np.zeros((nr_patches, n_bands) + (patch_size, patch_size)).astype(
        np.float32
    )

    # if height and width are divisible by patch size - border * 2,
    # add one extra patch at the end of the image
    if np.mod(dset.shape[0] - 2 * border, patch_size - 2 * border) != 0:
        range_i = np.append(range_i, (dset.shape[0] - patch_size))
    if np.mod(dset.shape[1] - 2 * border, patch_size - 2 * border) != 0:
        range_j = np.append(range_j, (dset.shape[1] - patch_size))

    patch_count = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            # make shape (p, c, w, h)
            patches[patch_count] = crop_array_to_window(
                dset,
                get_crop_window(upper_left_i, upper_left_j, patch_size, 1),
                rollaxis=True,
            )
            patch_count += 1

    assert patch_count == nr_patches == patches.shape[0]
    return patches


def get_test_patches(
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    patchSize: int = 128,
    border: int = 4,
    interp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Used for inference. Creates patches of specific size in the whole image (10m and 20m)"""

    patch_size_lr = patchSize // 2
    border_lr = border // 2

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(
        dset_10, ((border, border), (border, border), (0, 0)), mode="symmetric",
    )
    dset_20 = np.pad(
        dset_20,
        ((border_lr, border_lr), (border_lr, border_lr), (0, 0)),
        mode="symmetric",
    )

    patchesAlongi = (dset_20.shape[0] - 2 * border_lr) // (
        patch_size_lr - 2 * border_lr
    )
    patchesAlongj = (dset_20.shape[1] - 2 * border_lr) // (
        patch_size_lr - 2 * border_lr
    )

    image_10 = get_patches(dset_10, patchSize, border, patchesAlongi, patchesAlongj)
    image_20 = get_patches(
        dset_20, patch_size_lr, border_lr, patchesAlongi, patchesAlongj
    )

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
    else:
        data20_interp = image_20
    return image_10, data20_interp


def get_test_patches60(
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    dset_60: np.ndarray,
    patchSize: int = 192,
    border: int = 12,
    interp: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Used for inference. Creates patches of specific size in the whole image (10m, 20m and 60m)"""

    patch_size_20 = patchSize // 2
    patch_size_60 = patchSize // 6
    border_20 = border // 2
    border_60 = border // 6

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = np.pad(
        dset_10, ((border, border), (border, border), (0, 0)), mode="symmetric",
    )
    dset_20 = np.pad(
        dset_20,
        ((border_20, border_20), (border_20, border_20), (0, 0)),
        mode="symmetric",
    )
    dset_60 = np.pad(
        dset_60,
        ((border_60, border_60), (border_60, border_60), (0, 0)),
        mode="symmetric",
    )

    patchesAlongi = (dset_60.shape[0] - 2 * border_60) // (
        patch_size_60 - 2 * border_60
    )
    patchesAlongj = (dset_60.shape[1] - 2 * border_60) // (
        patch_size_60 - 2 * border_60
    )

    image_10 = get_patches(dset_10, patchSize, border, patchesAlongi, patchesAlongj)
    image_20 = get_patches(
        dset_20, patch_size_20, border_20, patchesAlongi, patchesAlongj
    )
    image_60 = get_patches(
        dset_60, patch_size_60, border_60, patchesAlongi, patchesAlongj
    )

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
        data60_interp = interp_patches(image_60, image_10_shape)

    else:
        data20_interp = image_20
        data60_interp = image_60

    return image_10, data20_interp, data60_interp


def save_test_patches(
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    file: str,
    patchSize: int = 128,
    border: int = 4,
    interp: bool = True,
):
    """Save patches for inference into files (10 and 20m)"""
    image_10, data20_interp = get_test_patches(
        dset_10, dset_20, patchSize=patchSize, border=border, interp=interp
    )

    print("Saving to file {}".format(file))

    np.save(file + "data10", image_10)
    np.save(file + "data20", data20_interp)
    print("Done!")


def save_test_patches60(
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    dset_60: np.ndarray,
    file: str,
    patchSize: int = 192,
    border: int = 12,
    interp: bool = True,
):
    """Save patches for inference into files (10m, 20m and 60m)"""
    image_10, data20_interp, data60_interp = get_test_patches60(
        dset_10, dset_20, dset_60, patchSize=patchSize, border=border, interp=interp
    )
    print("Saving to file {}".format(file))

    np.save(file + "data10", image_10)
    np.save(file + "data20", data20_interp)
    np.save(file + "data60", data60_interp)
    print("Done!")


def get_crop_window(
    upper_left_x: int, upper_left_y: int, patch_size: int, scale: int = 1
) -> List[int]:
    """From a x,y coordinate pair and patch size return a list ofpixel coordinates
    defining a window in an array. Optionally pass a scale factor."""
    crop_window = [
        upper_left_x,
        upper_left_y,
        upper_left_x + patch_size,
        upper_left_y + patch_size,
    ]
    crop_window = [p * scale for p in crop_window]
    return crop_window


def crop_array_to_window(
    array: np.ndarray, crop_window: List[int], rollaxis: bool = True
) -> np.ndarray:
    """Return a subset of a numpy array. Rollaxis optional from channels last
    to channels first and vice versa. """
    cropped_array = array[
        crop_window[0] : crop_window[2], crop_window[1] : crop_window[3]
    ]
    if rollaxis:
        return np.rollaxis(cropped_array, 2,)
    else:
        return cropped_array


def get_random_patches(
    dset_20gt: np.ndarray,
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    nr_patches: int = 8000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns a set number of patches randomly select from a 10m and 20m resolution."""
    patch_size = 16

    BANDS10 = dset_10.shape[2]
    BANDS20 = dset_20.shape[2]
    label_20 = np.zeros(
        (nr_patches, BANDS20) + (patch_size * 2, patch_size * 2)
    ).astype(np.float32)
    image_20 = np.zeros((nr_patches, BANDS20) + (patch_size, patch_size)).astype(
        np.float32
    )
    image_10 = np.zeros(
        (nr_patches, BANDS10) + (patch_size * 2, patch_size * 2)
    ).astype(np.float32)

    for i in range(0, nr_patches):
        # while True:
        upper_left_x = randrange(0, dset_20.shape[0] - patch_size)
        upper_left_y = randrange(0, dset_20.shape[1] - patch_size)

        label_20[i] = crop_array_to_window(
            dset_20gt,
            get_crop_window(upper_left_x, upper_left_y, patch_size, 2),
            rollaxis=True,
        )
        image_20[i] = crop_array_to_window(
            dset_20,
            get_crop_window(upper_left_x, upper_left_y, patch_size),
            rollaxis=True,
        )
        image_10[i] = crop_array_to_window(
            dset_10,
            get_crop_window(upper_left_x, upper_left_y, patch_size, 2),
            rollaxis=True,
        )

    image_20 = interp_patches(image_20, image_10.shape)

    return image_10, label_20, image_20


def get_random_patches60(
    dset_60gt: np.ndarray,
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    dset_60: np.ndarray,
    nr_patches: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns a set number of patches randomly select from a 10m, 20m and 60m resolution."""

    patch_size = 16

    BANDS10 = dset_10.shape[2]
    BANDS20 = dset_20.shape[2]
    BANDS60 = dset_60.shape[2]
    label_60 = np.zeros(
        (nr_patches, BANDS60) + (patch_size * 6, patch_size * 6)
    ).astype(np.float32)
    image_10 = np.zeros(
        (nr_patches, BANDS10) + (patch_size * 6, patch_size * 6)
    ).astype(np.float32)
    image_20 = np.zeros(
        (nr_patches, BANDS20) + (patch_size * 3, patch_size * 3)
    ).astype(np.float32)
    image_60 = np.zeros((nr_patches, BANDS60) + (patch_size, patch_size)).astype(
        np.float32
    )

    for i in range(0, nr_patches):
        upper_left_x = randrange(0, dset_60.shape[0] - patch_size)
        upper_left_y = randrange(0, dset_60.shape[1] - patch_size)

        label_60[i] = crop_array_to_window(
            dset_60gt,
            get_crop_window(upper_left_x, upper_left_y, patch_size, 6),
            rollaxis=True,
        )
        image_10[i] = crop_array_to_window(
            dset_10,
            get_crop_window(upper_left_x, upper_left_y, patch_size, 6),
            rollaxis=True,
        )
        image_20[i] = crop_array_to_window(
            dset_20,
            get_crop_window(upper_left_x, upper_left_y, patch_size, 3),
            rollaxis=True,
        )
        image_60[i] = crop_array_to_window(
            dset_60,
            get_crop_window(upper_left_x, upper_left_y, patch_size, 1),
            rollaxis=True,
        )

    image_20 = interp_patches(image_20, image_10.shape)
    image_60 = interp_patches(image_60, image_10.shape)
    return image_10, label_60, image_20, image_60


def save_random_patches(
    dset_20gt: np.ndarray,
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    file: str,
    NR_CROP: int = 8000,
):
    """Save patches into file for training (10 and 20m)"""
    image_10, label_20, image_20 = get_random_patches(
        dset_20gt, dset_10, dset_20, NR_CROP
    )

    np.save(file + "data10", image_10)
    del image_10
    np.save(file + "data20_gt", label_20)
    del label_20
    np.save(file + "data20", image_20)
    del image_20
    print("Done!")


def save_random_patches60(
    dset_60gt: np.ndarray,
    dset_10: np.ndarray,
    dset_20: np.ndarray,
    dset_60: np.ndarray,
    file: str,
    NR_CROP: int = 500,
):
    """Save patches into file for training (10, 20m and 60m)"""

    image_10, label_60, image_20, image_60 = get_random_patches60(
        dset_60gt, dset_10, dset_20, dset_60, NR_CROP
    )
    np.save(file + "data10", image_10)
    del image_10
    np.save(file + "data60_gt", label_60)
    del label_60

    np.save(file + "data20", image_20)
    del image_20

    np.save(file + "data60", image_60)
    del image_60

    print("Done!")


def splitTrainVal(
    train_path: str, train: List[np.ndarray], label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create test validation split from val_index.npy file generated from create_validation_set."""
    # val_ind is numpy array
    # pylint: disable=invalid-unary-operand-type
    try:
        val_ind = np.load(train_path + "val_index.npy")
    except IOError:
        print(
            "Please define the validation split indices, usually located in .../data/test/. To generate this file use"
            " createRandom.py"
        )
    val_tr = [p[val_ind] for p in train]
    train = [p[~val_ind] for p in train]
    val_lb = label[val_ind]
    label = label[~val_ind]
    print("Loaded {} patches for training.".format(val_ind.shape[0]))
    return train, label, val_tr, val_lb


def OpenDataFiles(
    path: str, run_60: bool, SCALE: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From path with train patches, return numpy array with train and val patches."""
    if run_60:
        train_path = path + "train60/"
    else:
        train_path = path + "train/"

    # Create list from path
    fileList = [os.path.basename(x) for x in sorted(glob.glob(train_path + "*SAFE"))]
    for i, dset in enumerate(fileList):
        if i == 0:
            data10 = np.load(train_path + dset + "/data10.npy")
            data20 = np.load(train_path + dset + "/data20.npy")
        else:
            data10_new = np.load(train_path + dset + "/data10.npy")
            data20_new = np.load(train_path + dset + "/data20.npy")
            data10 = np.concatenate((data10, data10_new))
            data20 = np.concatenate((data20, data20_new))

        if run_60:
            if i == 0:
                data60_gt = np.load(train_path + dset + "/data60_gt.npy")
                data60 = np.load(train_path + dset + "/data60.npy")
            else:
                data60_gt_new = np.load(train_path + dset + "/data60_gt.npy")
                data60_new = np.load(train_path + dset + "/data60.npy")
                data60_gt = np.concatenate((data60_gt, data60_gt_new))
                data60 = np.concatenate((data60, data60_new))

        else:
            if i == 0:
                data20_gt = np.load(train_path + dset + "/data20_gt.npy")
            else:
                data20_gt_new = np.load(train_path + dset + "/data20_gt.npy")
                data20_gt = np.concatenate((data20_gt, data20_gt_new))

    if SCALE:
        data10 /= SCALE
        data20 /= SCALE
        if run_60:
            data60 /= SCALE
            data60_gt /= SCALE
        else:
            data20_gt /= SCALE

    if run_60:
        return splitTrainVal(train_path, [data10, data20, data60], data60_gt)
    else:
        return splitTrainVal(train_path, [data10, data20], data20_gt)


def OpenDataFilesTest(
    path: str, run_60: bool, SCALE: int, true_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """From path with patches, return numpy array with patcches used for inference."""
    if not SCALE:
        SCALE = 1

    data10 = np.load(path + "/data10.npy")
    data20 = np.load(path + "/data20.npy")
    data10 /= SCALE
    data20 /= SCALE
    if run_60:
        data60 = np.load(path + "/data60.npy")
        data60 /= SCALE
        train = [data10, data20, data60]
    else:
        train = [data10, data20]

    with open(path + "/roi.json") as data_file:
        data = json.load(data_file)

    image_size = [(data[2] - data[0]), (data[3] - data[1])]

    print("The image size is: {}".format(image_size))
    print("The SCALE is: {}".format(SCALE))
    print("The true_scale is: {}".format(true_scale))
    return train, image_size


def downPixelAggr(img: np.ndarray, SCALE: int = 2) -> np.ndarray:
    """Down-scale array by scale factor. Applu gaussian blur and block reduce. """
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_blur = np.zeros(img.shape)
    # Filter the image with a Gaussian filter
    for i in range(0, img.shape[2]):
        img_blur[:, :, i] = gaussian_filter(img[:, :, i], 1 / SCALE)
    # New image dims
    new_dims = tuple(s // SCALE for s in img.shape)
    img_lr = np.zeros(new_dims[0:2] + (img.shape[-1],))
    # Iterate through all the image channels with avg pooling (pixel aggregation)
    for i in range(0, img.shape[2]):
        img_lr[:, :, i] = skimage.measure.block_reduce(
            img_blur[:, :, i], (SCALE, SCALE), np.mean
        )

    return np.squeeze(img_lr)


def recompose_images(a: np.ndarray, border: int, size=None) -> np.ndarray:
    """ From array with patches recompose original image."""
    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2] - border * 2

        # print('Patch has dimension {}'.format(patch_size))
        # print('Prediction has shape {}'.format(a.shape))
        x_tiles = int(ceil(size[1] / float(patch_size)))
        y_tiles = int(ceil(size[0] / float(patch_size)))
        # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        # print('Image size is: {}'.format(size))
        images = np.zeros((a.shape[1], size[0], size[1])).astype(np.float32)

        print(images.shape)
        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[0] - patch_size:
                ypoint = size[0] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[1] - patch_size:
                    xpoint = size[1] - patch_size
                images[
                    :, ypoint : ypoint + patch_size, xpoint : xpoint + patch_size
                ] = a[
                    current_patch,
                    :,
                    border : a.shape[2] - border,
                    border : a.shape[3] - border,
                ]
                current_patch += 1

    return images.transpose((1, 2, 0))
