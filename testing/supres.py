from __future__ import division
import numpy as np
import argparse
from skimage.transform import resize
from DSen2Net import s2model
from math import ceil

SCALE = 2000
MDL_PATH = '../models/'


def DSen2_20(d10, d20, deep=False):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 8
    p10, p20 = get_test_patches(d10, d20, patchSize=128, border=border)
    p10 /= SCALE
    p20 /= SCALE
    test = [p10, p20]
    input_shape = ((4, None, None), (6, None, None))
    prediction = _predict(test, input_shape, deep=deep)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def DSen2_60(d10, d20, d60, deep=False):
    # Input to the funcion must be of shape:
    #     d10: [x,y,4]      (B2, B3, B4, B8)
    #     d20: [x/2,y/4,6]  (B5, B6, B7, B8a, B11, B12)
    #     d60: [x/6,y/6,2]  (B1, B9) -- NOT B10
    #     deep: specifies whether to use VDSen2 (True), or DSen2 (False)

    border = 12
    p10, p20, p60 = get_test_patches60(d10, d20, d60, patchSize=192, border=border)
    p10 /= SCALE
    p20 /= SCALE
    p60 /= SCALE
    test = [p10, p20, p60]
    input_shape = ((4, None, None), (6, None, None), (2, None, None))
    prediction = _predict(test, input_shape, deep=deep, run_60=True)
    images = recompose_images(prediction, border=border, size=d10.shape)
    images *= SCALE
    return images


def _predict(test, input_shape, deep=False, run_60=False):
    # create model
    if deep:
        model = s2model(input_shape, num_layers=32, feature_size=256)
        predict_file = MDL_PATH+'s2_034_lr_1e-04.hdf5' if run_60 else MDL_PATH+'s2_033_lr_1e-04.hdf5'
    else:
        model = s2model(input_shape, num_layers=6, feature_size=128)
        predict_file = MDL_PATH+'s2_030_lr_1e-05.hdf5' if run_60 else MDL_PATH+'s2_032_lr_1e-04.hdf5'
    print('Symbolic Model Created.')

    model.load_weights(predict_file)
    print("Predicting using file: {}".format(predict_file))
    prediction = model.predict(test, verbose=1)
    return prediction


def recompose_images(a, border, size=None):
    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2]-border*2

        # print('Patch has dimension {}'.format(patch_size))
        # print('Prediction has shape {}'.format(a.shape))
        x_tiles = int(ceil(size[0]/float(patch_size)))
        y_tiles = int(ceil(size[1]/float(patch_size)))
        # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        # print('Image size is: {}'.format(size))
        images = np.zeros((a.shape[1], size[1], size[0])).astype(np.float32)

        print(images.shape)
        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[1] - patch_size:
                ypoint = size[1] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[0] - patch_size:
                    xpoint = size[0] - patch_size
                images[:, ypoint:ypoint+patch_size, xpoint:xpoint+patch_size] = a[current_patch, :, border:a.shape[2]-border, border:a.shape[2]-border]
                current_patch += 1

    return images.transpose((1, 2, 0))


def interp_patches(image_20lr, image_10lr_shape):
    data20_interp = np.zeros((image_20lr.shape[0:2] + image_10lr_shape[2:4])).astype(np.float32)
    for k in range(image_20lr.shape[0]):
        for w in range(image_20lr.shape[1]):
            data20_interp[k, w] = resize(image_20lr[k, w] / 30000, image_10lr_shape[2:4], mode='reflect') * 30000  # bicubic
    return data20_interp


def get_test_patches(dset_10lr, dset_20lr, patchSize=128, border=8, interp=True):

    PATCH_SIZE_HR = (patchSize, patchSize)
    PATCH_SIZE_LR = [p//2 for p in PATCH_SIZE_HR]
    BORDER_HR = border
    BORDER_LR = BORDER_HR//2

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10lr = np.pad(dset_10lr, ((BORDER_HR, BORDER_HR), (BORDER_HR, BORDER_HR), (0, 0)), mode='symmetric')
    dset_20lr = np.pad(dset_20lr, ((BORDER_LR, BORDER_LR), (BORDER_LR, BORDER_LR), (0, 0)), mode='symmetric')

    BANDS10 = dset_10lr.shape[2]
    BANDS20 = dset_20lr.shape[2]
    patchesAlongi = (dset_20lr.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    patchesAlongj = (dset_20lr.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)

    label_20 = np.zeros((nr_patches, BANDS20) + PATCH_SIZE_HR).astype(np.float32)
    image_20lr = np.zeros((nr_patches, BANDS20) + tuple(PATCH_SIZE_LR)).astype(np.float32)
    image_10lr = np.zeros((nr_patches, BANDS10) + PATCH_SIZE_HR).astype(np.float32)

    # print(label_20.shape)
    # print(image_20lr.shape)
    # print(image_10lr.shape)

    range_i = np.arange(0, (dset_20lr.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)) * (
        PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    range_j = np.arange(0, (dset_20lr.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)) * (
        PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    if not (np.mod(dset_20lr.shape[0] - 2 * BORDER_LR, PATCH_SIZE_LR[0] - 2 * BORDER_LR) == 0):
        range_i = np.append(range_i, (dset_20lr.shape[0] - PATCH_SIZE_LR[0]))
    if not (np.mod(dset_20lr.shape[1] - 2 * BORDER_LR, PATCH_SIZE_LR[1] - 2 * BORDER_LR) == 0):
        range_j = np.append(range_j, (dset_20lr.shape[1] - PATCH_SIZE_LR[1]))

    # print(range_i)
    # print(range_j)

    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            crop_point_lr = [upper_left_i,
                             upper_left_j,
                             upper_left_i + PATCH_SIZE_LR[0],
                             upper_left_j + PATCH_SIZE_LR[1]]
            crop_point_hr = [p*2 for p in crop_point_lr]
            image_20lr[pCount] = np.rollaxis(dset_20lr[crop_point_lr[0]:crop_point_lr[2],
                             crop_point_lr[1]:crop_point_lr[3]], 2)
            image_10lr[pCount] = np.rollaxis(dset_10lr[crop_point_hr[0]:crop_point_hr[2],
                             crop_point_hr[1]:crop_point_hr[3]], 2)
            pCount += 1

    image_10lr_shape = image_10lr.shape

    if interp:
        data20_interp = interp_patches(image_20lr, image_10lr_shape)
    else:
        data20_interp = image_20lr
    return image_10lr, data20_interp


def get_test_patches60(dset_10lr, dset_20lr, dset_60lr, patchSize=192, border=12, interp=True):

    PATCH_SIZE_10 = (patchSize, patchSize)
    PATCH_SIZE_20 = [p//2 for p in PATCH_SIZE_10]
    PATCH_SIZE_60 = [p//6 for p in PATCH_SIZE_10]
    BORDER_10 = border
    BORDER_20 = BORDER_10//2
    BORDER_60 = BORDER_10//6

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10lr = np.pad(dset_10lr, ((BORDER_10, BORDER_10), (BORDER_10, BORDER_10), (0, 0)), mode='symmetric')
    dset_20lr = np.pad(dset_20lr, ((BORDER_20, BORDER_20), (BORDER_20, BORDER_20), (0, 0)), mode='symmetric')
    dset_60lr = np.pad(dset_60lr, ((BORDER_60, BORDER_60), (BORDER_60, BORDER_60), (0, 0)), mode='symmetric')


    BANDS10 = dset_10lr.shape[2]
    BANDS20 = dset_20lr.shape[2]
    BANDS60 = dset_60lr.shape[2]
    patchesAlongi = (dset_60lr.shape[0] - 2 * BORDER_60) // (PATCH_SIZE_60[0] - 2 * BORDER_60)
    patchesAlongj = (dset_60lr.shape[1] - 2 * BORDER_60) // (PATCH_SIZE_60[1] - 2 * BORDER_60)

    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)

    image_10lr = np.zeros((nr_patches, BANDS10) + PATCH_SIZE_10).astype(np.float32)
    image_20lr = np.zeros((nr_patches, BANDS20) + tuple(PATCH_SIZE_20)).astype(np.float32)
    image_60lr = np.zeros((nr_patches, BANDS60) + tuple(PATCH_SIZE_60)).astype(np.float32)

    # print(image_10lr.shape)
    # print(image_20lr.shape)
    # print(image_60lr.shape)

    range_i = np.arange(0, (dset_60lr.shape[0] - 2 * BORDER_60) // (PATCH_SIZE_60[0] - 2 * BORDER_60)) * (
        PATCH_SIZE_60[0] - 2 * BORDER_60)
    range_j = np.arange(0, (dset_60lr.shape[1] - 2 * BORDER_60) // (PATCH_SIZE_60[1] - 2 * BORDER_60)) * (
        PATCH_SIZE_60[1] - 2 * BORDER_60)

    if not (np.mod(dset_60lr.shape[0] - 2 * BORDER_60, PATCH_SIZE_60[0] - 2 * BORDER_60) == 0):
        range_i = np.append(range_i, (dset_60lr.shape[0] - PATCH_SIZE_60[0]))
    if not (np.mod(dset_60lr.shape[1] - 2 * BORDER_60, PATCH_SIZE_60[1] - 2 * BORDER_60) == 0):
        range_j = np.append(range_j, (dset_60lr.shape[1] - PATCH_SIZE_60[1]))

    # print(range_i)
    # print(range_j)

    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            crop_point_60 = [upper_left_i,
                             upper_left_j,
                             upper_left_i + PATCH_SIZE_60[0],
                             upper_left_j + PATCH_SIZE_60[1]]
            crop_point_10 = [p*6 for p in crop_point_60]
            crop_point_20 = [p*3 for p in crop_point_60]
            image_10lr[pCount] = np.rollaxis(dset_10lr[crop_point_10[0]:crop_point_10[2],
                                             crop_point_10[1]:crop_point_10[3]], 2)
            image_20lr[pCount] = np.rollaxis(dset_20lr[crop_point_20[0]:crop_point_20[2],
                                             crop_point_20[1]:crop_point_20[3]], 2)
            image_60lr[pCount] = np.rollaxis(dset_60lr[crop_point_60[0]:crop_point_60[2],
                                             crop_point_60[1]:crop_point_60[3]], 2)
            pCount += 1

    image_10lr_shape = image_10lr.shape

    if interp:
        data20_interp = interp_patches(image_20lr, image_10lr_shape)
    else:
        data20_interp = image_20lr

    if interp:
        data60_interp = interp_patches(image_60lr, image_10lr_shape)
    else:
        data60_interp = image_60lr

    return image_10lr, data20_interp, data60_interp




