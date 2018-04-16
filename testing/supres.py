from __future__ import division
# import numpy as np
# import argparse
# from skimage.transform import resize
import sys
sys.path.append('../')
from utils.DSen2Net import s2model
from utils.patches import get_test_patches, get_test_patches60, recompose_images


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

