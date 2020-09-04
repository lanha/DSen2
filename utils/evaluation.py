from __future__ import print_function, division
import os

import time
import argparse
from glob import glob

from tensorflow import keras
import numpy as np
import skimage.transform

# For usage of eval in evaluation
# pylint: disable=unused-import
from image_similarity_measures.quality_metrics import (
    psnr,
    uiq,
    sam,
    sre,
    ssim,
    issm,
    fsim,
)

from data_utils import get_logger
from patches import recompose_images, OpenDataFilesTest

logger = get_logger(__name__)

SCALE = 2000
MODEL_PATH = "../models/"


def rmse(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Root Mean Squared Error
    """
    rmse_final = []
    for i in range(org_img.shape[2]):
        m = np.mean(((org_img[:, :, i] - pred_img[:, :, i])) ** 2)
        s = np.sqrt(m)
        rmse_final.append(s)
    return np.mean(rmse_final)


def write_final_dict(metric, metric_dict):
    # Create a directory to save the text file of including evaluation values.
    predict_path = "val_predict/"
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    with open(os.path.join(predict_path, metric + ".txt"), "w") as f:
        f.writelines("{}:{}\n".format(k, v) for k, v in metric_dict.items())


def predict_downsampled_img(path, model_path, folder, dset, border, final_name):

    model = keras.models.load_model(model_path)

    start = time.time()
    print("Timer started.")
    print("Predicting: {}.".format(dset))
    train, image_size = OpenDataFilesTest(
        os.path.join(path, folder + dset), args.run_60, SCALE, False
    )
    logger.info("Predicting ...")
    prediction = model.predict(train, batch_size=8, verbose=1)

    images = recompose_images(prediction, border=border, size=image_size)
    print("Writing to file...")
    np.save(os.path.join(final_name), images * SCALE)
    end = time.time()
    logger.info(f"Elapsed time: {end - start}.")


def evaluation(org_img, pred_img, metric, bic=False):
    org_img_array = np.load(org_img)
    pred_img_array = np.load(pred_img)
    print("eval %d %d %d" % pred_img_array.shape)
    if bic:
        pred_img_array = skimage.transform.resize(org_img_array, org_img_array.shape)

    print("eval %d %d %d" % org_img_array.shape)
    print("eval %d %d %d" % pred_img_array.shape)
    org_img_shape = org_img_array[:, :, :].shape
    pred_img_shape = pred_img_array[:, :, :].shape
    if org_img_shape != pred_img_shape:
        pred_img_array = pred_img_array[: org_img_shape[0], : org_img_shape[1]]

    # Fo usage of eval
    # pylint: disable=eval-used
    result = eval(f"{metric}(org_img_array, pred_img_array)")
    return result


def process(path, model_path, metric):
    if args.l1c:
        prefix = "l1c"
    if args.l2a:
        prefix = "l2a"
    if args.run_60:
        folder = prefix + "test60/"
        border = 12
    else:
        folder = prefix + "test/"
        border = 4

    path_to_patches = os.path.join(path, folder)

    fileList = [os.path.basename(x) for x in sorted(glob(path_to_patches + "*SAFE"))]

    mean_eval_value = []
    metric_dict = {}

    for dset in fileList:
        if args.run_60:
            org_img_path = os.path.join(
                path_to_patches, dset + "/no_tiling/data60_gt.npy"
            )
            bic_img_path = os.path.join(path_to_patches, dset + "/no_tiling/data60.npy")
            pred_img_path = os.path.join(
                path_to_patches, dset + "/no_tiling/data60_predicted.npy"
            )
        else:
            org_img_path = os.path.join(
                path_to_patches, dset + "/no_tiling/data20_gt.npy"
            )
            bic_img_path = os.path.join(path_to_patches, dset + "/no_tiling/data20.npy")
            pred_img_path = os.path.join(
                path_to_patches, dset + "/no_tiling/data20_predicted.npy"
            )

        predict_downsampled_img(path, model_path, folder, dset, border, pred_img_path)
        print(org_img_path, bic_img_path, pred_img_path)
        eval_value = evaluation(org_img_path, pred_img_path, metric)
        if args.bic:
            eval_value_bic = evaluation(org_img_path, bic_img_path, metric, bic=True)
            print(f"Bicubic: {eval_value_bic}")
        metric_dict[dset] = eval_value
        mean_eval_value.append(eval_value)
        print(f"NN: {eval_value}")

    metric_dict["mean"] = sum(mean_eval_value) / len(mean_eval_value)

    write_final_dict(metric, metric_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluates an Image Super Resolution Model"
    )
    parser.add_argument("--path", type=str, help="Path to image for evaluation")
    parser.add_argument("--model_path", type=str, help="Path to model weights")
    parser.add_argument("--l1c", action="store_true", help="Getting L1C samples")
    parser.add_argument("--l2a", action="store_true", help="Getting L2A samples")
    parser.add_argument("--bic", action="store_true", help="Compare bicubic result")
    parser.add_argument(
        "--metric",
        type=str,
        default="psnr",
        help="Use psnr, uiq, sam or sre as evaluation metric",
    )
    parser.add_argument(
        "--run_60",
        action="store_true",
        help="Whether to run a 60->10m network. Default 20->10m.",
    )

    args = parser.parse_args()
    path = args.path
    model_path = args.model_path
    metric = args.metric

    process(path, model_path, metric)
