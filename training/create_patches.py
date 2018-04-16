from __future__ import division
import argparse
import numpy as np
from osgeo import gdal
import sys
from collections import defaultdict
import re
import os
import imageio
import json
sys.path.append('../')
from utils.patches import downPixelAggr, save_test_patches, save_random_patches, save_random_patches60, save_test_patches60


data_filename = '/MTD_MSIL1C.xml'

# sleep(randint(0, 20))

def readS2fromFile(data_file,
                   test_data=False,
                   roi_x_y=None,
                   save_prefix="../data/",
                   write_images=False,
                   run_60=False,
                   true_data=False):

    if run_60:
        select_bands = "B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12"
    else:
        select_bands = "B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12"

    raster = gdal.Open(data_file + data_filename)

    datasets = raster.GetSubDatasets()
    tenMsets = []
    twentyMsets = []
    sixtyMsets = []
    unknownMsets = []
    for (dsname, dsdesc) in datasets:
        if '10m resolution' in dsdesc:
            tenMsets += [ (dsname, dsdesc) ]
        elif '20m resolution' in dsdesc:
            twentyMsets += [ (dsname, dsdesc) ]
        elif '60m resolution' in dsdesc:
            sixtyMsets += [ (dsname, dsdesc) ]
        else:
            unknownMsets += [ (dsname, dsdesc) ]

    if roi_x_y:
        roi_x1, roi_y1, roi_x2, roi_y2 = [float(x) for x in re.split(',', args.roi_x_y)]

    # case where we have several UTM in the data set
    # => select the one with maximal coverage of the study zone
    utm_idx = 0
    utm = ""
    all_utms = defaultdict(int)
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    largest_area = -1
    # process even if there is only one 10m set, in order to get roi -> pixels
    for (tmidx, (dsname, dsdesc)) in enumerate(tenMsets + unknownMsets):
        ds = gdal.Open(dsname)
        if roi_x_y:
            tmxmin = max(min(roi_x1, roi_x2, ds.RasterXSize - 1), 0)
            tmxmax = min(max(roi_x1, roi_x2, 0), ds.RasterXSize - 1)
            tmymin = max(min(roi_y1, roi_y2, ds.RasterYSize - 1), 0)
            tmymax = min(max(roi_y1, roi_y2, 0), ds.RasterYSize - 1)
            # enlarge to the nearest 60 pixel boundary for the super-resolution
            tmxmin = int(tmxmin / 36) * 36
            tmxmax = int((tmxmax + 1) / 36) * 36 - 1
            tmymin = int(tmymin / 36) * 36
            tmymax = int((tmymax + 1) / 36) * 36 - 1
        else:
            tmxmin = 0
            tmxmax = ds.RasterXSize - 1
            tmymin = 0
            tmymax = ds.RasterYSize - 1

        area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
        current_utm = dsdesc[dsdesc.find("UTM"):]
        if area > all_utms[current_utm]:
            all_utms[current_utm] = area
        if area > largest_area:
            xmin, ymin, xmax, ymax = tmxmin, tmymin, tmxmax, tmymax
            largest_area = area
            utm_idx = tmidx
            utm = dsdesc[dsdesc.find("UTM"):]

    # convert comma separated band list into a list
    select_bands = [x for x in re.split(',',select_bands) ]

    print("Selected UTM Zone:".format(utm))
    print("Selected pixel region: xmin=%d, ymin=%d, xmax=%d, ymax=%d:" % (xmin, ymin, xmax, ymax))
    print("Selected pixel region: tmxmin=%d, tmymin=%d, tmxmax=%d, tmymax=%d:" % (tmxmin, tmymin, tmxmax, tmymax))
    print("Image size: width=%d x height=%d" % (xmax - xmin + 1, ymax - ymin + 1))

    if xmax < xmin or ymax < ymin:
        print("Invalid region of interest / UTM Zone combination")
        sys.exit(0)

    selected_10m_data_set = None
    if not tenMsets:
        selected_10m_data_set = unknownMsets[0]
    else:
        selected_10m_data_set = tenMsets[utm_idx]
    selected_20m_data_set = None
    for (dsname, dsdesc) in enumerate(twentyMsets):
        if utm in dsdesc:
            selected_20m_data_set = (dsname, dsdesc)
    # if not found, assume the listing is in the same order
    # => OK if only one set
    if not selected_20m_data_set: selected_20m_data_set = twentyMsets[utm_idx]
    selected_60m_data_set = None
    for (dsname, dsdesc) in enumerate(sixtyMsets):
        if utm in dsdesc:
            selected_60m_data_set = (dsname, dsdesc)
    if not selected_60m_data_set: selected_60m_data_set = sixtyMsets[utm_idx]

    ds10 = gdal.Open(selected_10m_data_set[0])
    ds20 = gdal.Open(selected_20m_data_set[0])
    ds60 = gdal.Open(selected_60m_data_set[0])

    def validate_description(description):
        m = re.match("(.*?), central wavelength (\d+) nm", description)
        if m:
            return m.group(1) + " (" + m.group(2) + " nm)"
        # Some HDR restrictions... ENVI band names should not include commas

        pos = description.find(',')
        return description[:pos] + description[(pos + 1):]

    def get_band_short_name(description):
        if ',' in description:
            return description[:description.find(',')]
        if ' ' in description:
            return description[:description.find(' ')]
        return description[:3]

    validated_10m_bands = []
    validated_10m_indices = []
    validated_20m_bands = []
    validated_20m_indices = []
    validated_60m_bands = []
    validated_60m_indices = []
    validated_descriptions = defaultdict(str)

    sys.stdout.write("Selected 10m bands:")
    for b in range(0, ds10.RasterCount):
        desc = validate_description(ds10.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(desc)
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_10m_bands += [shortname]
            validated_10m_indices += [b]
            validated_descriptions[shortname] = desc
    sys.stdout.write("\nSelected 20m bands:")
    for b in range(0, ds20.RasterCount):
        desc = validate_description(ds20.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(desc)
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_20m_bands += [shortname]
            validated_20m_indices += [b]
            validated_descriptions[shortname] = desc
    sys.stdout.write("\nSelected 60m bands:")
    for b in range(0, ds60.RasterCount):
        desc = validate_description(ds60.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(desc)
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_60m_bands += [shortname]
            validated_60m_indices += [b]
            validated_descriptions[shortname] = desc
    sys.stdout.write("\n")

    if validated_10m_indices:
        print("Loading selected data from: %s" % selected_10m_data_set[1])
        data10 = np.rollaxis(
            ds10.ReadAsArray(xoff=xmin, yoff=ymin, xsize=xmax - xmin + 1, ysize=ymax - ymin + 1, buf_xsize=xmax - xmin + 1,
                             buf_ysize=ymax - ymin + 1), 0, 3)[:, :, validated_10m_indices]

    if validated_20m_indices:
        print("Loading selected data from: %s" % selected_20m_data_set[1])
        data20 = np.rollaxis(
            ds20.ReadAsArray(xoff=xmin // 2, yoff=ymin // 2, xsize=(xmax - xmin + 1) // 2, ysize=(ymax - ymin + 1) // 2,
                             buf_xsize=(xmax - xmin + 1) // 2, buf_ysize=(ymax - ymin + 1) // 2), 0, 3)[:, :,
                 validated_20m_indices]

    if validated_60m_indices:
        print("Loading selected data from: %s" % selected_60m_data_set[1])
        data60 = np.rollaxis(
            ds60.ReadAsArray(xoff=xmin // 6, yoff=ymin // 6, xsize=(xmax - xmin + 1) // 6, ysize=(ymax - ymin + 1) // 6,
                             buf_xsize=(xmax - xmin + 1) // 6, buf_ysize=(ymax - ymin + 1) // 6), 0, 3)[:, :,
                 validated_60m_indices]

    # The percentile_data argument is used to plot superresolved and original data
    # with a comparable black/white scale
    def save_band(data, name, percentile_data=None):
        if percentile_data is None:
            percentile_data = data
        mi, ma = np.percentile(percentile_data, (1, 99))
        band_data = np.clip(data, mi, ma)
        band_data = (band_data - mi) / (ma - mi)
        imageio.imsave(save_prefix + name + ".png", band_data)  # img_as_uint(band_data))

    chan3 = data10[:, :, 0]
    vis = (chan3 < 1).astype(np.int)
    if np.sum(vis) > 0:
        print('The selected image has some blank pixels')
        # sys.exit()

    scale20 = 2
    scale60 = 6

    data10_gt = data10
    data20_gt = data20

    if not true_data:
        if run_60:
            data60_gt = data60
            data10_lr = downPixelAggr(data10_gt, SCALE=scale60)
            data20_lr = downPixelAggr(data20_gt, SCALE=scale60)
            data60_lr = downPixelAggr(data60_gt, SCALE=scale60)
        else:
            data10_lr = downPixelAggr(data10_gt, SCALE=scale20)
            data20_lr = downPixelAggr(data20_gt, SCALE=scale20)
            if scale20 > 2:
                data20_lr = downPixelAggr(data20_gt, SCALE=scale20//2)

    if data_file.endswith('/'):
        tmp = os.path.split(data_file)[0]
        data_file = os.path.split(tmp)[1]
    else:
        data_file = os.path.split(data_file)[1]
    print(data_file)

    if test_data:
        if run_60:
            out_per_image0 = save_prefix + 'test60/'
            out_per_image = save_prefix + 'test60/' + data_file + '/'
        else:
            out_per_image0 = save_prefix + 'test/'
            out_per_image = save_prefix + 'test/' + data_file + '/'
        if not os.path.isdir(out_per_image0):
            os.mkdir(out_per_image0)
        if not os.path.isdir(out_per_image):
            os.mkdir(out_per_image)

        print('Writing files for testing to:{}'.format(out_per_image))
        if run_60:
            save_test_patches60(data10_lr, data20_lr, data60_lr, out_per_image)
            with open(out_per_image + 'roi.json', 'w') as f:
                json.dump([tmxmin // scale60, tmymin // scale60, (tmxmax + 1) // scale60, (tmymax + 1) // scale60], f)
        else:
            save_test_patches(data10_lr, data20_lr, out_per_image)
            with open(out_per_image + 'roi.json', 'w') as f:
                json.dump([tmxmin // scale20, tmymin // scale20, (tmxmax+1) // scale20, (tmymax+1) // scale20], f)

        if not os.path.isdir(out_per_image + 'no_tiling/'):
            os.mkdir(out_per_image + 'no_tiling/')

        print("Now saving the whole image without tiling...")
        if run_60:
            np.save(out_per_image + 'no_tiling/' + 'data60_gt', data60_gt.astype(np.float32))
            np.save(out_per_image + 'no_tiling/' + 'data60', data60_lr.astype(np.float32))
        else:
            np.save(out_per_image + 'no_tiling/' + 'data20_gt', data20_gt.astype(np.float32))
            save_band(data10_lr[:, :, 0:3], '/test/' + data_file + '/RGB')
        np.save(out_per_image + 'no_tiling/' + 'data10', data10_lr.astype(np.float32))
        np.save(out_per_image + 'no_tiling/' + 'data20', data20_lr.astype(np.float32))

    elif write_images:
        print('Creating RGB images...')
        save_band(data10_lr[:, :, 0:3], '/raw/rgbs/' + data_file + 'RGB')
        save_band(data20_lr[:, :, 0:3], '/raw/rgbs/' + data_file + 'RGB20')

    elif true_data:
        out_per_image0 = save_prefix + 'true/'
        out_per_image = save_prefix + 'true/' + data_file + '/'
        if not os.path.isdir(out_per_image0):
            os.mkdir(out_per_image0)
        if not os.path.isdir(out_per_image):
            os.mkdir(out_per_image)

        print('Writing files for testing to:{}'.format(out_per_image))
        save_test_patches60(data10_gt, data20_gt, data60_gt, out_per_image, patchSize=384, border=12)

        with open(out_per_image + 'roi.json', 'w') as f:
            json.dump([tmxmin, tmymin, tmxmax+1, tmymax+1], f)

        if not os.path.isdir(out_per_image + 'no_tiling/'):
            os.mkdir(out_per_image + 'no_tiling/')

        print("Now saving the whole image without tiling...")
        np.save(out_per_image + 'no_tiling/' + 'data10', data10_gt.astype(np.float32))
        np.save(out_per_image + 'no_tiling/' + 'data20', data20_gt.astype(np.float32))
        np.save(out_per_image + 'no_tiling/' + 'data60', data60_gt.astype(np.float32))

    else:
        if run_60:
            out_per_image0 = save_prefix + 'train60/'
            out_per_image = save_prefix + 'train60/' + data_file + '/'
        else:
            out_per_image0 = save_prefix + 'train/'
            out_per_image = save_prefix + 'train/' + data_file + '/'
        if not os.path.isdir(out_per_image0):
            os.mkdir(out_per_image0)
        if not os.path.isdir(out_per_image):
            os.mkdir(out_per_image)
        print('Writing files for training to:{}'.format(out_per_image))
        if run_60:
            save_random_patches60(data60_gt, data10_lr, data20_lr, data60_lr, out_per_image)
        else:
            save_random_patches(data20_gt, data10_lr, data20_lr, out_per_image)

    print("Success.")


parser = argparse.ArgumentParser(description="Read Sentinel-2 data. The code was adapted from N. Brodu.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("data_file", help="An input Sentinel-2 data file. This can be either the original ZIP file, or the S2A[...].xml file in a SAFE directory extracted from that ZIP.")
parser.add_argument("--roi_x_y", default="",
    help="Sets the region of interest to extract as pixels locations on the 10m bands. Use this syntax: x_1,y_1,x_2,y_2. E.g. --roi_x_y \"2000,2000,3200,3200\"")
parser.add_argument("--test_data", default=False, action="store_true", help="Store test patches in a separate dir.")
parser.add_argument("--write_images", default=False, action="store_true", help="If set, write PNG images for the original and the superresolved bands, together with a composite rgb image (first three 10m bands), all with a quick and dirty clipping to 99%% of the original bands dynamic range and a quantization of the values to 256 levels.")
parser.add_argument("--save_prefix", default="../data/", help="If set, speficies the name of a prefix for all output files. Use a trailing / to save into a directory. The default of no prefix will save into the current directory. Example: --save_prefix result/")
parser.add_argument("--run_60", default=False, action="store_true", help="If set, it will create patches also from the 60m channels.")
parser.add_argument("--true_data", default=False, action="store_true", help="If set, it will create patches for S2 without GT. This option is not really useful here, please check the testing folder for predicting S2 images.")
args = parser.parse_args()

# args.data_file = sorted(glob.glob(data_prefix + 'S2*' + data_filename))

if __name__ == '__main__':
    # if type(args.data_file) is list:
    #     fileList = args.data_file
    #     for s2file in fileList:
    #         args.data_file = os.path.split(os.path.split(s2file)[0])[1]
    #         readS2fromFile(args.data_file,
    #                        args.test_data,
    #                        args.roi_x_y,
    #                        args.save_prefix,
    #                        args.write_images,
    #                        args.run_60,
    #                        args.true_data)
    # else:
    print('I will proceed with file {}'.format(args.data_file))
    readS2fromFile(args.data_file,
                   args.test_data,
                   args.roi_x_y,
                   args.save_prefix,
                   args.write_images,
                   args.run_60,
                   args.true_data)
