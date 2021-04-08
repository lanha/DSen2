# DSen2
Deep Sentinel-2

[Super-Resolution of Sentinel-2 Images: Learning a Globally Applicable Deep Neural Network](https://arxiv.org/abs/1803.04271)

Contact: Charis Lanaras, charis.lanaras@alumni.ethz.ch

## Requirements

- tensorflow-gpu (or tensorflow)
- keras
- nupmy
- scikit-image
- argparse
- imageio
- matplotlib (optional)
- GDAL >= 2.2 (optional)

## Training

See the detailed description in the `training` directory. Use the `--resume` option with your application related Sentinel-2 tiles to refine the provided network weights.

## Using the Trained Network

The network can be used directly on downloaded Sentinel-2 tiles. See details in the `s2_tiles_supres.py` file. An example follows:
```
 python s2_tiles_supres.py /path/to/S2A_MSIL1C_20161230T074322_N0204_R092_T37NCE_20161230T075722.SAFE/MTD_MSIL1C.xml /path/to/output_file.tif --roi_x_y "100,100,2000,2000"
```

Point to the `.xml` file of the uzipped S2 tile. You must also provide an output file -- consider using a `.tif` extension that is easily read by QGIS. If you want to also copy the high resolution (10m bands) you can do so, with the option `--copy_original_bands`.
To also predict the lowest resolution bands (60m) use the `--run_60` option.

## MATLAB Demo

The demo is also ported to MATLAB: `demoDSen2.m`. However, MATLAB 2018a or newer is needed to run. It utilizes the Neural Network toolbox that can be accelerated with the Parallel Computing Toolbox.

## Used Sentinel-2 tiles

The Sentinel-2 tiles used for training and testing are listed in:

- `S2_tiles_training.txt`
- `S2_tiles_testing.txt`

They can be downloaded from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/).

