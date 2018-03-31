# DSen2
Deep Sentinel-2
[Super-Resolution of Sentinel-2 Images: Learning a Globally Applicable Deep Neural Network](https://arxiv.org/abs/1803.04271)


## Requirements

- tensorflow-gpu (or tensorflow)
- keras
- nupmy
- scikit-image
- argparse
- matplotlib (optional)
- GDAL >= 2.2 (optional)

## Training

Scripts implementing training are coming on 15th April 2018

## Using Trained Network

The network can be used directly on downloaded Sentinel-2 tiles.

## MATLAB Demo

The demo is also ported to MATLAB `demoDSen2.m`. However, MATLAB 2017b or newer is needed to run. It utilizes the Neural Network toolbox that can be accelerated with the Parallel Computing Toolbox.

## Used Sentinel-2 tiles

The Sentinel-2 tiles used for training and testing are listed in:

- `S2_tiles_training.txt`
- `S2_tiles_testing.txt`

They can be downloaded from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/).

