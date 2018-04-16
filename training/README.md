## Training

The training is implemented with the following procedure:

1) Create the random patches used for training:
```
python create_patches.py [location of *.SAFE/ dir]
```
In the given setting the command has to be run for each Sentinel-2 tile, by specifying the location of the unziped S2 directory (with the *.SAFE/ extension). Hint: use GNU parallel to speed this step up.

Moreover, use the switch ``--run_60`` to create patches for DSen2_60 (6x network). This applies to all the steps below, whenever the 6x network is used. By default the 2x network is used.

2) Create the validation split for training with `create_random.py`. Default is 10% of the training data. Take care to place the `.npy` file in the correct train folder.

3) Run `supres_train.py`. You can resume a model from previous weights with `--resume`. You can use the `--deep` switch to train a VDSen2 (very deep network), this applies whenever a very deep network is used.
Caution: You must change the model number for every new run, otherwise it will overwrite previous runs.

Note: The network was trained by loading all the data on a 64GB RAM. In case the training data don't fit in the RAM, a generator can be used.

## Testing

Testing can be preformed as follows:

1) Create patches tiles that are able to reconstruct the full image after prediction, with a small overlap for border artifacts.
Specify the ``--test_data`` option:
```
python create_patches.py [location of *.SAFE/ dir] --test_data
```

2) Use a model with weights to predict the images.
```
python supres_train.py --predict [location of the model weights]
```
