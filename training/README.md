## Training

The training is implemented with the following procedure:

1) Create the patches used for training:
```
python create_patches.py [location of *.SAFE/ file]
```
In the given setting the command has to be run for each given Sentinel-2 tile, by givining the location of the unziped S2 file (with the *.SAFE/ extension). Hint: use GNU parallel to speed this step up.
Moreover, use the switch ``--run_60`` to create patches for DSen2_60 (6x network). This applies to all the steps below.

2) Create the validation split for training with `create_randon`. Default is 10% of the training data. Take care to place the `.npy` file in the train folder.

3) Run `supres_train.py`, change the model number for every new run. You can resume a model from previous weights with `--resume`. You can use the `--deep` switch to train a VDSen2 (very deep network), this applies whenever a very deep network is used.


## Testing

Testing used as follows:

1) Create the overlapping and tiled testing patches to predict a full image with the ``--test_data``:
```
python create_patches.py [location of *.SAFE/ file] --test_data
```

2) Use a model with weights to predict the images.
```
python supres_train.py --predict [location of the model weights]
```
