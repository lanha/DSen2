from __future__ import division
import numpy as np
import datetime
import glob
import time
import argparse
import os
import sys
import matplotlib as mpl
mpl.use('Agg')
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from keras.utils import plot_model
import keras.backend as K

sys.path.append('../')
from utils.patches import recompose_images, OpenDataFilesTest, OpenDataFiles
from utils.DSen2Net import s2model

K.set_image_data_format('channels_first')

# Define file prefix for new training, must be 7 characters of this form:
model_nr = 's2_038_'
SCALE = 2000
lr = 1e-4


path = '../data/'
if not os.path.isdir(path):
    os.mkdir(path)
out_path = '../data/network_data/'
if not os.path.isdir(out_path):
    os.mkdir(out_path)


class PlotLosses(Callback):
    def __init__(self, model_nr, lr):
        self.model_nr = model_nr
        self.lr = lr

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.i = 0
        self.x = []
        self.filename = out_path + self.model_nr + '_lr_{:.1e}.txt'.format(self.lr)
        open(self.filename, 'w').close()

    def on_epoch_end(self, epoch, logs=None):
        import matplotlib.pyplot as plt
        plt.ioff()

        lr = float(K.get_value(self.model.optimizer.lr))
        # data = np.loadtxt("training.log", skiprows=1, delimiter=',')
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.x.append(self.i)
        self.i += 1
        try:
            with open(self.filename, 'a') as self.f:
                self.f.write('Finished epoch {:5d}: loss {:.3e}, valid: {:.3e}, lr: {:.1e}\n'
                             .format(epoch, logs.get('loss'), logs.get('val_loss'), lr))

            if epoch > 500:
                plt.clf()
                plt.plot(self.x[475:], self.losses[475:], label='loss')
                plt.plot(self.x[475:], self.val_losses[475:], label='val_loss')
                plt.legend()
                plt.xlabel('epochs')
                # plt.waitforbuttonpress(0)
                plt.savefig(out_path + self.model_nr + '_loss4.png')
            elif epoch > 250:
                plt.clf()
                plt.plot(self.x[240:], self.losses[240:], label='loss')
                plt.plot(self.x[240:], self.val_losses[240:], label='val_loss')
                plt.legend()
                plt.xlabel('epochs')
                # plt.waitforbuttonpress(0)
                plt.savefig(out_path + self.model_nr + '_loss3.png')
            elif epoch > 100:
                plt.clf()
                plt.plot(self.x[85:], self.losses[85:], label='loss')
                plt.plot(self.x[85:], self.val_losses[85:], label='val_loss')
                plt.legend()
                plt.xlabel('epochs')
                # plt.waitforbuttonpress(0)
                plt.savefig(out_path + self.model_nr + '_loss2.png')
            elif epoch > 50:
                plt.clf()
                plt.plot(self.x[50:], self.losses[50:], label='loss')
                plt.plot(self.x[50:], self.val_losses[50:], label='val_loss')
                plt.legend()
                plt.xlabel('epochs')
                # plt.waitforbuttonpress(0)
                plt.savefig(out_path + self.model_nr + '_loss1.png')
            else:
                plt.clf()
                plt.plot(self.x[0:], self.losses[0:], label='loss')
                plt.plot(self.x[0:], self.val_losses[0:], label='val_loss')
                plt.legend()
                plt.xlabel('epochs')
                # plt.waitforbuttonpress(0)
                plt.savefig(out_path + self.model_nr + '_loss0.png')
        except IOError:
            print('Network drive unavailable.')
            print(datetime.datetime.now().time())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SupResS2.')
    parser.add_argument('--predict', action='store', dest='predict_file', help='Predict.')
    parser.add_argument('--resume', action='store', dest='resume_file', help='Resume training.')
    parser.add_argument('--true', action='store_true', help='Use true scale data. No simulation or different resolutions.')
    parser.add_argument('--run_60', action='store_true', help='Whether to run a 60->10m network. Default 20->10m.')
    parser.add_argument('--deep', action='store_true', help='.')
    parser.add_argument('--path', help='Path of data. Only relevant if set.')
    args = parser.parse_args()

    if args.path is not None:
        path = args.path

    # input_shape = ((4,32,32),(6,16,16))
    if args.run_60:
        input_shape = ((4, None, None), (6, None, None), (2, None, None))
    else:
        input_shape = ((4, None, None), (6, None, None))
    # create model
    if args.deep:
        model = s2model(input_shape, num_layers=32, feature_size=256)
        batch_size = 8
    else:
        model = s2model(input_shape, num_layers=6, feature_size=128)
        batch_size = 128
    print('Symbolic Model Created.')

    nadam = Nadam(lr=lr,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-8,
                  schedule_decay=0.004)
                  # clipvalue=0.000005)

    model.compile(optimizer=nadam, loss='mean_absolute_error', metrics=['mean_squared_error'])
    print('Model compiled.')
    model.count_params()
    # model.summary()

    if args.predict_file:
        if args.true:
            folder = 'true/'
            border = 12
        elif args.run_60:
            folder = 'test60/'
            border = 12
        else:
            folder = 'test/'
            border = 4
        model_nr = args.predict_file[-20:-13]
        print('Changing the model number to: {}'.format(model_nr))
        model.load_weights(args.predict_file)
        print("Predicting using file: {}".format(args.predict_file))
        fileList = [os.path.basename(x) for x in sorted(glob.glob(path + folder + '*SAFE'))]
        for dset in fileList:
            start = time.time()
            print("Timer started.")
            print("Predicting: {}.".format(dset))
            train, image_size = OpenDataFilesTest(path + folder + dset, args.run_60, SCALE, args.true)
            prediction = model.predict(train,
                                       batch_size=8,
                                       verbose=1)
            prediction_file = model_nr + '-predict'
            # np.save(path + 'test/' + dset + '/' + prediction_file + 'pat', prediction * SCALE)
            images = recompose_images(prediction, border=border, size=image_size)
            print('Writing to file...')
            np.save(path + folder + dset + '/' + prediction_file, images * SCALE)
            end = time.time()
            print('Elapsed time: {}.'.format(end - start))
        sys.exit(0)

    if args.resume_file:
        print("Will resume from the weights {}".format(args.resume_file))
        model.load_weights(args.resume_file)
        model_nr = args.resume_file[-20:-13]
        print('Changing the model number to: {}'.format(model_nr))

    else:
        print('Model number is {}'.format(model_nr))
        plot_model(model, to_file=out_path + model_nr+'model.png', show_shapes=True, show_layer_names=True)

        model_yaml = model.to_yaml()
        with open(out_path + model_nr + "model.yaml", 'w') as yaml_file:
            yaml_file.write(model_yaml)

    filepath = out_path + model_nr + 'lr_{:.0e}.hdf5'.format(lr)
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='auto')
    plot_losses = PlotLosses(model_nr, lr)
    LRreducer = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=5,
                                  verbose=1,
                                  epsilon=1e-6,
                                  cooldown=20,
                                  min_lr=1e-5)

    callbacks_list = [checkpoint, plot_losses, LRreducer]

    print('Loading the training data...')
    train, label, val_tr, val_lb = OpenDataFiles(path, args.run_60, SCALE)

    print('Training starts...')

    model.fit(x=train,
              y=label,
              batch_size=batch_size,
              epochs=8 * 1024,
              verbose=1,
              callbacks=callbacks_list,
              validation_split=0.,
              validation_data=(val_tr, val_lb),
              shuffle=True,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              validation_steps=None)

