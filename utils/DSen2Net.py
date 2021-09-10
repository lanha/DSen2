from __future__ import division
import keras
from keras.models import Model, Input
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Concatenate,
    Activation,
    Lambda,
    Add,
    BatchNormalization,
    ReLU,
)

keras.backend.set_image_data_format("channels_first")


def resBlock(x, channels, kernel_size, scale=0.1):
    """
    A residual block v = ResBlock(z, f ) is defined as a series of layers that operate on
    an input image z to generate an output z4, then adds that output to the input image as follows:
    z1 = conv(z, f ) #convolution (5a)
    z2 = max(z1, 0) #ReLU layer (5b)
    z3 = conv(z2, f ) #convolution (5c)
    z4 = lamda*z3 #residual scaling (5d)
    v = z4 + z #skip connection

    Conv2D: 2D convolution layer
    Activation function: A function that is added into an artificial neural network in order
    to help the network learn complex patterns in the data.
    Add function: Simply adding two layers

    """
    tmp = Conv2D(
        channels, kernel_size, kernel_initializer="he_uniform", padding="same",
    )(x)
    tmp = Activation("relu")(tmp)
    tmp = Conv2D(
        channels, kernel_size, kernel_initializer="he_uniform", padding="same",
    )(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def init(input_shape):
    """
    Input function: is used to instantiate a Keras tensor.
    """
    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    res = [input10, input20]
    channels = input_shape[1][0]
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
        res.append(input60)
        channels = input_shape[2][0]
    else:
        x = Concatenate(axis=1)([input10, input20])
    return x, res, channels


def aesrmodel(input_shape, n1=64):
    x, _input, channels = init(input_shape)

    level1_1 = Conv2D(n1, (3, 3), activation="relu", padding="same")(x)
    level2_1 = Conv2D(n1, (3, 3), activation="relu", padding="same")(level1_1)

    level2_2 = Conv2DTranspose(n1, (3, 3), activation="relu", padding="same")(level2_1)
    level2 = Add()([level2_1, level2_2])

    level1_2 = Conv2DTranspose(n1, (3, 3), activation="relu", padding="same")(level2)
    level1 = Add()([level1_1, level1_2])

    decoded = Conv2D(channels, (5, 5), activation="linear", padding="same",)(level1)

    model = Model(inputs=_input, outputs=decoded)
    return model


def srcnn(input_shape):
    f1 = 9
    f2 = 1
    f3 = 5

    n1 = 64
    n2 = 32
    x, _input, channels = init(input_shape)

    x = Conv2D(n1, (f1, f1), activation="relu", padding="same", name="level1")(x)
    x = Conv2D(n2, (f2, f2), activation="relu", padding="same", name="level2")(x)

    out = Conv2D(channels, (f3, f3), padding="same", name="output")(x)

    model = Model(inputs=_input, outputs=out)
    return model


def rednetsr(input_shape):
    def _build_layer_list(model):
        model_outputs = [layer.output for layer in model.layers]
        return model_outputs

    n_conv_layers = 15
    n_deconv_layers = 15
    n_skip = 2
    n = 32

    x, _input, channels = init(input_shape)

    for i in range(n_conv_layers):
        conv_idx = i + 1
        if conv_idx == 1:
            conv = Conv2D(n, (3, 3), activation="relu", padding="same")(x)
        else:
            conv = Conv2D(n, (3, 3), activation="relu", padding="same")(conv)

    encoded = conv
    encoder = Model(inputs=_input, outputs=encoded, name="encoder")
    # Create encoder layer and output lists
    encoder_outputs = _build_layer_list(encoder)

    # CREATE AUTOENCODER MODEL
    for i, skip in enumerate(reversed(encoder_outputs[len(_input) + 1 :])):

        deconv_idx = i + 1
        deconv_filters = n
        if deconv_idx == n_deconv_layers:
            deconv_filters = channels

        if deconv_idx == 1:
            deconv = Conv2DTranspose(
                deconv_filters, (3, 3), activation="relu", padding="same"
            )(encoded)
        else:
            deconv = Conv2DTranspose(
                deconv_filters, (3, 3), activation="relu", padding="same"
            )(deconv)

        if deconv_idx % n_skip == 0:
            deconv = Add()([deconv, skip])
            ReLU()(deconv)

    decoded = deconv  # (decoder_inputs)
    model = Model(inputs=_input, outputs=decoded)
    return model


def resnetsr(input_shape):
    def _residual_block(ip, _id):
        channel_axis = 1  # channels first
        init = ip

        x = Conv2D(n, (3, 3), padding="same", name="sr_res_conv_" + str(_id) + "_1")(ip)
        x = BatchNormalization(
            momentum=0.5, axis=channel_axis, name="sr_res_batchnorm_" + str(_id) + "_1"
        )(x)
        x = ReLU()(x)
        x = Conv2D(n, (3, 3), padding="same", name="sr_res_conv_" + str(_id) + "_2")(x)
        x = BatchNormalization(
            momentum=0.5, axis=channel_axis, name="sr_res_batchnorm_" + str(_id) + "_2"
        )(x)

        m = Add(name="sr_res_merge_" + str(_id))([x, init])
        return m

    n = 64
    x, _input, channels = init(input_shape)

    x0 = Conv2D(n, (9, 9), padding="same", name="sr_res_conv1")(x)
    x0 = ReLU()(x0)
    x = x0

    nb_residual = 16
    for i in range(nb_residual):
        x0 = _residual_block(x0, i + 1)

    x0 = Conv2D(filters=n, kernel_size=3, strides=1, padding="same")(x0)
    x0 = BatchNormalization(axis=1, momentum=0.5)(x0)
    x0 = Add()([x, x0])

    x0 = Conv2D(channels, (9, 9), padding="same", name="sr_res_conv_final")(x0)
    x0 = Activation("tanh")(x0)
    model = Model(inputs=_input, outputs=x0)

    return model


def s2model(input_shape, num_layers=32, feature_size=256):
    """
    This function contains the model architecture which contains a resBlock and 2 extra Conv2D layer.
    """

    x, _input, _ = init(input_shape)

    # Treat the concatenation
    x = Conv2D(
        feature_size,
        (3, 3),
        kernel_initializer="he_uniform",
        activation="relu",
        padding="same",
    )(x)

    for _ in range(num_layers):
        x = resBlock(x, feature_size, [3, 3])

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(
        input_shape[-1][0], (3, 3), kernel_initializer="he_uniform", padding="same",
    )(x)
    if len(input_shape) == 3:
        x = Add()([x, _input[2]])
        model = Model(inputs=_input, outputs=x)
    else:
        x = Add()([x, _input[1]])
        model = Model(inputs=_input, outputs=x)
    return model
