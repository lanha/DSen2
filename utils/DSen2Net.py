from __future__ import division
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, Concatenate, Activation, Lambda, Add


def resBlock(x, channels, kernel_size, scale=0.1):
    tmp = Conv2D(
        channels,
        kernel_size,
        data_format="channels_first",
        kernel_initializer="he_uniform",
        padding="same",
    )(x)
    tmp = Activation("relu")(tmp)
    tmp = Conv2D(
        channels,
        kernel_size,
        data_format="channels_first",
        kernel_initializer="he_uniform",
        padding="same",
    )(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def init(input_shape):
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

    level1_1 = Conv2D(
        n1, (3, 3), data_format="channels_first", activation="relu", padding="same"
    )(x)
    level2_1 = Conv2D(
        n1, (3, 3), data_format="channels_first", activation="relu", padding="same"
    )(level1_1)

    level2_2 = Conv2DTranspose(
        n1, (3, 3), data_format="channels_first", activation="relu", padding="same"
    )(level2_1)
    level2 = Add()([level2_1, level2_2])

    level1_2 = Conv2DTranspose(
        n1, (3, 3), data_format="channels_first", activation="relu", padding="same"
    )(level2)
    level1 = Add()([level1_1, level1_2])

    decoded = Conv2D(
        channels,
        (5, 5),
        data_format="channels_first",
        activation="linear",
        padding="same",
    )(level1)

    model = Model(inputs=_input, outputs=decoded)
    return model


def s2model(input_shape, num_layers=32, feature_size=256):

    x, _input, _ = init(input_shape)

    # Treat the concatenation
    x = Conv2D(
        feature_size,
        (3, 3),
        data_format="channels_first",
        kernel_initializer="he_uniform",
        activation="relu",
        padding="same",
    )(x)

    for _ in range(num_layers):
        x = resBlock(x, feature_size, [3, 3])

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(
        input_shape[-1][0],
        (3, 3),
        data_format="channels_first",
        kernel_initializer="he_uniform",
        padding="same",
    )(x)
    if len(input_shape) == 3:
        x = Add()([x, _input[2]])
        model = Model(inputs=_input, outputs=x)
    else:
        x = Add()([x, _input[1]])
        model = Model(inputs=_input, outputs=x)
    return model
