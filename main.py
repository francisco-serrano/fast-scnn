from tensorflow.python import keras
import tensorflow as tf


def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True):
    if conv_type == 'ds':
        x = keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)
    else:
        x = keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides=strides)(inputs)

    x = keras.layers.BatchNormalization()(x)

    if relu:
        x = keras.activations.relu(x)

    return x


input_layer = keras.layers.Input(shape=(2048, 1024, 3), name='input_layer')

lds_layer = conv_block(input_layer, 'conv', kernel=32, kernel_size=(3, 3), strides=(2, 2))
lds_layer = conv_block(lds_layer, 'ds', kernel=48, kernel_size=(3, 3), strides=(2, 2))
lds_layer = conv_block(lds_layer, 'ds', kernel=64, kernel_size=(3, 3), strides=(2, 2))


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = keras.layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)
    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)
    return x


gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t=6, strides=2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t=6, strides=2, n=3)
gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t=6, strides=1, n=3)


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    w = 64
    h = 32
    for bin_size in bin_sizes:
        x = keras.layers.AveragePooling2D(pool_size=(w // bin_size, h // bin_size),
                                          strides=(w // bin_size, h // bin_size))(input_tensor)
        x = keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        x = keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(x)
        concat_list.append(x)
    return keras.layers.concatenate(concat_list)


gfe_layer = pyramid_pooling_block(gfe_layer, [2, 4, 6, 8])
ff_layer1 = conv_block(lds_layer, 'conv', 128, (1, 1), padding='same', strides=(1, 1), relu=False)

ff_layer2 = keras.layers.UpSampling2D((4, 4))(gfe_layer)
ff_layer2 = keras.layers.DepthwiseConv2D(128, strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
ff_layer2 = keras.layers.BatchNormalization()(ff_layer2)
ff_layer2 = keras.activations.relu(ff_layer2)
ff_layer2 = keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)(ff_layer2)

ff_final = keras.layers.add([ff_layer1, ff_layer2])
ff_final = keras.layers.BatchNormalization()(ff_final)
ff_final = keras.activations.relu(ff_final)

classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1), name='DSConv1_classifier')(
    ff_final)
classifier = keras.layers.BatchNormalization()(classifier)
classifier = keras.activations.relu(classifier)

classifier = keras.layers.SeparableConv2D(128, (3, 3), padding='same', strides=(1, 1), name='DSConv2_classifier')(
    classifier)
classifier = keras.layers.BatchNormalization()(classifier)
classifier = keras.activations.relu(classifier)

classifier = conv_block(classifier, 'conv', 19, (1, 1), strides=(1, 1), padding='same', relu=True)

classifier = keras.layers.Dropout(0.3)(classifier)

classifier = keras.layers.UpSampling2D((8, 8))(classifier)
classifier = keras.activations.softmax(classifier)

optimizer = keras.optimizers.SGD(momentum=0.9, lr=0.045)

fast_scnn = keras.Model(inputs=input_layer, outputs=classifier, name='Fast_SCNN')
fast_scnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
fast_scnn.summary()

keras.utils.plot_model(fast_scnn, show_layer_names=True, show_shapes=True)