from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers.core import Flatten, Dense, Dropout
import h5py

def get_model():
    model = Sequential()

    # Input = 128 x 128 x 3 = 49152
    model.add(Conv2D(8, (3, 3), name='Conv1-1'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 126 x 126 x 8
    model.add(Conv2D(16, (3, 3), strides=(2, 2), name='Conv2-2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 62 x 62 x 16
    model.add(Conv2D(32, (3, 3), strides=(2, 2), name='Conv3-2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 30 x 30 x 32
    model.add(Conv2D(64, (3, 3), strides=(2, 2), name='Conv4-2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 14 x 14 x 64
    model.add(Conv2D(128, (3, 3), strides=(2, 2), name='Conv5-1'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 6 x 6 x 128 = 4608
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(160))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    return model

def load_model_weights(model, weights_path):
    print('Loading model.')
    f = h5py.File(weights_path)
    print(list(f.attrs))
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    return model

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name].get_output()
    return layer