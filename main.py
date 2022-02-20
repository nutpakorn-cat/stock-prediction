from __future__ import absolute_import

# data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# model

import numpy as np
import theano.tensor as T
from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers.recurrent import Recurrent

# build
import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

# train
import time
import sys

directory = 'dataset'
filename = 'set-index.csv'

df = pd.read_csv(directory + '/' + filename).replace('\,', '', regex=True)
raw_data = np.transpose(np.array(df[['Open']]))[0][::-1]

prepared_data = np.zeros((1, len(raw_data)), dtype=np.float32)
prepared_data[0] = raw_data


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig("output.png")


class ITOSFM(Recurrent):

    def __init__(self, output_dim, freq_dim, hidden_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(ITOSFM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.states = [None, None, None, None, None]
        self.W_i = self.init((input_dim, self.hidden_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.hidden_dim,), name='{}_b_i'.format(self.name))

        self.W_ste = self.init((input_dim, self.hidden_dim),
                               name='{}_W_ste'.format(self.name))
        self.U_ste = self.inner_init((self.hidden_dim, self.hidden_dim),
                                     name='{}_U_ste'.format(self.name))
        self.b_ste = self.forget_bias_init((self.hidden_dim,),
                                           name='{}_b_ste'.format(self.name))

        self.W_fre = self.init((input_dim, self.freq_dim),
                               name='{}_W_fre'.format(self.name))
        self.U_fre = self.inner_init((self.hidden_dim, self.freq_dim),
                                     name='{}_U_fre'.format(self.name))
        self.b_fre = self.forget_bias_init((self.freq_dim,),
                                           name='{}_b_fre'.format(self.name))

        self.W_c = self.init((input_dim, self.hidden_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.hidden_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.hidden_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.hidden_dim, self.hidden_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.hidden_dim,), name='{}_b_o'.format(self.name))

        self.U_a = self.inner_init((self.freq_dim, 1),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.hidden_dim,), name='{}_b_a'.format(self.name))

        self.W_p = self.init((self.hidden_dim, self.output_dim),
                             name='{}_W_p'.format(self.name))
        self.b_p = K.zeros((self.output_dim,), name='{}_b_p'.format(self.name))

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_ste, self.U_ste, self.b_ste,
                                  self.W_fre, self.U_fre, self.b_fre,
                                  self.W_o, self.U_o, self.b_o,
                                  self.U_a, self.b_a,
                                  self.W_p, self.b_p]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_initial_states(self, x):

        init_state_h = K.zeros_like(x)
        init_state_h = K.sum(init_state_h, axis=1)
        reducer_s = K.zeros((self.input_dim, self.hidden_dim))
        reducer_f = K.zeros((self.hidden_dim, self.freq_dim))
        reducer_p = K.zeros((self.hidden_dim, self.output_dim))
        init_state_h = K.dot(init_state_h, reducer_s)

        init_state_p = K.dot(init_state_h, reducer_p)

        init_state = K.zeros_like(init_state_h)
        init_freq = K.dot(init_state_h, reducer_f)

        init_state = K.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = K.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = K.cast_to_floatx(0.)

        initial_states = [init_state_p, init_state_h,
                          init_state_S_re, init_state_S_im, init_state_time]
        return initial_states

    def step(self, x, states):
        p_tm1 = states[0]
        h_tm1 = states[1]
        S_re_tm1 = states[2]
        S_im_tm1 = states[3]
        time_tm1 = states[4]
        B_U = states[5]
        B_W = states[6]
        frequency = states[7]

        x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
        x_ste = K.dot(x * B_W[0], self.W_ste) + self.b_ste
        x_fre = K.dot(x * B_W[0], self.W_fre) + self.b_fre
        x_c = K.dot(x * B_W[0], self.W_c) + self.b_c
        x_o = K.dot(x * B_W[0], self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1 * B_U[0], self.U_i))

        ste = self.inner_activation(x_ste + K.dot(h_tm1 * B_U[0], self.U_ste))
        fre = self.inner_activation(x_fre + K.dot(h_tm1 * B_U[0], self.U_fre))

        ste = K.reshape(ste, (-1, self.hidden_dim, 1))
        fre = K.reshape(fre, (-1, 1, self.freq_dim))
        f = ste * fre

        c = i * self.activation(x_c + K.dot(h_tm1 * B_U[0], self.U_c))

        time = time_tm1 + 1

        omega = K.cast_to_floatx(2*np.pi) * time * frequency
        re = T.cos(omega)
        im = T.sin(omega)

        c = K.reshape(c, (-1, self.hidden_dim, 1))

        S_re = f * S_re_tm1 + c * re
        S_im = f * S_im_tm1 + c * im

        A = K.square(S_re) + K.square(S_im)

        A = K.reshape(A, (-1, self.freq_dim))
        A_a = K.dot(A * B_U[0], self.U_a)
        A_a = K.reshape(A_a, (-1, self.hidden_dim))
        a = self.activation(A_a + self.b_a)

        o = self.inner_activation(x_o + K.dot(h_tm1 * B_U[0], self.U_o))

        h = o * a
        p = K.dot(h, self.W_p) + self.b_p

        return p, [p, h, S_re, S_im, time]

    def get_constants(self, x):
        constants = []
        constants.append([K.cast_to_floatx(1.) for _ in range(6)])
        constants.append([K.cast_to_floatx(1.) for _ in range(7)])
        array = np.array(
            [float(ii)/self.freq_dim for ii in range(self.freq_dim)])
        constants.append([K.cast_to_floatx(array)])

        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "fre_dim": self.fre_dim,
                  "hidden_dim": self.hidden_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        base_config = super(ITOSFM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def load_data(step):
    # load data from the data file
    day = step
    data = prepared_data
    data = data[:, :]
    gt_test = data[:, day:]
    # data normalization
    max_data = np.max(data, axis=1)
    min_data = np.min(data, axis=1)
    max_data = np.reshape(max_data, (max_data.shape[0], 1))
    min_data = np.reshape(min_data, (min_data.shape[0], 1))
    data = (2 * data - (max_data + min_data)) / (max_data - min_data)
    # dataset split
    train_split = int(round(0.8 * data.shape[1]))
    val_split = int(round(0.9 * data.shape[1]))

    x_train = data[:, :train_split]
    y_train = data[:, day:train_split+day]
    x_val = data[:, :val_split]
    y_val = data[:, day:val_split+day]
    x_test = data[:, :-day]
    y_test = data[:, day:]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    return [x_train, y_train, x_val, y_val, x_test, y_test, gt_test, max_data, min_data]

# build the model


def build_model(layers, freq, learning_rate):
    model = Sequential()

    model.add(ITOSFM(
        input_dim=layers[0],
        hidden_dim=layers[1],
        output_dim=layers[2],
        freq_dim=freq,
        return_sequences=True))

    start = time.time()

    rms = keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss="mse", optimizer="rmsprop")

    print("Compilation Time : ", time.time() - start)
    return model


np.set_printoptions(threshold=sys.maxsize)

step = 1
hidden_dim = 10
freq_dim = 10
learning_rate = 0.01
iteration_size = 4000
snapshot_size = 20

global_start_time = time.time()

print('> Loading data... ')

X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = load_data(
    step)
train_len = X_train.shape[1]
val_len = X_val.shape[1]-X_train.shape[1]
test_len = X_test.shape[1]-X_val.shape[1]

print('> Data Loaded. Compiling...')
model = build_model([1, hidden_dim, 1], freq_dim, learning_rate)
best_error = np.inf
best_epoch = 0

for ii in range(int(iteration_size/snapshot_size)):
    model.fit(
        X_train,
        y_train,
        batch_size=50,
        nb_epoch=snapshot_size,
        validation_split=0)

    num_iter = str(snapshot_size * (ii+1))
    model.save_weights(
        './snapshot/weights{}.hdf5'.format(num_iter), overwrite=True)

    predicted = model.predict(X_train)
    train_error = np.sum((predicted[:, :, 0] - y_train[:, :, 0])
                         ** 2) / (predicted.shape[0] * predicted.shape[1])

    print(num_iter, ' training error ', train_error)

    predicted = model.predict(X_val)
    val_error = np.sum(
        (predicted[:, -val_len:, 0] - y_val[:, -val_len:, 0])**2) / (val_len * predicted.shape[0])

    print(' val error ', val_error)

    if(val_error < best_error):
        best_error = val_error
        best_iter = snapshot_size * (ii+1)

print('Training duration (s) : ', time.time() - global_start_time)
print('best iteration ', best_iter)
print('smallest error ', best_error)

print '> Predicting... '
predicted = model.predict(X_test)
# denormalization
prediction = (predicted[:, :, 0] * (max_data -
                                    min_data) + (max_data + min_data))/2

error = np.sum((prediction[:, -test_len:] - gt_test[:, -
                test_len:])**2) / (test_len * prediction.shape[0])
print 'The mean square error is: %f' % error

for ii in range(0, len(prediction)):
    plot_results(prediction[ii, -test_len:], gt_test[ii, -test_len:])
