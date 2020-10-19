from __future__ import print_function
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda, Multiply
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

def binary_accuracy(y_true,y_pred):
  return K.greater(y_true*y_pred, 0)

def build_dataset(x, y, k):
  x_pos = x[y[:,0]==1]
  x_neg = x[y[:,0]==-1]
  y_pos = y[y[:,0]==1]
  y_neg = y[y[:,0]==-1]
  idx = [np.random.permutation(x.shape[0]) for i in range(k)]
  idx_pos = [np.random.permutation(x_pos.shape[0]) for i in range(k)]
  idx_neg = [np.random.permutation(x_neg.shape[0]) for i in range(k)]

  x = [x[idx[i]] for i in range(k)]
  y = [y[idx[i]] for i in range(k)]
  x = np.concatenate(x,axis=-1)
  y = np.prod(y,axis=0)

  x = x.reshape(x.shape[0], -1)
  x = x.astype('float32')
  x /= 255

  return (x,y)

# ================ PARAMETERS =================
batch_size = 128
epochs = 20
k = 3
q = 512
# =============================================

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_shape = (img_rows*img_cols*k,)

y_train = 1-2*(y_train[:,None].astype('int8') % 2)
y_test = 1-2*(y_test[:,None].astype('int8') % 2)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


for architecture in ['ReLU', 'GaLU', 'RF_gaussian', 'RF_ReLU']:
  inputs = keras.Input(shape=input_shape)
  if architecture == 'ReLU':
    h = Dense(q, activation='relu')(inputs)
    outputs = Dense(1)(h)
  elif architecture == 'RF_ReLU':
    h = Dense(q, activation='relu', trainable=False)(inputs)
    outputs = Dense(1)(h)
  elif architecture == 'RF_gaussian':
    h = Dense(q, activation=tf.cos, trainable=False,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=4/np.sqrt(input_shape[0])),
                bias_initializer=tf.keras.initializers.RandomUniform(minval=0., maxval=2*np.pi))(inputs)
    outputs = Dense(1)(h)
  elif architecture == 'GaLU':
    h = Dense(q)(inputs)
    gate = Dense(q, trainable=False, bias_initializer='zeros')(inputs)
    gate = Lambda(lambda x : K.cast(K.greater(x, 0), 'float32'))(gate)
    h = Multiply()([gate, h])
    outputs = Dense(1)(h)

  model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

  model.compile(loss=keras.losses.hinge,
                optimizer=keras.optimizers.Adadelta(),
                metrics=[binary_accuracy])

  scores = []
  for e in range(epochs):
    cur_x_train, cur_y_train = build_dataset(x_train, y_train, k)
    cur_x_test, cur_y_test = build_dataset(x_test, y_test, k)

    model.fit(cur_x_train, cur_y_train,
              batch_size=batch_size,
              epochs=1,
              verbose=1)

    score = model.evaluate(cur_x_test, cur_y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    scores.append(score[1])

  with open('experiment_k%d_q%d_%s_%d.pickle' % (k, q, architecture, epochs), 'wb') as f:
    pickle.dump(scores, f)