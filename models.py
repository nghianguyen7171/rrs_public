from common import *
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import random
import torch
import sys
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from tqdm import tqdm
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix

from keras.layers import Input, merge, LSTM, Dense, SimpleRNN, Masking, Bidirectional, Dropout, concatenate, Embedding, TimeDistributed, multiply, add, dot, Conv2D
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from keras import regularizers, callbacks
from keras.layers.core import *
from keras.models import *

# params
seed = 42
num_folds = 5
scoring = "roc_auc"
batch_size = 1028

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    tf.random.set_seed(seed)

seed_everything(seed)

#Loss
from tensorflow.keras import backend as K

smooth  = 1.
epsilon = 1e-7

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# dice_coef

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# dice_coef_loss
def dice_coef_multi(y_true, y_pred):
    y_true_f = K.flatten(y_true[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])

    y_true_sum = K.sum(K.cast(y_true_f > epsilon, dtype="float32"))
    y_pred_sum = K.sum(y_pred_f)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
# dice_coef_multi

def dice_coef_multi_loss(y_true, y_pred):
    return 1.0 - dice_coef_multi(y_true, y_pred)
# dice_coef_multi_loss

def mean_acc(y_true, y_pred):
    y_true_label = K.argmax(y_true, axis = 1)
    y_pred_label = K.argmax(y_pred, axis = 1)
    cm = tf.math.confusion_matrix(y_true_label, y_pred_label)
    cm_norm = cm / tf.reshape(tf.reduce_sum(cm, axis = 1), (-1, 1))
    zero_pos = tf.where(tf.math.is_nan(cm_norm))
    n_zero   = tf.shape(zero_pos)[0]
    cm_norm  = tf.tensor_scatter_nd_update(cm_norm, zero_pos, tf.zeros(n_zero, dtype=tf.double))
    mean_acc_val = tf.reduce_mean(tf.linalg.diag_part(cm_norm))
    return mean_acc_val


metrics = ["acc", dice_coef_multi, mean_acc, tf.keras.metrics.AUC()]
loss_fn = ["categorical_crossentropy", dice_coef_multi_loss] # "categorical_crossentropy",
optimizer_fn = tf.keras.optimizers.Adam(learning_rate=0.0001)
weights = None


# TVAE model

# Create a sampling layer

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# TVAE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

tf.keras.backend.clear_session()


def build_VAE(input_shape, latent_dim):
    # Encoder
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(100, activation='tanh', return_sequences=True)(encoder_inputs)
    x = layers.LSTM(50, activation='tanh', return_sequences=True)(x)
    x = layers.LSTM(25, activation='tanh')(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    # Clf decoder
    decoder1 = layers.Dense(8, activation='relu')(z)
    decoder1 = layers.Dropout(0.2)(decoder1)

    decoder1 = layers.Dense(64, activation='relu')(decoder1)
    decoder1 = layers.Dropout(0.2)(decoder1)

    decoder1 = layers.Dense(32, activation='relu')(decoder1)
    decoder1 = layers.Dropout(0.2)(decoder1)

    decoder1 = layers.Dense(16, activation='relu')(decoder1)
    decoder1 = layers.Dropout(0.1)(decoder1)

    decoder_out = layers.Dense(2, activation='sigmoid')(decoder1)
    VAE_clf = tf.keras.Model(inputs=encoder_inputs, outputs=decoder_out)

    VAE_clf.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return VAE_clf


# FCN model

#tf.keras.backend.clear_session()


def build_FCN(

        # input_shape = (400,),
        input_shape,
        n_dims=[192, 96, 48, 24],
        n_dropouts=[0.1, 0.1, 0.1],

        output_dims=2,
        output_activation="sigmoid",

        # metrics = [tf.keras.metrics.AUC, dice_coef],
        metrics=["acc", dice_coef_multi, mean_acc, tf.keras.metrics.AUC()],
        loss_fn=["categorical_crossentropy", dice_coef_multi_loss],  # "categorical_crossentropy",

        optimizer_fn=tf.keras.optimizers.Adam(learning_rate=0.0001),

        weights=None,
):
    inputs = []
    outputs = []

    input_ = keras.Input(shape=input_shape)
    inputs.append(input_)

    x = input_

    for n_dim, n_dropout in zip(n_dims, n_dropouts):
        x = keras.layers.Dense(n_dim, activation="relu")(x)
        x = keras.layers.Dropout(n_dropout)(x)

    x = keras.layers.Dense(output_dims, activation=output_activation)(x)
    outputs.append(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # if weights is not None: model.load_weight(weights)

    model.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )
    # model.compile(
    # optimizer='adam',
    # loss='categorical_crossentropy',
    # metrics='accuracy',
    # )
    return model

# model = build_model()
# model.summary()

# Multi head transformer

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


# tf.keras.backend.clear_session()

def build_trans(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        # metrics = [tf.keras.metrics.AUC, dice_coef],
        metrics=["acc", dice_coef_multi, mean_acc, tf.keras.metrics.AUC()],
        loss_fn=["categorical_crossentropy", dice_coef_multi_loss],  # "categorical_crossentropy",

        optimizer_fn=tf.keras.optimizers.Adam(learning_rate=0.0001),

        weights=None):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(2, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
    )
    return model

# model = build_trans(
#   input_shape = x_train.shape[1:],
#   head_size=256,
#   num_heads=4,
#   ff_dim=4,
#   num_transformer_blocks=4,
#   mlp_units=[128],
#   mlp_dropout=0.4,
#   dropout=0.25)


# model.summary()

# Kwon et al. model¶

def build_kwon_RNN(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(64, activation='tanh', dropout=0.2, return_sequences=True)(inputs)
    x = layers.LSTM(32, activation='tanh', dropout=0.2, return_sequences=True)(x)
    x = layers.LSTM(16, dropout=0.1)(x)
    out = layers.Dense(2, activation='sigmoid')(x)
    model_kwon = tf.keras.Model(inputs=inputs, outputs=out)
    model_kwon.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return model_kwon


# Attention blocks used in DEWS
def attention_block(inputs_1, num):
    # num is used to label the attention_blocks used in the model

    # Compute eij i.e. scoring function (aka similarity function) using a feed forward neural network
    v1 = Dense(10, use_bias=True)(inputs_1)
    v1_tanh = Activation('relu')(v1)
    e = Dense(1)(v1_tanh)
    e_exp = Lambda(lambda x: K.exp(x))(e)
    sum_a_probs = Lambda(lambda x: 1 / K.cast(K.sum(x, axis=1, keepdims=True) + K.epsilon(), K.floatx()))(e_exp)
    a_probs = multiply([e_exp, sum_a_probs], name='attention_vec_' + str(num))

    context = multiply([inputs_1, a_probs])
    context = Lambda(lambda x: K.sum(x, axis=1))(context)

    return context


# Shamount et al
def build_shamount_Att_BiLSTM(input_shape):
    inputs = keras.Input(shape=input_shape)
    enc = Bidirectional(
        LSTM(100, return_sequences=True, kernel_regularizer=regularizers.l2(0.05), kernel_initializer='random_uniform'),
        'ave')(inputs)
    dec = attention_block(enc, 1)
    dec_out = Dense(5, activation='relu')(dec)
    dec_drop = Dropout(0.2)(dec_out)
    out = Dense(2, activation='sigmoid')(dec_drop)
    model_shamount = tf.keras.Model(inputs=inputs, outputs=out)
    model_shamount.compile(
        optimizer=optimizer_fn,
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=True,
    )

    return model_shamount