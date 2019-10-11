from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from sklearn.metrics import confusion_matrix

import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import os

np.set_printoptions(suppress=True, linewidth=120, precision=2)

name_of_particle = 'Egammac'

X_train = np.load("matrices/" + name_of_particle + "_train.npy", ).astype('float32')
y_train = np.load("matrices/" + name_of_particle + "_y_train.npy", ).astype('float32')
X_val = np.load("matrices/" + name_of_particle + "_val.npy", ).astype('float32')
y_val = np.load("matrices/" + name_of_particle + "_y_val.npy", ).astype('float32')
X_test = np.load("matrices/" + name_of_particle + "_test.npy", ).astype('float32')
y_test = np.load("matrices/" + name_of_particle + "_y_test.npy", ).astype('float32')
X_train = X_train[:, :-3]
X_val = X_val[:, :-3]
X_test = X_test[:, :-3]
_, V = X_train.shape
print("V:", V)
batch_size = 512

X_train_pos = X_train[np.where(y_train == 0)]
X_train_neg = X_train[np.where(y_train == 1)]
y_train_pos = y_train[np.where(y_train == 0)]
y_train_neg = y_train[np.where(y_train == 1)]

X_val_pos = X_val[np.where(y_val == 0)]
X_val_neg = X_val[np.where(y_val == 1)]
y_val_pos = y_val[np.where(y_val == 0)]
y_val_neg = y_val[np.where(y_val == 1)]

X_test_pos = X_test[np.where(y_test == 0)]
X_test_neg = X_test[np.where(y_test == 1)]
y_test_pos = y_test[np.where(y_test == 0)]
y_test_neg = y_test[np.where(y_test == 1)]

print(X_train_pos.shape[0] + X_train_neg.shape[0] == X_train.shape[0])

# test_ds_pos = tf.data.Dataset.from_tensor_slices((X_test_pos, y_test_pos)).batch(batch_size)  # .shuffle(1000)
# test_ds_neg = tf.data.Dataset.from_tensor_slices((X_test_neg, y_test_neg)).batch(batch_size)  # .shuffle(1000)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
#
# train_ds_pos = tf.data.Dataset.from_tensor_slices((X_train_pos, y_train_pos)).batch(batch_size)  # .shuffle(1000)
# train_ds_neg = tf.data.Dataset.from_tensor_slices((X_train_neg, y_train_neg)).batch(batch_size)  # .shuffle(1000)
# train_ds = (train_ds_pos, train_ds_neg)
#
# val_ds_pos = tf.data.Dataset.from_tensor_slices((X_val_pos, y_val_pos)).batch(batch_size)  # .shuffle(1000)
# val_ds_neg = tf.data.Dataset.from_tensor_slices((X_val_neg, y_val_neg)).batch(batch_size)  # .shuffle(1000)
# val_ds = (val_ds_pos, val_ds_neg)

# batch_size = 100
# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)  # .shuffle(1000)
# val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)  #.shuffle(1000)
# test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)  # .shuffle(1000)


def perf_measure(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    FSCORE = np.divide((2 * PPV * TPR), (PPV + TPR))

    return PPV, TPR, FSCORE, FNR, FPR, TNR


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# encoder, decoder = models
# x_test, y_test = data

# display a 2D plot of the digit classes in the latent space
# z_mean, _, _ = encoder.predict(x_test,
#                                batch_size=batch_size)

# z_sample = np.array([[xi, yi]])
# x_decoded = decoder.predict(z_sample)


# Model Setting:
original_shape = V  # X_train.shape[1]
original_dim = original_shape * original_shape
# original_dim = original_shape
latent_dim = 128
intermediate_dim = 128
final_dim = 64
epochs = 200
# epsilon_std = 1.0


# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=(original_shape, ), name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])  #

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_shape, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[0])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (X_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()

    if args.weights:

        vae.load_weights(args.weights)
        print("vae:", vae)
        mse = np.mean(np.power(X_test - vae.predict(X_test), 2), axis=1)

        df_error = pd.DataFrame({'reconstruction_error': mse, 'y_test': y_test}, )

        y_pred = (df_error.reconstruction_error > 0.5).tolist()
        y_pred = [1 if i == True else 0 for i in y_pred]

        PPV, TPR, FSCORE, FNR, FPR, TNR = perf_measure(y_true=y_test, y_pred=y_pred)
        print(PPV, TPR, FSCORE, FNR, FPR, TNR)
        print(" weights")

    else:
        # train the autoencoder
        vae.fit(X_train,
                shuffle=True
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, None))
        vae.save_weights("NN-ckecks/VAE" + name_of_particle + ".h5")

    # test_inputs = encoder.predict(X_test)[0]
    # print("test_inputs", test_inputs)
    #
    # test_outputs = decoder.predict(test_inputs)
    # print("test_outputs:", test_outputs)

    mse = np.mean(np.power(X_test - vae.predict(X_test), 2), axis=1)
    df_error = pd.DataFrame({'reconstruction_error': mse, 'y_test': y_test}, )

    y_pred = (df_error.reconstruction_error > 0.5).tolist()
    y_pred = [1 if i == True else 0 for i in y_pred]

    PPV, TPR, FSCORE, FNR, FPR, TNR = perf_measure(y_true=y_test, y_pred=y_pred)
    print(PPV, TPR, FSCORE, FNR, FPR, TNR)
    print(" ")

