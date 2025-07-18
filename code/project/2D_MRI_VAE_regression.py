##
# Usage: python 3D_MRI_VAE_regression.py ROI_x ROI_y ROI_z Size_x Size_y Size_z
# ROI_x,y,z, Size_x,y,z: Selecting a specific ROI box for analysis
# Reach out to http://cnslab.stanford.edu/ for data usage


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    UpSampling2D,
    Input,
    ZeroPadding2D,
    Lambda,
    Reshape,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import (
    MeanSquaredError,
    BinaryCrossentropy,
    MeanAbsoluteError,
)

# from keras.losses import mse, binary_crossentropy,mean_absolute_error
from tensorflow.keras.utils import plot_model
from tensorflow.keras.constraints import unit_norm, max_norm
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from sklearn.model_selection import StratifiedKFold
import nibabel as nib
import scipy as sp
import scipy.ndimage
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import sys
import argparse
import os
import glob
from PIL import Image


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def augment_by_transformation(data, age, n):
    augment_scale = 1

    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]
        m = n - raw_n
        for i in range(0, m):
            new_data = np.zeros((1, data.shape[1], data.shape[2], data.shape[3], 1))
            idx = np.random.randint(0, raw_n)
            new_age = age[idx]
            new_data[0] = data[idx].copy()
            new_data[0, :, :, :, 0] = sp.ndimage.interpolation.rotate(
                new_data[0, :, :, :, 0],
                np.random.uniform(-1, 1),
                axes=(1, 0),
                reshape=False,
            )
            new_data[0, :, :, :, 0] = sp.ndimage.interpolation.rotate(
                new_data[0, :, :, :, 0],
                np.random.uniform(-1, 1),
                axes=(0, 1),
                reshape=False,
            )
            new_data[0, :, :, :, 0] = sp.ndimage.shift(
                new_data[0, :, :, :, 0], np.random.uniform(-1, 1)
            )
            data = np.concatenate((data, new_data), axis=0)
            age = np.append(age, new_age)

        return data, age


def augment_by_noise(data, n, sigma):
    if n <= data.shape[0]:
        return data
    else:
        m = n - data.shape[0]
        for i in range(0, m):
            new_data = np.zeros((1, data.shape[1], data.shape[2], data.shape[3], 1))
            new_data[0] = data[np.random.randint(0, data.shape[0])]
            noise = np.clip(
                np.random.normal(
                    0, sigma, (data.shape[1], data.shape[2], data.shape[3], 1)
                ),
                -3 * sigma,
                3 * sigma,
            )
            new_data[0] += noise
            data = np.concatenate((data, new_data), axis=0)
        return data


def augment_by_flip(data):
    data_flip = np.flip(data, 1)
    data = np.concatenate((data, data_flip), axis=0)
    return data


####### Main Script #######
""" not cropping for ROI
min_x = int(sys.argv[1])    
min_y = int(sys.argv[2])  
min_z = int(sys.argv[3])  
patch_x = int(sys.argv[4])    
patch_y = int(sys.argv[5])    
patch_z = int(sys.argv[6]) 
"""

# dropout_alpha = float(sys.argv[7])
# L2_reg = float(sys.argv[8])

## CNN Parameters
dropout_alpha = 0.5
ft_bank_baseline = 16
latent_dim = 16
augment_size = 1000
L2_reg = 0.00
binary_image = False


def normalize_slice(slice_data):
    # Clip data to 0.5–99.5 percentile
    lower, upper = np.percentile(slice_data, [0.5, 99.5])
    slice_clipped = np.clip(slice_data, lower, upper)
    # Normalize all values to be between [0, 1]
    normalized = (slice_clipped - lower) / (
        upper - lower + 1e-8
    )  # prevent divion by zero
    return normalized


## Load data
# file_idx = np.loadtxt('./access.txt')
# age = np.loadtxt('./age.txt')
H = W = 192  # all 2D slices resized to 192x192


def get_ages(sub_id, dataset_num):
    sub_id = int(sub_id)
    if dataset_num == 1:
        df = pd.read_csv("/projectnb/ace-ig/ABIDE/Phenotypic_V1_0b.csv")
        age = df[(df["SUB_ID"] == sub_id)]["AGE_AT_SCAN"].values[0]
        if age > 21:
            return None
    elif dataset_num == 2:
        df = pd.read_csv(
            "/projectnb/ace-ig/ABIDE/ABIDEII_Composite_Phenotypic.csv",
            encoding="cp1252",
        )
        if df[(df["SUB_ID"] == sub_id)].empty:
            # print(f"[Warning] Subject {sub_id} not found in ABIDE 2 = Longitudinal Subject")
            return None
        age = df[(df["SUB_ID"] == sub_id)]["AGE_AT_SCAN "].values[
            0
        ]  # key has extra space at end for ABIDE II
        if age > 21:
            return None

    # print(f"Subject {sub_id} Age: {age}")
    return age


folder_paths = [
    "/projectnb/ace-ig/ABIDE/ABIDE_I_2D/axial",
    # "/projectnb/ace-ig/ABIDE/ABIDE_I_2D/coronal",
    # "/projectnb/ace-ig/ABIDE/ABIDE_I_2D/sagittal",
    # "/projectnb/ace-ig/ABIDE/ABIDE_II_2D/axial",
    # "/projectnb/ace-ig/ABIDE/ABIDE_II_2D/coronal",
    # "/projectnb/ace-ig/ABIDE/ABIDE_II_2D/sagittal"
]
image_list = []
age_list = []
for folder_path in folder_paths:
    print(f"Processing folder: {folder_path}")
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            # add subject's age
            subject_id = filename[2:7]
            dataset_num = 1 if "ABIDE_I_2D" in folder_path else 2
            age = get_ages(subject_id, dataset_num)
            if age is None:
                continue  # skip adding data if subject is not found in the appropriate dataset or age is above 21
            age_list.append(age)

            # add subject's scan
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("L")  # convert to grayscale
            img = img.resize(
                (192, 192)
            )  # Resize all images to 192x192 so output will match input perfectly after layers
            img = np.array(img)
            img = normalize_slice(img)
            image_list.append(img)

data = np.expand_dims(np.array(image_list), axis=-1)
ages = np.array(
    age_list
)  # order of data (subjects' scans) should correlate with their ages

## Cross Validation
print("Data size: ", data.shape)  # e.g. (num_subjects, 192, 192, 1)

skf = StratifiedKFold(n_splits=5, shuffle=True)
fake = np.zeros((data.shape[0]))
pred = np.zeros((ages.shape))

for train_idx, test_idx in skf.split(data, fake):

    train_data = data[train_idx]
    train_age = ages[train_idx]

    test_data = data[test_idx]
    test_age = ages[test_idx]

    # build encoder model
    input_r = Input(shape=(1,), name="ground_truth")
    input_image = Input(shape=(H, W, 1), name="input_image")
    feature = Conv2D(
        ft_bank_baseline, activation="relu", kernel_size=(3, 3), padding="same"
    )(input_image)
    feature = MaxPooling2D(pool_size=(2, 2))(feature)

    feature = Conv2D(
        ft_bank_baseline * 2, activation="relu", kernel_size=(3, 3), padding="same"
    )(feature)
    feature = MaxPooling2D(pool_size=(2, 2))(feature)

    feature = Conv2D(
        ft_bank_baseline * 4, activation="relu", kernel_size=(3, 3), padding="same"
    )(feature)
    feature = MaxPooling2D(pool_size=(2, 2))(feature)

    feature = Flatten()(feature)
    feature = Dropout(dropout_alpha)(feature)
    feature_dense = Dense(
        latent_dim * 4, activation="tanh", kernel_regularizer=regularizers.l2(L2_reg)
    )(feature)

    feature_z_mean = Dense(latent_dim * 2, activation="tanh")(feature_dense)
    z_mean = Dense(latent_dim, name="z_mean")(feature_z_mean)
    feature_z_log_var = Dense(latent_dim * 2, activation="tanh")(feature_dense)
    z_log_var = Dense(latent_dim, name="z_log_var")(feature_z_log_var)

    feature_r_mean = Dense(latent_dim * 2, activation="tanh")(feature_dense)
    r_mean = Dense(1, name="r_mean")(feature_r_mean)
    feature_r_log_var = Dense(latent_dim * 2, activation="tanh")(feature_dense)
    r_log_var = Dense(1, name="r_log_var")(feature_r_log_var)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    r = Lambda(sampling, output_shape=(1,), name="r")([r_mean, r_log_var])

    # instantiate encoder model
    encoder = Model(
        [input_image], [z_mean, z_log_var, z, r_mean, r_log_var, r], name="encoder"
    )
    # encoder = Model([input_image, input_r], [z_mean, z_log_var, z, r_mean, r_log_var, r], name='encoder') = old code with inputs not connected to outputs error due to input_r?
    encoder.summary()

    # build generator model
    generator_input = Input(shape=(1,), name="generator_input")
    # inter_z_1 = Dense(int(latent_dim/4), activation='tanh', kernel_constraint=unit_norm(), name='inter_z_1')(generator_input)
    # inter_z_2 = Dense(int(latent_dim/2), activation='tanh', kernel_constraint=unit_norm(), name='inter_z_2')(inter_z_1)
    # pz_mean = Dense(latent_dim, name='pz_mean')(inter_z_2)
    pz_mean = Dense(latent_dim, name="pz_mean", kernel_constraint=unit_norm())(
        generator_input
    )
    pz_log_var = Dense(1, name="pz_log_var", kernel_constraint=max_norm(0))(
        generator_input
    )
    # instantiate generator model
    generator = Model(generator_input, [pz_mean, pz_log_var], name="generator")
    generator.summary()

    # build decoder model
    latent_input = Input(shape=(latent_dim,), name="z_sampling")
    decoded = Dense(
        latent_dim * 2, activation="tanh", kernel_regularizer=regularizers.l2(L2_reg)
    )(latent_input)
    decoded = Dense(
        latent_dim * 4, activation="tanh", kernel_regularizer=regularizers.l2(L2_reg)
    )(decoded)
    # After 3 poolings (each /2), H and W go from 192 → 96 → 48 → 24
    downsampled_H = H // 8
    downsampled_W = W // 8
    flattened_size = downsampled_H * downsampled_W * ft_bank_baseline * 4
    decoded = Dense(
        flattened_size, activation="relu", kernel_regularizer=regularizers.l2(L2_reg)
    )(decoded)
    decoded = Reshape((downsampled_H, downsampled_W, ft_bank_baseline * 4))(decoded)

    decoded = Conv2D(ft_bank_baseline * 4, kernel_size=(3, 3), padding="same")(decoded)
    decoded = Activation("relu")(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(ft_bank_baseline * 2, kernel_size=(3, 3), padding="same")(decoded)
    decoded = Activation("relu")(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(ft_bank_baseline, kernel_size=(3, 3), padding="same")(decoded)
    decoded = Activation("relu")(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(1, kernel_size=(3, 3), padding="same")(decoded)
    if binary_image:
        outputs = Activation("sigmoid")(decoded)
    else:
        outputs = decoded

    # instantiate decoder model
    decoder = Model(latent_input, outputs, name="decoder")
    decoder.summary()

    # instantiate VAE model
    # 2 lines of old code commented out below?
    # pz_mean,pz_log_var = generator(encoder([input_image,input_r])[5])
    # outputs = decoder(encoder([input_image,input_r])[2])
    pz_mean, pz_log_var = generator(encoder([input_image])[5])
    outputs = decoder(encoder([input_image])[2])
    # vae = Model([input_image,input_r], [outputs, pz_mean,pz_log_var], name='vae_mlp')
    vae = Model([input_image], [outputs, pz_mean, pz_log_var], name="vae_mlp")

    if binary_image:
        reconstruction_loss = K.mean(
            binary_crossentropy(input_image, outputs), axis=[1, 2, 3]
        )
    else:
        reconstruction_loss = K.mean(
            mean_absolute_error(input_image, outputs), axis=[1, 2, 3]
        )

    kl_loss = (
        1
        + z_log_var
        - pz_log_var
        - K.square(z_mean - pz_mean) / K.exp(pz_log_var)
        - K.exp(z_log_var) / K.exp(pz_log_var)
    )
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    label_loss = 0.5 * K.square(r_mean - input_r) / K.exp(r_log_var) + 0.5 * r_log_var

    vae_loss = K.mean(reconstruction_loss + kl_loss + label_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")
    vae.summary()

"""
    #break
    # augment data
    train_data_aug,train_age_aug = augment_by_transformation(train_data,train_age,augment_size)
    print("Train data shape: ",train_data_aug.shape)

    # training
    vae.fit([train_data_aug,train_age_aug],
            verbose=2,
            batch_size=64,
            epochs = 80)

    vae.save_weights('vae_weights.h5')
    encoder.save_weights('encoder_weights.h5')
    generator.save_weights('generator_weights.h5')
    decoder.save_weights('decoder_weights.h5')

    # testing
    [z_mean, z_log_var, z, r_mean, r_log_var, r_vae] = encoder.predict([test_data,test_age],batch_size=64)
    pred[test_idx] = r_mean[:,0]

    filename = 'prediction_'+str(dropout_alpha)+'_'+str(L2_reg)+'.npy'
    np.save(filename,pred)

## CC accuracy
print("MSE: ", mean_squared_error(age,pred))
print("R2: ", r2_score(age, pred))

exit()

## Training on all data to learn a mega generative model
train_data_aug,train_age_aug = augment_by_transformation(data,age,augment_size)
vae.fit([data,age],
        verbose=2,
        batch_size=64,
        epochs = 80)

## Sample from latent space for visualizing the aging brain
#generator.load_weights('generator_weights.h5')
#decoder.load_weights('decoder_weights.h5')
# this range depends on the resulting encoded latent space
r = [-2, -1.5, -1, -0.5, 0, 1, 1.5, 2.5, 3.5, 4.5]

pz_mean = generator.predict(r,batch_size=64)
outputs = decoder.predict(pz_mean,batch_size=64)

for i in range(0,10):   
    array_img = nib.Nifti1Image(np.squeeze(outputs[i,:,:,:,0]),np.diag([1, 1, 1, 1]))
    
    filename = 'generated'+str(i)+'.nii.gz'
    nib.save(array_img,filename)

exit() """
