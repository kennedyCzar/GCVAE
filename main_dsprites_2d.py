

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

seed = 124
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.models import load_model
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from train_gcvae_2d_ds import train_gcvae as gcvae
#----import trainer for distributed training
#from train_gcvae_distrib import train_gcvae_distrib
from utils import plot_latent_space, compute_metric, model_saver, model_saver_dsprites
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 120
import seaborn as sns
#set random seed
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


#declare path
path = os.getcwd()


#import data
datatype = "dsprites"

batch_size = 64

#%%% Datasets

dataset_zip = np.load(os.path.join(path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), allow_pickle=True, encoding = 'latin1' )

imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]

#%%

x_train, x_test, y_train, y_test = train_test_split(imgs, latents_values, test_size = 0.999, random_state = 42)
x_test, _, y_test, _ = train_test_split(x_test, y_test, test_size = 0.992, random_state = 42)
N, L, M = x_train.shape
x_train = x_train.reshape(-1, L, M, 1).astype('float32')
x_test = x_test.reshape(-1, L, M, 1).astype('float32')

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

#test data
test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(buffer_size = 1024).batch(batch_size)


#%% Train Model

loss_index = 0
# vae_type = 'gcvae' #else infovae
inp_shape =  x_train.shape[1:]
num_features = inp_shape[0]

#the parameters are only to change fixed weights
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 10), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }


for ii in [2, 10]:
    lr = 1e-3
    epochs = 250000
    hidden_dim = 200
    latent_dims = ii
    loss_type = list(params.keys())[loss_index] #elbo -> 0; betavae -> 1; controlvae -> 2; infovae -> 3; gcvae -> 4
    archi_type = 'v2'
    #params
    distrib_type = 'b'
    beta, gamma = params[f'{loss_type}']
    mmd_typ = 'mmd' #['mmd', 'mah', 'mah_rkhs', 'mah_gcvae']
    save_model_arg = False
    save_model_after = 50000
    
    model = gcvae(inp_shape = inp_shape, 
                        num_features = num_features, 
                        hidden_dim = hidden_dim, 
                        latent_dim = latent_dims, 
                        batch_size = batch_size, 
                        beta = beta,
                        gamma = gamma,
                        dist = distrib_type,
                        vloss = loss_type,
                        lr = lr, 
                        epochs = epochs,
                        architecture = archi_type,
                        mmd_type = mmd_typ).fit(train_dataset, x_test, 
                                                datatype, stopping = False, 
                                                save_model = save_model_arg, 
                                                save_model_iter = save_model_after)
                    
    #save model....
    model_saver_dsprites(model,\
                x_test,\
                y_test,\
                hidden_dim,\
                latent_dims,\
                batch_size,\
                beta,\
                gamma,\
                distrib_type,\
                loss_type,\
                lr,\
                epochs,\
                archi_type,\
                mmd_typ,\
                datatype)
        
        
        










