
#import dependencies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

seed = 42
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import tensorflow_datasets as tfds
from train_gcvae_2d import train_gcvae as gcvae
#----import trainer for distributed training
from train_gcvae_distrib import train_gcvae_distrib
from sklearn.model_selection import train_test_split
from utils import (plot_latent_space, compute_metric, 
                   model_saver, model_saver_2d,
                   model_saver_scriterion_2d)
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
datatype = "mnist"

batch_size = 64
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
N, L, M = x_train.shape
x_train = x_train.reshape(-1, L, M, 1).astype('float32') / x_train.max()
x_train, _ = train_test_split(x_train, test_size = .99, random_state = 42)
x_test = x_test.reshape(-1, L, M, 1).astype('float32') / x_train.max()
x_test, _ = train_test_split(x_test, test_size = .99, random_state = 42)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

#test data
test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(buffer_size = 1024).batch(batch_size)


#%%
loss_index = 4
# vae_type = 'gcvae' #else infovae
inp_shape =  x_train.shape[1:]
num_features = inp_shape[0]

#the parameters are only to change fixed weights
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 5), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

for ii in [2, 10, 50, 100, 200, 500]:
    for ij in ['mmd', 'mah', 'mah_gcvae']:
        lr = 1e-3
        epochs = 250000
        hidden_dim = 200
        latent_dims = ii
        loss_type = list(params.keys())[loss_index] #elbo -> 0; controlvae -> 1; infovae -> 2; gcvae -> 3
        archi_type = 'v2'
        #params
        distrib_type = 'b'
        beta, gamma = params[f'{loss_type}']
        mmd_typ = ij #['mmd', 'mah', 'mah_rkhs', 'mah_gcvae']
        save_model_arg = False
        save_model_after = 10
        #-----------------------------Ignore this section if no stopping criterion is needed------------------------------------------
        stop_criterion = 'useStop' #'useStop' or 'igstop'  --> useStop --> Use stopping criterion or igstop --> Ignore stopping
        stopping = True if stop_criterion == 'useStop' else False
        pid_a = True if stopping == True else False
        pid_b = True if stopping == True else False
        #-----------------------------------------End stopping criterion params ------------------------------------------------------
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
                                                    datatype, stopping = stopping,
                                                    save_model = save_model_arg,
                                                    save_model_iter = save_model_after,
                                                    pid_a = pid_a,
                                                    pid_b = pid_b,
                                                    epsilon_a = 1e-5,
                                                    epsilon_b = 1e-4)
            
        if stop_criterion == 'igstop':
           #save model....
           model_saver_2d(model,\
                       x_test,\
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
        else:
            model_saver_scriterion_2d(model,\
                    x_test,\
                    hidden_dim,\
                    latent_dims,\
                    batch_size,\
                    beta,\
                    gamma,\
                    distrib_type,\
                    loss_type,\
                    lr,\
                    model.epoch,\
                    archi_type,\
                    mmd_typ,\
                    datatype,
                    stop_criterion)
            
            
            
            
            
            
