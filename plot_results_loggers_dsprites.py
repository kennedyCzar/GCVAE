from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

seed = 124
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 100
plt.rcParams["figure.figsize"] = (20,3)
from utils import plot_latent_space, compute_metric, model_saver

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


#declare path
path = os.getcwd()


#import data
datatype = "3dshapes"

batch_size = 64

#%%% Datasets

dataset_zip = np.load(os.path.join(path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), allow_pickle=True, encoding = 'latin1' )

imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]

#%%

x_train, x_test, y_train, y_test = train_test_split(imgs, latents_values, test_size = 0.33, random_state = 0)
N, L, M = x_train.shape
x_train = x_train.reshape(-1, L*M, 1).astype('float32')
x_test = x_test.reshape(-1, L*M, 1).astype('float32')

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

#test data
test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(buffer_size = 1024).batch(batch_size)



#%% For Kenneth...Visualize metrics..
datatype = "dsprites"
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

lls = ['elbo', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
lr = 1e-3
epochs = 20
hidden_dim = 512
latent_dims = 2
archi_type = 'v2'
#params
distrib_type = 'b'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
save_model_arg = True
save_model_after = 2
    
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/loggers.npy"))
            
dt = {f"{x.upper()}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}
    
fig, ax = plt.subplots(1, len(dt))
fig.subplots_adjust(top=0.89,
                    bottom=0.172,
                    left=0.031,
                    right=0.993,
                    hspace=0.5,
                    wspace=0.318)
ax = ax.ravel()
        
for w, (i, j) in zip(range(len(dt)), dt.items()):
    for q, (n, m) in zip(range(7),  j.items()):
        ax[q].plot(range(epochs), m, label = f"{i}", lw=.7)
        ax[q].set_title(f'{n.upper()}')
        ax[q].set_xlabel('epochs')
        ax[q].set_ylabel(f'{n}')
        ax[q].legend()
    # ax[w].set_title(f'{i}')
        
plt.savefig('dsprite_metric_2.png')
        
#%% For kenneth...Latent space...

params = { #(beta, gamma)
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

lls = ['elbo', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
lr = 1e-3
epochs = 20
hidden_dim = 512
latent_dims = 2
archi_type = 'v1'
#params
distrib_type = 'b'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
save_model_arg = True
save_model_after = 2
    

file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/model.h5"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/model.h5"))
            
dt = {f"{x.upper()}": load_model(os.path.join(path, f'{y}')) for (x, y)\
      in zip(lls, file_path)}
    
# plot_latent_space(model, n= 10)

fig, ax = plt.subplots(1, len(dt))
fig.subplots_adjust(top=0.89,
                    bottom=0.172,
                    left=0.031,
                    right=0.993,
                    hspace=0.5,
                    wspace=0.318)
ax = ax.ravel()
        
for w, (i, j) in zip(range(len(dt)), dt.items()):
    mu, sigma, z = j.encoder.predict(x_test, batch_size = batch_size)
    ax[w].scatter(z[:, 0], z[:, 1], label = f"{i}", lw=.7)
    ax[w].set_title(f'{i.upper()}')
    ax[w].set_xlabel('z[0]')
    ax[w].set_ylabel('z[1]')
    ax[w].legend()


plt.savefig('dsprite_latent_2.png')

#%% for Kenneth...plot 2-D img representation...
datatype = "dsprites"

plt.rcParams.update({'font.size': 8})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 100
params = { #(beta, gamma)
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }
distrib_type = 'b'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
lls = ['elbo', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
latent_dims = 2
epochs = 250000


file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/model.h5"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/model.h5"))
            
dt = {f"{x.upper()}": load_model(os.path.join(path, f'{y}')) for (x, y)\
      in zip(lls, file_path)}
    
    
fig, ax = plt.subplots(1, len(dt))
fig.subplots_adjust(top=0.948,
                    bottom=0.221,
                    left=0.009,
                    right=0.991,
                    hspace=0.5,
                    wspace=0.018)
ax = ax.ravel()
        
for w, (p, q) in zip(range(len(dt)), dt.items()):
    # mu, sigma, z = j.encoder.predict(x_test, batch_size = batch_size)
    n = 5
    digit_size = 64
    scale = 1.0
    figsize = 10
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = q.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit
    
    #plt.figure(figsize=(figsize, figsize))
    ax[w].axis('off')
    ax[w].set_xlabel(f'({chr(w+97)}) {p}')
    ax[w].imshow(figure, cmap="Greys_r", interpolation='nearest')
    ax[w].set_title(f'({chr(w+97)}) {p}',y=-0.1,pad=-14)
    

# plt.savefig('dsprite_reconstruction_2.png')

#%% Disentanglement Metric....| results.npy

epochs = 250000
latent_dims = 6
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/results.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/results.npy"))
            
metrics = {f"{x.upper()}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}


print('-'*120)
print("|\t\t Model \t\t|\t\t Factor-VAE \t\t|\t\t MIG \t\t|\t\t Modularity \t\t|\t\t Jemmig \t\t|")
print('-'*120)
for i, j in metrics.items():
    print(f"|\t {i} \t\t|\t\t {j['factorvae_score_mu']:.3f} +/- {j['factorvae_score_sigma']:.3f} \t\t|"+
          f"\t\t {j['mig_score_mu']:.4f} \t\t|\t\t {j['modularity']:.3f} \t\t|\t\t {j['jemmig']:.3f} \t\t|")
print('-'*120)

#%% Total losses, Reconstruction and KL-divergence...


epochs = 250000
latent_dims = 6
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/loggers.npy"))
            
logger = {f"{x.upper()}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}


print('-'*120)
print("|\t\t Model \t\t|\t\t Total loss \t\t|\t\t Reconstruction \t\t|\t\t KL divergence |")
print('-'*120)
for i, j in logger.items():
    print(f"|\t {i} \t\t\t|\t\t\t {j['elbo'][-1]:.3f} \t\t\t|\t\t\t {j['reconstruction'][-1]:.3f} \t\t\t|"+
          f"\t\t\t {j['kl_div'][-1]:.4f} \t\t\t|")
print('-'*120)

    
    
    
    
    
    


