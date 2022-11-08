from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

seed = 124
import os
import random
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7})
plt.rc('text', usetex=False)
plt.rc('font', family='monospace')
plt.rcParams['figure.dpi'] = 90
plt.rcParams["figure.figsize"] = (20,3)
from utils import plot_latent_space, compute_metric, model_saver

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
datatype = "mnist"
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
    }

lls = ['vae', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
lr = 1e-3
epochs = 250000
hidden_dim = 200
latent_dims = 2
archi_type = 'v1'
#params
distrib_type = 'b'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
save_model_arg = True
save_model_after = 2
    
path  = '...'
file_path = []
for i in list(params.keys()):
    if not i == 'gcvae':
        file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"))
    else:
        for j in mmd_typ:
            file_path.append(os.path.join(path, f"{distrib_type}/{i}/{datatype}/latent_{latent_dims}/{epochs}/{j}/loggers.npy"))
            
dt = {f"{x.upper()}": np.load(os.path.join(path, f'{y}'), allow_pickle=True).ravel()[0] for (x, y)\
      in zip(lls, file_path)}
    
import matplotlib as mpl
mpl.style.use('ggplot')
fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(top=0.89,
                    bottom=0.172,
                    left=0.031,
                    right=0.993,
                    hspace=0.5,
                    wspace=0.318)
ax = ax.ravel()

lss = ['reconstruction', 'kl_div']

for w, (i, j) in zip(range(len(dt)), dt.items()):
    indx = 0
    for q, (n, m) in zip(range(7),  j.items()):
        if n in lss:
            ax[indx].plot(range(epochs), m, label = f"{i}", lw=1)
            if n == 'kl_div':
                ax[indx].set_title('KL(q(z|x)||p(z))',y=-0.1,pad=-14)
            ax[indx].set_title(f'{n.upper()}')
            ax[indx].set_xlabel('epochs')
            # ax[indx].set_ylabel(f'{n}')
            ax[indx].legend()
            indx += 1
    # ax[w].set_title(f'{i}')

            
#%% For ...Latent space...

datatype = "mnist"
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
epochs = 250000
hidden_dim = 200
latent_dims = 2
archi_type = 'v2'
#params
distrib_type = 'b'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']

path  = '...'
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

#%% for Kenneth...plot 2-D img representation...
datatype = "mnist"
distrib_type = 'b'
mmd_typ = ['mmd', 'mah', 'mah_gcvae']
save_model_arg = True
lls = ['elbo', 'betavae', 'controlvae', 'infovae',
       'gcvae-i', 'gcvae-ii', 'gcvae-iii']
lr = 1e-3
epochs = 250000
hidden_dim = 200
latent_dims = 2
archi_type = 'v2'
params = { #beta, gamma
            'elbo': (1, 0),
            'betavae': ((1, 16), 0),
            'controlvae': (0, 0),
            'infovae': (0, 500),
            'gcvae': (1, 1), #not necessarily useful inside algo
            }
plt.rcParams.update({'font.size': 10})
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams['figure.dpi'] = 100

    
latent_dims = 2
path  = '...'
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
fig.subplots_adjust(
            top=0.945,
            bottom=0.202,
            left=0.023,
            right=0.977,
            hspace=0.5,
            wspace=0.076
    )

ax = ax.ravel()
plt.style.use('grayscale')
for w, (p, q) in zip(range(len(dt)), dt.items()):
    # mu, sigma, z = j.encoder.predict(x_test, batch_size = batch_size)
    n = 4
    digit_size = 28
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
    ax[w].imshow(figure, cmap="Greys_r")
    # plt.savefig(f'{p}.png')
    if p == 'ELBO':
        ax[w].set_title('VAE',y=-0.1,pad=-14)
    else:
        ax[w].set_title(f'{p}',y=-0.1,pad=-14)
    
# plt.savefig(f'reconstruction_')


#%% Disentanglement Metric....| results.npy
datatype = 'dsprites'

mmd_typ = ['mmd', 'mah', 'mah_gcvae']#'mah_gcvae'

epochs = 250000
latent_dims = 10
path  = '...'
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

datatype = '3dshapes'

mmd_typ = ['mmd', 'mah', 'mah_gcvae']#'mah_gcvae'
epochs = 250000
latent_dims = 6
path  = '...'
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

#%% All metric together

print('-'*200)
print("|\t\t Model \t\t|\t\t Factor-VAE \t\t|\t\t MIG \t\t|\t\t Modularity \t\t|\t\t Jemmig \t\t|\t\t Reconstruction \t\t|\t\t KL divergence |")
print('-'*200)
for (i, j), (p, q) in zip(metrics.items(), logger.items()):
    print(f"|\t {i} \t\t|\t\t {j['factorvae_score_mu']:.3f} +/- {j['factorvae_score_sigma']:.3f} \t\t|"+
          f"\t\t {j['mig_score_mu']:.4f} \t\t|\t\t {j['modularity']:.3f} \t\t|\t\t {j['jemmig']:.3f} \t\t\t|\t\t\t {q['reconstruction'][-1]:.3f} \t\t\t|"+
          f"\t\t\t {q['kl_div'][-1]:.4f} \t\t\t|")
print('-'*200)


    
#%% reconstruction plot

def convert_to_display(samples, channel = 1):
    cnt, height, width = int(np.floor(np.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    print(cnt, height, width)
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt, channel])
    return samples
random_sample = np.random.randn(100, 2)
encoded = dt['GCVAE-II'].encoder.predict(x_test)
decoded = dt['GCVAE-II'].decoder.predict(encoded[0])


plt.imshow(convert_to_display(decoded))
plt.show()
         
   
#%%
num_imgs = 6
rand = np.random.randint(1, x_test.shape[0]-6) 

xtestsample = x_test[rand:rand+num_imgs]
rows = 2 # defining no. of rows in figure
cols = 3 # defining no. of colums in figure
cell_size = 1.5
f = plt.figure(figsize=(cell_size*cols,cell_size*rows*2)) # defining a figure 
f.tight_layout()
for i in range(rows):
    for j in range(cols): 
        f.add_subplot(rows*2,cols, (2*i*cols)+(j+1)) # adding sub plot to figure on each iteration
        plt.imshow(xtestsample[i*cols + j]) 
        plt.axis("off")
    
    for j in range(cols): 
        f.add_subplot(rows*2,cols,((2*i+1)*cols)+(j+1)) # adding sub plot to figure on each iteration
        plt.imshow(x_decoded[i*cols + j])
        plt.axis("off")

plt.show()



#%%

def generate():
    n=5
    channels = 1
    width = 64
    height = 64
    figure = np.zeros((width *2 , height * 10, channels))
    for k in range(2):
        for l in range(10):
            z_sample =np.random.rand(20)
            z_out=np.array([z_sample])
            x_decoded = dt['GCVAE-II'].decoder.predict(encoded[-1])
            digit = x_decoded[0].reshape(width, height, channels)
            figure[k * width: (k + 1) * width,
                    l * height: (l + 1) * height] = digit

    plt.figure(figsize=(10, 10))
#Reshape for visualization
    fig_shape = np.shape(figure)
    figure = figure.reshape((fig_shape[0], fig_shape[1],1))
    plt.show()
    
generate()
    
    
