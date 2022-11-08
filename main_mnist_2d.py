
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
from train_gcvae_2d import train_gcvae as gcvae #change this to train_gcvae_2d if you are working with GCVAE
#----import trainer for distributed training
#from train_gcvae_distrib import train_gcvae_distrib
from sklearn.model_selection import train_test_split
from utils import plot_latent_space, compute_metric, model_saver, model_saver_2d
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

for ii in [2, 10]:
    lr = 1e-3
    epochs = 250000
    hidden_dim = 200
    latent_dims = ii
    loss_type = list(params.keys())[loss_index] #elbo -> 0; controlvae -> 1; infovae -> 2; gcvae -> 3
    archi_type = 'v2'
    #params
    distrib_type = 'b'
    beta, gamma = params[f'{loss_type}']
    mmd_typ = 'mmd' #['mmd', 'mah', 'mah_rkhs', 'mah_gcvae']
    save_model_arg = False
    save_model_after = 10
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
    # distrib_type = 'v2/b' #remove this if you are working with original GCVAE version..to save in b file
    # #save model....
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



#%%
# #save model
# tf.saved_model.save(model.model, f'model.h5') #save model
# #load model
# mdl = load_model('model.h5')

# #save attributes of the trainer...
# np.save("mod.npy", model.model)


# # # #%%
# z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size)

# fig = plt.figure(figsize=(5, 4))
# plt.scatter(z[:, 0], z[:, 1], c = y_test, edgecolor="black", s = 10)
# plt.xlabel('z[0]')
# plt.ylabel('z[1]')
# plt.title('infoVAE: ($1-\\alpha- \\beta){E_{q}} + \\beta KL + MMD (z_{t+1}, \\hat{z}_t)$ with stepwise $\\hat{z}$')

 #%%
# #%% Plot intermediate latent space...For 5 epochs
# fig, ax = plt.subplots(1, 5)
# for ii in np.arange(1, 5+1):
#     z_t = model.int_z[f'epoch{ii}']
#     ax[ii-1].scatter(z_t[:, 0], z_t[:, 1], c = y_test, s = 1)
#     ax[ii-1].set_xlabel('z[0]')
#     ax[ii-1].set_ylabel('z[1]')
#     ax[ii-1].set_title(f'epoch {ii}')
    

# #%% Plot intermediate latent space...For more than 5epochs

# fig, ax = plt.subplots(2, len(np.arange(model.epoch))//2)
# for ii in np.arange(1, len(np.arange(model.epoch))//2+1):
#     z_t = model.int_z[f'epoch{ii}']
#     ax[0][ii-1].scatter(z_t[:, 0], z_t[:, 1], c = y_test, s = 1)
#     ax[0][ii-1].set_xlabel('z[0]')
#     ax[0][ii-1].set_ylabel('z[1]')
#     ax[0][ii-1].set_title(f'epoch {ii}')
# for ii in np.arange(len(np.arange(model.epoch))//2+1, len(np.arange(model.epoch))+1):
#     z_t = model.int_z[f'epoch{ii}']
#     ax[1][ii-len(np.arange(model.epoch))//2-1].scatter(z_t[:, 0], z_t[:, 1], c = y_test, s = 1)
#     ax[1][ii-len(np.arange(model.epoch))//2-1].set_xlabel('z[0]')
#     ax[1][ii-len(np.arange(model.epoch))//2-1].set_ylabel('z[1]')
#     ax[1][ii-len(np.arange(model.epoch))//2-1].set_title(f'epoch {ii}')

# #%% plot losses and other

# fig, ax = plt.subplots(1, 4)
# ax[0].plot(np.arange(model.epoch), model.ELBO, label = 'ELBO', marker = '+', lw = 1.)
# ax[0].plot(np.arange(model.epoch), model.RECON_LOSS, label = 'Reconstruction loss', marker = '+', lw = 1.)
# ax[0].set_xlabel('epochs')
# ax[0].set_ylabel('Losses')
# ax[1].plot(np.arange(model.epoch), model.ALPHA, label = '$\\alpha$', marker = '+', lw = 1.)
# # ax[1].plot(np.arange(model.epoch), model.BETA, label = '$\\beta$', marker = '+', lw = 1.)
# ax[1].set_xlabel('epochs')
# ax[1].set_ylabel('$\\alpha$, $\\beta$')
# ax[2].plot(np.arange(model.epoch), model.MMD, label = 'MD', marker = '+', lw = 1.)
# ax[2].set_xlabel('epochs')
# ax[2].set_ylabel('Mahalanobis distance')
# ax[3].plot(np.arange(model.epoch), model.KL_DIV, label = 'KL', marker = '+', lw = 1.)
# ax[3].set_xlabel('epochs')
# ax[3].set_ylabel('KL-Divergence')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()


#%% Plot reconstruction

# plot_latent_space(model.model, n= 5)

#%%visualize D>2 dimension
# import math
# z_mean, z_std, z = model.model.encoder.predict(x_test, batch_size = batch_size)
# decoded = model.model.decoder(z)

# def convert_to_display(samples):
#     cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
#     samples = np.transpose(samples, axes=[1, 0, 2, 3])
#     samples = np.reshape(samples, [height, cnt, cnt, width])
#     samples = np.transpose(samples, axes=[1, 0, 2, 3])
#     samples = np.reshape(samples, [height*cnt, width*cnt])
#     return samples

# plt.imshow(convert_to_display(decoded[:100]), cmap = 'Greys_r')


    
#%% disentanglement metrics
# metrics = compute_metric(x_test.reshape(x_test.shape[0], x_test.shape[1]), z, True, 10).run()
# print(disentanglement_metric_mingetal_(z, x_test.reshape(x_test.shape[0], x_test.shape[1])))

#%% PCA disentanglement metric


# pc = PCA(k = 10).fit(x_train.reshape(x_train.shape[0], x_train.shape[1]))
# pcc = pc.fit_transform(x_test.reshape(x_test.shape[0], x_test.shape[1]))


#%% plot PCA subspace

# fig = plt.figure(figsize=(5, 4))
# plt.scatter(pcc[:, 0], pcc[:, 1], c = y_test, edgecolor="black", s = 10)
# plt.xlabel('pc[0]')
# plt.ylabel('pc[1]')
# plt.title(f'PCA subspace. {round(np.mean(np.array(metric)), 2)} +/- {round(np.std(np.array(metric)), 2)}')

# fig = plt.figure(figsize=(5, 4))
# plt.scatter(z[:, 0], z[:, 1], c = y_test, edgecolor="black", s = 10)
# plt.xlabel('z[0]')
# plt.ylabel('z[1]')
# plt.title(f'infoVAE: {round(np.mean(np.array(metricc)), 2)} +/- {round(np.std(np.array(metricc)), 2)}')

# fig = plt.figure(figsize=(5, 4))
# plt.scatter(z[:, 0], z[:, 1], c = y_test, edgecolor="black", s = 10)
# plt.xlabel('z[0]')
# plt.ylabel('z[1]')
# plt.title(f'GCVAE: {round(np.mean(np.array(metriccs)), 2)} +/- {round(np.std(np.array(metriccs)), 2)}')


#%% plot reconstructed latent space

# lr = 1e-3
# epochs = 2
# hidden_dim = 512
# latent_dims = 10
# loss_type = list(params.keys())[0] #elbo -> 0; controlvae -> 1; infovae -> 2; gcvae -> 3
# archi_type = 'v1'
# #params
# distrib_type = 'b'
# beta, gamma = params[f'{loss_type}']
# mmd_typ = 'mah_gcvae' #['mmd', 'mah', 'mah_rkhs', 'mah_gcvae']


# # results = np.load(os.path.join(path, f"{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"), allow_pickle=True).ravel()[0]
# model = load_model(os.path.join(path, f"{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/model.h5"))
# z_mean, z_std, z = model.encoder.predict(x_test, batch_size = batch_size)
# plot_latent_space(model, n= 10)


#%%%results only

# lr = 1e-3
# losstypes = 'g'
# epochs = 500
# hidden_dim = 512
# latent_dims = 10
# loss_type = 'controlvae'
# archi_type = 'v1'
# #params
# distrib_type = 'b'

# results = {}
# for ii in list(params.keys()):
#     results[f"{ii}"] = np.load(os.path.join(path, f"{ii}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"), allow_pickle=True).ravel()[0]
# results = np.load(os.path.join(path, f"{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"), allow_pickle=True).ravel()[0]

#%%

# lls = [x for x in list(params.keys()) if not 'elbo' in x]
# for ii in lls:
#     plt.plot(np.arange(epochs), results[ii]['kl_div'], label =f"{ii}")
    # ul = np.array(results[ii]['kl_div']) + np.random.randint(epochs) * 0.02
    # ll = np.array(results[ii]['kl_div']) - np.random.randint(epochs) * 0.015
    # plt.fill_between(results[ii]['kl_div'], ul, ll, alpha = .1)
    # plt.legend()



#%%%results only

# lr = 1e-3
# losstypes = 'g'
# epochs = 150
# hidden_dim = 512
# latent_dims = 10
# loss_type = 'elbo'
# archi_type = 'v1'
# #params
# distrib_type = 'b'


# results= np.load(os.path.join(path, f"model_2/{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"), allow_pickle=True).ravel()[0]


# plt.plot(np.arange(epochs), results['elbo'], label =f"{ii}")

    
#%%

# results = compute_metric(x_test.reshape(x_test.shape[0], x_test.shape[1]), z).run()
# results['z'] = z

#%%

# latent_dims = 2
# with open(os.path.join(path, f"{loss_type}/{datatype}/latent_{latent_dims}/model.pkl"), "rb") as pickl:
#     pcd = pickle.load(pickl)

# pcc = pcd.fit_transform(x_test.reshape(x_test.shape[0], x_test.shape[1]))
# filtered = tf.tensordot(pcc, pcd.components_, axes=1) + pcd.mean

# #%%plot latent space

# plt.scatter(z[:, 0], z[:, 1], c= y_test, s= 5, edgecolors='black')
# plt.xlabel('z[1]')
# plt.ylabel('z[2]')
# # plt.axis('off')

# #%%
    
# def plot_digits(data):
#     fig, axes = plt.subplots(10, 10, figsize=(10, 4),
#                               subplot_kw={'xticks':[], 'yticks':[]},
#                               gridspec_kw=dict(hspace=0.1, wspace=0.1))
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(data[i].reshape(28, 28),
#                   cmap='Greys_r', interpolation='nearest',
#                   clim=(0, 16))
        
        
# #%%

# plot_digits(filtered.numpy())   
#%% visualizing the result of 
# import scipy

# lr = 1e-3
# losstypes = 'g'
# epochs = 500
# hidden_dim = 512
# latent_dims = 2
# loss_type = 'gcvae'
# archi_type = 'v1'
# #params
# distrib_type = 'b'

# results = {}
# for ii in list(params.keys()):
#     if ii != 'gcvae':
#         results[f"{ii}"] = np.load(os.path.join(path, f"{ii}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"), allow_pickle=True).ravel()[0]
#     else:
#         for ij in ['mah_gcvae']:
#             results[f"{ij}"] = np.load(os.path.join(path, f"{ii}/{datatype}/latent_{latent_dims}/{epochs}/{ij}/loggers.npy"), allow_pickle=True).ravel()[0]

# lls = list(params.keys()) + ['mah_gcvae']
# lls.pop(4)
# plot_of = 'reconstruction'
# for ii in lls:
#     plt.plot(np.arange(epochs), results[ii][f'{plot_of}'], '-', label =f"{ii}", lw = 1.)
#     #sns.lineplot(x =np.arange(epochs), y=results[ii][f'{plot_of}'])
#     plt.title(f'{plot_of.upper()}')
#     # ul = np.array(results[ii]['kl_div']) + np.random.randint(epochs) * 0.02
#     # ll = np.array(results[ii]['kl_div']) - np.random.randint(epochs) * 0.015
#     # plt.fill_between(results[ii]['kl_div'], ul, ll, alpha = .1)
#     plt.legend()

#%%
# loss_type = 'gcvae'
# model = load_model(os.path.join(path, f"{loss_type}/{datatype}/latent_{latent_dims}/{epochs}/{mmd_typ}/model.h5"))
# z_mean, z_std, z = model.encoder.predict(x_test, batch_size = batch_size)
# plot_latent_space(model, n= 10)
# dt = {}
# for ii in lls:
#     dt[f'{ii}'] = pd.DataFrame({'epoch': np.arange(epochs),
#                                 'res': results[ii][f'{plot_of}']})
    
# dt = pd.DataFrame({'epoch': np.arange(epochs),
#                     'elbo': results['elbo'][f'{plot_of}'],
#                     'controlvae': results['controlvae'][f'{plot_of}'],
#                     'infovae': results['infovae'][f'{plot_of}'],
#                     'mah': results['mah'][f'{plot_of}'],
#                     'mah_gcvae': results['mah_gcvae'][f'{plot_of}'],
#                     'mmd': results['mmd'][f'{plot_of}']})


  #%%

# sns.lineplot( data=dt, x="epoch", y="elbo")
# sns.lineplot( data=dt, x="epoch", y="controlvae")
# sns.lineplot( data=dt, x="epoch", y="infovae")
# sns.lineplot( data=dt, x="epoch", y="mah")
# sns.lineplot( data=dt, x="epoch", y="mah_gcvae")
# sns.lineplot( data=dt, x="epoch", y="mmd")
   
#%%
# sns.lineplot( data=dtt, x="epoch", y="vals", hue="cols")
# plt.show()
    
   
    
#%%
# fmri = sns.load_dataset("fmri")
# fmri.head()
# sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")


#%% visualizing the result of model_100
# import scipy

# lr = 1e-3
# losstypes = 'g'
# epochs = 100
# hidden_dim = 512
# latent_dims = 2
# loss_type = 'elbo'
# archi_type = 'v1'
# #params
# distrib_type = 'b'

# results = {}
# for ii in list(params.keys()):
#     results[f"{ii}"] = np.load(os.path.join(path, f"model_100/{ii}/{datatype}/latent_{latent_dims}/{epochs}/loggers.npy"), allow_pickle=True).ravel()[0]


# lls = list(params.keys())

# plot_of = 'reconstruction'
# for ii in lls:
#     plt.plot(np.arange(epochs), results[ii][f'{plot_of}'], label =f"{ii}", lw = 1.)
#     plt.legend()
   
   

    
   
    
   
    
   
    
   
    
    
