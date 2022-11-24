## GCVAE: Generalized-Controllable Variational Autoencoder

### ABSTRACT: 
```
Variational autoencoders (VAEs) have recently been used for unsupervised disentanglement learning
of complex densitydistributions. Numerous variants exist to encourage disen tanglement in latent
space while improving reconstruction. However, none have simultaneously managed the trade-off 
between attaining extremely low reconstruction error and a high disentanglement score. We present
a generalized framework to handle this challenge under constrained optimization and demonstrate
that it outperforms state-of-the-art existing models as regards disentanglement while balancing
reconstruction. We introduce three controllable Lagrangian hyperparameters to control 
reconstruction loss, KL divergence loss and correlation measure. We prove that maximizing 
information in the reconstruction network is equivalent to information maximization during
amortized inference under reasonable assumptions and constraint relaxation.
```
-----------------------------------

### How to use

[1] Download all the files to your local machine using
```python
git clone https://github.com/kennedyCzar/GCVAE
```

[2] Depending on the experiment of interest, load the appropriate script in a python IDE of choice. For MNITS load,
```python
load main_mnist_2d.py
```
[3] Set parameters ad run (The current parameters are optimal for experiment)
```python
python main_mnist_2d.py
```
example on MNIST dataset
```python
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


lr = 1e-3
epochs = 250000
hidden_dim = 200
latent_dims = 2
loss_type = list(params.keys())[loss_index] #elbo -> 0; beta-vae --> 1; controlvae -> 2; infovae -> 3; gcvae -> 4
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
```
-----------------------------------
### Cite

```
@misc{https://doi.org/10.48550/arxiv.2206.04225,
  doi = {10.48550/ARXIV.2206.04225},
  url = {https://arxiv.org/abs/2206.04225},
  author = {Ezukwoke, Kenneth and Hoayek, Anis and Batton-Hubert, Mireille and Boucher, Xavier},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences, 62F15, 62F30},
  title = {GCVAE: Generalized-Controllable Variational AutoEncoder},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
                        
```

### License
MIT
