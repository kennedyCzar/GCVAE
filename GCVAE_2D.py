from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from utils import (plot_latent,
                   plot_losses,
                   plot_losses_with_latent,
                   best_gmm_model,
                   Mergefeatures,
                   SimilarityMat,
                   keywordReturn,
                   PIDControl_v1,
                   PIDControl_v2,
                   )
from os.path import join
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.patheffects as pe
tf.config.run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.model_selection import train_test_split
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


#---set environment variables...

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = f"{tf.config.experimental.list_physical_devices('CPU')}"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
class gcvae_v1(keras.Model):
    def __init__(self, input_shape:tuple, 
                 feature_size:int, 
                 batch_size:int, 
                 hidden_dim:int, 
                 latent_dim:int = 10, 
                 dist:str = None,
                 **kwargs):
        '''Convolutional Variational Autoencoder
        

        Parameters
        ----------
        input_shape : tuple
            input shape of training data. (DX1) for 1D CNN
        feature_size : int
            Equivalent to D, the size of the features in the data.
        batch_size : int
            batch size.
        hidden_dim : int
            hidden dimension.
        latent_dim : int, optional
            latent dimension. The default is 10.
        dist : str, optional
            distribution type. The default is None.
        **kwargs : dict
            None.

        Returns
        -------
        None.

        '''
        super(gcvae_v1, self).__init__(**kwargs)
        if dist == None:
            dist = 'g'
            self.dist = dist
        else:
            self.dist = dist
        #encoder -- section
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        #-----encoder section begins here--------------------------------------------------------------------------
        #---------------------------------Concolutional part-------------------------------------------------------
        encoder_inputs = tf.keras.layers.Input(shape = input_shape, name='input') #input layer
        x = tf.keras.layers.Conv2D(64, 3, activation = tf.nn.relu, padding = 'same')(encoder_inputs) #Conv 1
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation = tf.nn.relu, padding = 'valid')(x) #Conv 2
        x = tf.keras.layers.MaxPooling2D(2)(x) 
        x = tf.keras.layers.BatchNormalization()(x)
        max_pool_sz = x.shape[1:] #extract shape of final convolution
        x = layers.Flatten()(x)
        flatt = x.shape[1:][0] #extract for decoder reshape
        #--------------------------------Feed forward part (FC 2)--------------------------------------------------------
        x = tf.keras.layers.Dense(self.hidden_dim, activation = tf.nn.relu)(x) #hidden layer of encoder
        x = tf.keras.layers.Dense(self.hidden_dim//8, activation = tf.nn.relu)(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x) #mean
        z_log_cov = tf.keras.layers.Dense(self.latent_dim, name = "z_log_cov")(x) #log variance/ variance
        z = tf.keras.layers.Lambda(gcvae_v1.gauss_reparam, name='sampling')([z_mean, z_log_cov]) #reparameterization
        #---econder object --------------------------------------------------------------------------------------
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_cov, z], name="encoder") #encoder model
        self.encoder = encoder
        #---------------------------------------------------------------------------------------------------------
        #--------------decoder -- section-------------------------------------------------------------------------
        latent_inputs = keras.layers.Input(shape = (self.latent_dim,), name = 'latent_dim') #layer after latent dimension
        x = tf.keras.layers.Dense(self.hidden_dim//8, activation = tf.nn.relu)(latent_inputs)
        x = tf.keras.layers.Dense(self.hidden_dim, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Dense(flatt, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Reshape(max_pool_sz)(x) #reshape to the size of the final convolutional layer
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation = tf.nn.relu, padding="valid")(x) #Deconv layer 1
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation = tf.nn.relu, padding="same")(x) #Deconv layer 2
        if self.dist == 'b':
            decoder_outputs = layers.Conv2DTranspose(1,1, activation = tf.nn.sigmoid, padding="same")(x) #output layer
        else:
            decoder_outputs = layers.Conv2DTranspose(1,1, activation = tf.nn.relu, padding="same")(x) #output layer
        #---deconder object --------------------------------------------------------------------------------------
        decoder = keras.Model(latent_inputs, decoder_outputs, name = "decoder")
        self.decoder = decoder
        #--- loss function tracker
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta_tracker = keras.metrics.Mean(name="betas")
        self.alpha_tracker = keras.metrics.Mean(name="alphas")
        self.gamma_tracker = keras.metrics.Mean(name="gammas")
        self.mmd_tracker = keras.metrics.Mean(name="mmd") #mmd here could be mmd or mahalanobis distance...

    @property
    def metrics(self):
        #---Note that the metric is the mean...
        return [
            self.total_loss_tracker, #track the ELBO or variational mutual information loss
            self.reconstruction_loss_tracker, #track the reconstruction loss
            self.kl_loss_tracker, #track the KL divergence
            self.beta_tracker, #track the beta weights on KL divergence
            self.alpha_tracker, #track alpha weight on lg-likelihood
            self.gamma_tracker, #track the MMD weight
            self.mmd_tracker
        ]

    @staticmethod
    def gauss_reparam(args):
        """Reparameterization trick. Instead of sampling from Q(z|X), 
        sample eps = N(0,I) z = z_mean + sqrt(var)*eps.

        Parameters:
        -----------
        args: list of Tensors
            Mean and log of stddev of Q(z|X)

        Returns
        -------
        z: Tensor
            Sampled latent vector
        """

        z_mean, z_log_cov = args
        eps = tf.keras.backend.random_normal(tf.shape(z_log_cov), dtype = tf.float32, mean = 0., stddev = 1.0)
        z = z_mean + tf.exp(z_log_cov / 2) * eps
        return z
    
    @staticmethod
    def vae_univ_gauss(mu, lg_sigma, r_loss, beta = None, vloss = 'elbo'):
        '''Computing the KL loss and VAE loss using 
            parameters of alpha and beta

        Parameters
        ----------
        mu : np.array
            vector mean.
        lg_sigma : np.array
            log of variance.
        r_loss : np.float32
            reconstruction loss.
        beta : np.int
            Beta constraint. source: https://openreview.net/pdf?id=Sy2fzU9gl, default is None.
        vloss: str, optional
                choice of parameter to optimize based on loss function

        Returns
        ---------
        dict:
            vae loss
            kl loss
            alpha
            beta
        '''
        kl_loss = 1 + lg_sigma - tf.square(mu) - tf.exp(lg_sigma)
        kl_loss = 0.5 * tf.reduce_sum(kl_loss, axis = -1)
        kl_loss = -tf.reduce_mean(kl_loss)
        #select parameters...
        if vloss == 'elbo':
            alpha, beta = -1, beta
        elif vloss == 'betavae':
            alpha, beta = -1, beta
        elif vloss == 'controlvae':
            alpha = 0 
            beta = PIDControl_v2().pid(30, kl_loss)
        elif vloss == 'infovae':
            alpha, beta = 0, 0
        elif vloss == 'factorvae':
            alpha, beta = -1, 1
        elif vloss == 'gcvae':
            alpha = PIDControl_v2().pid(10, r_loss) #reconstruction weight
            beta = PIDControl_v2().pid(30, kl_loss) #weight on KL-divergence
        else:
            return ValueError(f'Unknown loss type: {vloss}')
        if not vloss == 'betavae':
            vae_loss = (1-alpha-beta)*r_loss + beta*kl_loss
            return {'vae_loss': vae_loss,
                'kl_loss': kl_loss,
                'alpha': alpha,
                'beta': beta
                }
        else:
            assert len(beta) == 2, 'length of beta cannot be less than 2'
            vae_loss = (1-alpha-beta[0])*r_loss + beta[1]*kl_loss
            return {'vae_loss': vae_loss,
                    'kl_loss': kl_loss,
                    'alpha': alpha,
                    'beta': beta[1]
                    }
            
    @staticmethod
    def reconstruction(difference):
        return tf.reduce_mean(difference)
    
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=-1) / tf.cast(dim, tf.float32))
    
    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    
    def z_mahalanobis_fn(self, z, diag:bool = False, psd = False)->float:
        '''
        Parameters
        ----------
        z : numpy array
            latent array/code.
        diag : bool, optional
            Diagonal of the covariance matrix. The default is False.
    
        Returns
        -------
        float
            mahalanobis mean of the latent vector.
    
        '''
        z = z.numpy()
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        
        cov = 1/(len(z)-1)*z_m.T.dot(z_m)
        diag_cov = np.diag(np.diag(cov))
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            diag_cov = np.diag(np.diag(cov))
        else:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            cov = np.where(cov < 0, 0, cov)
            diag_cov = np.diag(np.diag(cov))
            diag_cov = np.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        trans_x = z_m.dot(inv_cov).dot(z_m.T)
        mah_mat_mean = np.mean(trans_x.diagonal())
        return tf.Variable(mah_mat_mean, dtype=tf.float32)
    
    def z_mahalanobis_rkhs_fn(self, z, diag:bool = False, psd = False)->float:
        '''Reproducing Kernel Hilbert Space (RKHS)
           Mahalanobis distance
        
    
        Parameters
        ----------
        z : numpy array
            latent array/code.
        diag : bool, optional
            Diagonal of the covariance matrix. The default is False.
        
        psd: bool, optional
            is matrix is not positive semi definite
            
        Returns
        -------
        float
            mahalanobis mean of the latent vector.
    
        '''
        #z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
        z = self.compute_kernel(z, z)
        z = z.numpy()
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            diag_cov = np.diag(np.diag(cov))
        else:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            cov = np.where(cov < 0, 0, cov)
            diag_cov = np.diag(np.diag(cov))
            diag_cov = np.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        trans_x = z_m.dot(inv_cov).dot(z_m.T)
        mah_mat_mean = np.mean(trans_x.diagonal())
        return tf.Variable(mah_mat_mean, dtype=tf.float32)


    def z_mahalanobis_gcvae(self, z, diag:bool = False, psd = False)->float:
        '''Reproducing Kernel Hilbert Space (RKHS)
           Mahalanobis distance
        
    
        Parameters
        ----------
        z : numpy array
            latent array/code.
        diag : bool, optional
            Diagonal of the covariance matrix. The default is False.
        
        psd: bool, optional
            is matrix is not positive semi definite
            
        Returns
        -------
        float
            mahalanobis mean of the latent vector.
    
        '''
        z = z.numpy()
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            diag_cov = np.diag(np.diag(cov))
        else:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            cov = np.where(cov < 0, 0, cov)
            diag_cov = np.diag(np.diag(cov))
            diag_cov = np.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
        mah_gcvae = inv_cov.dot(self.compute_mmd(z_sample, z))
        mah_gcvae_mean = np.mean(mah_gcvae.diagonal())
        return tf.Variable(mah_gcvae_mean, dtype=tf.float32)
    
    def mmd(self, z):
        z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
        return self.compute_mmd(z_sample, z)
    
    def z_mahalanobis(self, z):
        return self.z_mahalanobis_fn(z)
    
    def z_mahalanobis_rkhs_mmd(self, z):
        return self.z_mahalanobis_rkhs_fn(z)
    
    def z_mah_gcvae(self, z):
        return self.z_mahalanobis_gcvae(z)
    
   
    
     
#%%

class gcvae_v2(keras.Model):
    def __init__(self, input_shape:tuple, 
                 feature_size:int, 
                 batch_size:int, 
                 hidden_dim:int, 
                 latent_dim:int = 10, 
                 dist:str = None,
                 **kwargs):
        '''Convolutional Variational Autoencoder
        

        Parameters
        ----------
        input_shape : tuple
            input shape of training data. (DX1) for 1D CNN
        feature_size : int
            Equivalent to D, the size of the features in the data.
        batch_size : int
            batch size.
        hidden_dim : int
            hidden dimension.
        latent_dim : int, optional
            latent dimension. The default is 10.
        dist : str, optional
            distribution type. The default is None.
        **kwargs : dict
            None.

        Returns
        -------
        None.

        '''
        super(gcvae_v2, self).__init__(**kwargs)
        if dist == None:
            dist = 'g'
            self.dist = dist
        else:
            self.dist = dist
        #encoder -- section
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        #-----encoder section begins here--------------------------------------------------------------------------
        #---------------------------------Convolutional part-------------------------------------------------------
        encoder_inputs = tf.keras.layers.Input(shape = input_shape, name='input') #input layer
        x = tf.keras.layers.Conv2D(64, 2, activation = tf.nn.relu, padding = 'same')(encoder_inputs) #Conv 1
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 2, activation = tf.nn.relu, padding = 'same')(x) #Conv 2
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 2, activation = tf.nn.relu, padding = 'same')(x) #Conv 3
        x = tf.keras.layers.MaxPooling2D(3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 2, activation = tf.nn.relu, padding = 'same')(x) #Conv 4
        x = tf.keras.layers.MaxPooling2D(2)(x) 
        x = tf.keras.layers.BatchNormalization()(x)
        max_pool_sz = x.shape[1:] #extract shape of final convolution
        x = layers.Flatten()(x)
        flatt = x.shape[1:][0] #extract for decoder reshape
        #--------------------------------Feed forward part (FC 2)--------------------------------------------------------
        x = tf.keras.layers.Dense(self.hidden_dim, activation = tf.nn.relu)(x) #hidden layer of encoder
        x = tf.keras.layers.Dense(self.hidden_dim//8, activation = tf.nn.relu)(x)
        z_mean = tf.keras.layers.Dense(self.latent_dim, name="z_mean")(x) #mean
        z_log_cov = tf.keras.layers.Dense(self.latent_dim, name = "z_log_cov")(x) #log variance/ variance
        z = tf.keras.layers.Lambda(gcvae_v2.gauss_reparam, name='sampling')([z_mean, z_log_cov]) #reparameterization
        #---econder object --------------------------------------------------------------------------------------
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_cov, z], name="encoder") #encoder model
        self.encoder = encoder
        #---------------------------------------------------------------------------------------------------------
        #--------------decoder -- section-------------------------------------------------------------------------
        latent_inputs = keras.layers.Input(shape = (self.latent_dim,), name = 'latent_dim') #layer after latent dimension
        x = tf.keras.layers.Dense(self.hidden_dim//8, activation = tf.nn.relu)(latent_inputs)
        x = tf.keras.layers.Dense(self.hidden_dim, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Dense(flatt, activation = tf.nn.relu)(x)
        x = tf.keras.layers.Reshape(max_pool_sz)(x) #reshape to the size of the final convolutional layer
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2DTranspose(32, 2, activation = tf.nn.relu, padding="valid")(x) #Deconv layer 1
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2DTranspose(32, 2, activation = tf.nn.relu, padding="valid")(x) #Deconv layer 2
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2DTranspose(64, 2, activation = tf.nn.relu, padding="same")(x) #Deconv layer 3
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2DTranspose(64, 2, activation = tf.nn.relu, padding="same")(x) #Deconv layer 4
        if self.dist == 'b':
            decoder_outputs = layers.Conv2DTranspose(1, 1, activation = tf.nn.sigmoid, padding="same")(x) #output layer
        else:
            decoder_outputs = layers.Conv2DTranspose(1, 1, activation = tf.nn.relu, padding="same")(x) #output layer
        #---deconder object --------------------------------------------------------------------------------------
        decoder = keras.Model(latent_inputs, decoder_outputs, name = "decoder")
        self.decoder = decoder
        #--- loss function tracker
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.beta_tracker = keras.metrics.Mean(name="betas")
        self.alpha_tracker = keras.metrics.Mean(name="alphas")
        self.gamma_tracker = keras.metrics.Mean(name="gammas")
        self.mmd_tracker = keras.metrics.Mean(name="mmd") #mmd here could be mmd or mahalanobis distance...

    @property
    def metrics(self):
        #---Note that the metric is the mean...
        return [
            self.total_loss_tracker, #track the ELBO or variational mutual information loss
            self.reconstruction_loss_tracker, #track the reconstruction loss
            self.kl_loss_tracker, #track the KL divergence
            self.beta_tracker, #track the beta weights on KL divergence
            self.alpha_tracker, #track alpha weight on lg-likelihood
            self.gamma_tracker, #track the MMD weight
            self.mmd_tracker
        ]

    @staticmethod
    def gauss_reparam(args):
        """Reparameterization trick. Instead of sampling from Q(z|X), 
        sample eps = N(0,I) z = z_mean + sqrt(var)*eps.

        Parameters:
        -----------
        args: list of Tensors
            Mean and log of stddev of Q(z|X)

        Returns
        -------
        z: Tensor
            Sampled latent vector
        """

        z_mean, z_log_cov = args
        eps = tf.keras.backend.random_normal(tf.shape(z_log_cov), dtype = tf.float32, mean = 0., stddev = 1.0)
        z = z_mean + tf.exp(z_log_cov / 2) * eps
        return z
    
    @staticmethod
    def vae_univ_gauss(mu, lg_sigma, r_loss, beta = None, vloss = 'elbo'):
        '''Computing the KL loss and VAE loss using 
            parameters of alpha and beta

        Parameters
        ----------
        mu : np.array
            vector mean.
        lg_sigma : np.array
            log of variance.
        r_loss : np.float32
            reconstruction loss.
        beta : np.int
            Beta constraint. source: https://openreview.net/pdf?id=Sy2fzU9gl, default is None.
        vloss: str, optional
                choice of parameter to optimize based on loss function

        Returns
        ---------
        dict:
            vae loss
            kl loss
            alpha
            beta
        '''
        kl_loss = 1 + lg_sigma - tf.square(mu) - tf.exp(lg_sigma)
        kl_loss = 0.5 * tf.reduce_sum(kl_loss, axis = -1)
        kl_loss = -tf.reduce_mean(kl_loss)
        #select parameters...
        if vloss == 'elbo':
            alpha, beta = -1, beta
        elif vloss == 'betavae':
            alpha, beta = -1, beta
        elif vloss == 'controlvae':
            alpha = 0 
            beta = PIDControl_v2().pid(30, kl_loss)
        elif vloss == 'infovae':
            alpha, beta = 0, 0
        elif vloss == 'factorvae':
            alpha, beta = -1, 1
        elif vloss == 'gcvae':
            alpha = PIDControl_v2().pid(10, r_loss) #reconstruction weight
            beta = PIDControl_v2().pid(30, kl_loss) #weight on KL-divergence
        else:
            return ValueError(f'Unknown loss type: {vloss}')
        if not vloss == 'betavae':
            vae_loss = (1-alpha-beta)*r_loss + beta*kl_loss
            return {'vae_loss': vae_loss,
                'kl_loss': kl_loss,
                'alpha': alpha,
                'beta': beta
                }
        else:
            #used for beta-VAE
            assert len(beta) == 2, 'length of beta cannot be less than 2'
            vae_loss = (1-alpha-beta[0])*r_loss + beta[1]*kl_loss
            return {'vae_loss': vae_loss,
                    'kl_loss': kl_loss,
                    'alpha': alpha,
                    'beta': beta[1]
                    }
            
    @staticmethod
    def reconstruction(difference):
        return tf.reduce_mean(difference)
    
    def compute_kernel(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=-1) / tf.cast(dim, tf.float32))
    
    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    
    def z_mahalanobis_fn(self, z, diag:bool = False, psd = False)->float:
        '''
        Parameters
        ----------
        z : numpy array
            latent array/code.
        diag : bool, optional
            Diagonal of the covariance matrix. The default is False.
    
        Returns
        -------
        float
            mahalanobis mean of the latent vector.
    
        '''
        z = z.numpy()
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        
        cov = 1/(len(z)-1)*z_m.T.dot(z_m)
        diag_cov = np.diag(np.diag(cov))
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            diag_cov = np.diag(np.diag(cov))
        else:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            cov = np.where(cov < 0, 0, cov)
            diag_cov = np.diag(np.diag(cov))
            diag_cov = np.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        trans_x = z_m.dot(inv_cov).dot(z_m.T)
        mah_mat_mean = np.mean(trans_x.diagonal())
        return tf.Variable(mah_mat_mean, dtype=tf.float32)
    
    def z_mahalanobis_rkhs_fn(self, z, diag:bool = False, psd = False)->float:
        '''Reproducing Kernel Hilbert Space (RKHS)
           Mahalanobis distance
        
    
        Parameters
        ----------
        z : numpy array
            latent array/code.
        diag : bool, optional
            Diagonal of the covariance matrix. The default is False.
        
        psd: bool, optional
            is matrix is not positive semi definite
            
        Returns
        -------
        float
            mahalanobis mean of the latent vector.
    
        '''
        #z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
        z = self.compute_kernel(z, z)
        z = z.numpy()
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            diag_cov = np.diag(np.diag(cov))
        else:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            cov = np.where(cov < 0, 0, cov)
            diag_cov = np.diag(np.diag(cov))
            diag_cov = np.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        trans_x = z_m.dot(inv_cov).dot(z_m.T)
        mah_mat_mean = np.mean(trans_x.diagonal())
        return tf.Variable(mah_mat_mean, dtype=tf.float32)


    def z_mahalanobis_gcvae(self, z, diag:bool = False, psd = False)->float:
        '''Reproducing Kernel Hilbert Space (RKHS)
           Mahalanobis distance
        
    
        Parameters
        ----------
        z : numpy array
            latent array/code.
        diag : bool, optional
            Diagonal of the covariance matrix. The default is False.
        
        psd: bool, optional
            is matrix is not positive semi definite
            
        Returns
        -------
        float
            mahalanobis mean of the latent vector.
    
        '''
        z = z.numpy()
        m = lambda z: z - z.mean(axis = 0) #mean of vectors
        z_m = m(z) #mean centered data
        #check if matrix entries are 
        if not psd:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            diag_cov = np.diag(np.diag(cov))
        else:
            cov = 1/(len(z)-1)*z_m.T.dot(z_m)
            cov = np.where(cov < 0, 0, cov)
            diag_cov = np.diag(np.diag(cov))
            diag_cov = np.where(diag_cov < 0, 0, diag_cov)
        if not diag:
            inv_cov = np.linalg.inv(cov) #inverse of a full covariance matrix
        else:
            inv_cov = np.linalg.inv(diag_cov) #inverse of diagonal covariance matrix
        z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
        mah_gcvae = inv_cov.dot(self.compute_mmd(z_sample, z))
        mah_gcvae_mean = np.mean(mah_gcvae.diagonal())
        return tf.Variable(mah_gcvae_mean, dtype=tf.float32)
    
    def mmd(self, z):
        z_sample = tf.keras.backend.random_normal(tf.shape(z), dtype = tf.float32, mean = 0., stddev = 1.0)
        return self.compute_mmd(z_sample, z)
    
    def z_mahalanobis(self, z):
        return self.z_mahalanobis_fn(z)
    
    def z_mahalanobis_rkhs_mmd(self, z):
        return self.z_mahalanobis_rkhs_fn(z)
    
    def z_mah_gcvae(self, z):
        return self.z_mahalanobis_gcvae(z)
    
   
    
    
    
    
