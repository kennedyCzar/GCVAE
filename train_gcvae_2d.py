from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from GCVAE_2D import gcvae_v1, gcvae_v2
from utils import (z_mahalanobis, z_mahalanobis_rkhs,
                   z_mahalanobis_v2,
                   PIDControl_v2, PIDControl_v1,
                   Metric, model_saver, model_saver_2d
                   )


class train_gcvae(object):
    def __init__(self, 
                 inp_shape:tuple, 
                 num_features:int, 
                 hidden_dim:int = 50, 
                 latent_dim:int = 10, 
                 batch_size:int = 300, 
                 beta:float = 1.,
                 gamma:float = 1.,
                 dist:str = 'b',
                 vloss:str = 'elbo',
                 lr:float = 1e-3, 
                 epochs:int = 3,
                 architecture = 'v1',
                 mmd_type = 'default',
                 save_latent = False, #save latent model at every epoch,
                 **kwargs):
        '''
        

        Parameters
        ----------
        inp_shape : tuple
            input shape. Usually a tuple of (Dx1) dimension.
        num_features : int
            Number of features in the data. This is equivalent to D.
        hidden_dim : int, optional
            number of units in the hidden layer. The default is 50.
        latent_dim : int, optional
            latent dimension of interest. The default is 2.
        batch_size : int, optional
            batch size used for training. The default is 128.
        beta : float, optional
            beta value. The default is 1. beta >1 is equivalent to beta VAE.
        beta : float, optional
            info value. The default is 1. beta >1 is equivalent to beta VAE.
        dist : str, optional
            distribution type. can either be Guassian or Bernoulli. The default is 'b'.
        vloss : str, optional
            loss type e.g albo, controlvae, infovae, factorvae. The default is elbo.
        lr : float, optional
            learning rate. The default is 1e-3.
        epochs : int, optional
            numbers of epochs to train model. The default is 3.
        architecture: str, optional
            type of neural architecture to use.
        **kwargs : dict
            None.

        Returns
        -------
        None.

        '''
        super(train_gcvae, self).__init__()
        self.inp_shape = inp_shape
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.gamma = gamma
        self.dist = dist
        self.vloss = vloss
        self.lr = lr
        self.epochs = epochs
        self.optimizers = keras.optimizers.Adam(learning_rate = self.lr)
        self.architecture = architecture
        self.save_latent = save_latent
        if self.architecture == 'v1':
            self.model = gcvae_v1(self.inp_shape, self.num_features, self.batch_size,
                              self.hidden_dim, self.latent_dim, self.dist)
        elif self.architecture == 'v2':
            self.model = gcvae_v2(self.inp_shape, self.num_features, self.batch_size, 
                              self.hidden_dim, self.latent_dim, self.dist)
        else:
            raise ValueError(f'Unknown architecture type: {self.architecture}. Only "v1" or "v2" is allowed')
        self.mmd_type = mmd_type
        if self.mmd_type == 'mmd':
            self.mmd_fn = self.model.mmd
        elif self.mmd_type == 'mah':
            self.mmd_fn = self.model.z_mahalanobis
        elif self.mmd_type == 'mah_rkhs':
            self.mmd_fn = self.model.z_mahalanobis_rkhs_mmd
        elif self.mmd_type == 'mah_gcvae':
            self.mmd_fn = self.model.z_mah_gcvae
        else:
            raise ValueError(f"Unexpected mmd type: {self.mmd_type}. Only types 'mmd', 'mah', 'mah_rkhs' are allowed")
            
            
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_cov, z = self.model.encoder(data)
            reconstruction = self.model.decoder(z)
            #reconstruction loss
            if len(self.inp_shape) <=2:
                if self.model.dist == 'Gauss' or self.model.dist == 'Gaussian' or self.model.dist == 'G' or self.model.dist == 'g':
                    marginal_likelihood = tf.reduce_sum(keras.losses.MSE(data, reconstruction), axis = -1)
                elif self.model.dist == 'Bern' or self.model.dist == 'Bernoulli' or self.model.dist == 'b' or self.model.dist == 'B': 
                    marginal_likelihood = tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis = -1)
                else:
                    raise ValueError(f'{self.model.dist} specified is unknown\nPlease use a known distribution type')
            else:
                if self.model.dist == 'Gauss' or self.model.dist == 'Gaussian' or self.model.dist == 'G' or self.model.dist == 'g':
                    marginal_likelihood = tf.reduce_sum(keras.losses.MSE(data, reconstruction), axis = (1,2))
                elif self.model.dist == 'Bern' or self.model.dist == 'Bernoulli' or self.model.dist == 'b' or self.model.dist == 'B': 
                    marginal_likelihood = tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis = -1)
                else:
                    raise ValueError(f'{self.model.dist} specified is unknown\nPlease use a known distribution type')
            reconstruction_loss = self.model.reconstruction(marginal_likelihood)
            vae_loss_params = self.model.vae_univ_gauss(z_mean, z_log_cov, reconstruction_loss, self.beta, self.vloss)
            if self.vloss == 'elbo':
                gamma = 0
                self._mmd = 0
            elif self.vloss == 'betavae':
                gamma = 0
                self._mmd = 0
            elif self.vloss == 'controlvae':
                gamma = 0
                self._mmd = 0
            elif self.vloss == 'infovae':
                gamma = self.gamma
                self._mmd = self.mmd_fn(z)
            elif self.vloss == 'factorvae':
                gamma = -1
                self._mmd = 0
            elif self.vloss == 'gcvae':
                self._mmd = self.mmd_fn(z) #make sure to change this loss when dealing with GCVAE main
                gamma = PIDControl_v1().pid(0.1, self._mmd) #adaptive gamma
            else:
                raise ValueError(f'Unknown vloss paramater {self.vloss} set')
            vae_loss_params['vae_loss'] += gamma*self._mmd
            grads = tape.gradient(vae_loss_params['vae_loss'], self.model.trainable_weights)
            self.optimizers.apply_gradients(zip(grads, self.model.trainable_weights))
            self.model.total_loss_tracker.update_state(vae_loss_params['vae_loss'])
            self.model.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.model.kl_loss_tracker.update_state(vae_loss_params['kl_loss'])
            self.model.alpha_tracker.update_state(vae_loss_params['alpha'])
            self.model.beta_tracker.update_state(vae_loss_params['beta'])
            self.model.gamma_tracker.update_state(gamma)
            self.model.mmd_tracker.update_state(self._mmd)
            return {
                        "vae_loss": self.model.total_loss_tracker.result(),
                        "reconstruction_loss": self.model.reconstruction_loss_tracker.result(),
                        "kl_loss": self.model.kl_loss_tracker.result(),
                        "alphas": self.model.alpha_tracker.result(),
                        "betas": self.model.beta_tracker.result(),
                        "gammas": self.model.gamma_tracker.result(),
                        "mmd": self.model.mmd_tracker.result(),
                    }

    def fit(self, data, test, datatype:str,
            intermediate:bool = False, 
            stopping:bool = False,
            save_model:bool = True,
            save_model_iter = 1000,
            pid_a:bool = False,
            pid_b:bool = False,
            epsilon_a:float = 1e-6,
            epsilon_b:float = 1e-5):
        '''
        Parameters
        ---------------
        data : np.array
            (NxDx1) input/training data.
        test : np.array
            (NxDx1) test data.
        datatype : str
            data used for training model.
        intermediate : bool, optional
            Returns latent space before and after entering the negative zone. 
            The default is False.
        stopping : bool, optional
            Initiates the stopping criterion when difference between abs($\beta$[t] - $\beta$[t-1]) > epsilon
            The default is False.
        save_model : bool, optional
            Save model after certain number of iterations. For instance, model is asked to save after every 1000
            iterations.
            The default is True.
        save_model_iter : int, optional
            This works with save_model argument. The number of iterations to reach before saving a model.
            The default is 1000.
        pid_a : bool, optional
            Initiate stopping criterion for $\alpha$. The default is 1e-5
        pid_b : bool, optional
            Initiate stopping criterion for $\alpha$. The default is 1e-4
        epsilon_a : float, optional
            Threshold to stopping learning reconstruction loss. The default is 1e-3
        epsilon_b : float, optional
            Threshold to stopping learning KL divergence. The default is 1e-3

        Returns
        -------
        tf.keras.Model
            trained Model class.

        '''
        self.data = data
        self.test = test
        self.datatype = datatype
        self.intermediate = intermediate
        self.stopping = stopping
        self.save_model = save_model
        self.save_model_iter = save_model_iter
        self.pid_a = pid_a
        self.pid_b = pid_b
        self.epsilon_a = epsilon_a
        self.epsilon_b = epsilon_b
        self.z_latent_pos = 0   #positive latent space
        self.z_latent_neg = 0   #negative latent space
        self.ELBO, self.RECON_LOSS, self.KL_DIV, self.BETA, self.ALPHA, self.GAMMA, self.MMD = [], [], [], [], [], [], []
        self.z_t = 0
        self.int_z = {} #save intermediate latent after every epoch when necessary..;only for evaluation
        for self.epoch in range(1, self.epochs + 1):
            B_ELBO, B_RECON_LOSS, B_KL_DIV, B_ALPHA, B_BETA, B_GAMMA, B_MMD = [], [], [], [], [], [], []
            for ij in tqdm(self.data):
                loss_params = self.train_step(ij)
                # self.z_t = z
                B_ELBO.append(loss_params['vae_loss'])
                B_RECON_LOSS.append(loss_params['reconstruction_loss'])
                B_KL_DIV.append(loss_params['kl_loss'])
                B_ALPHA.append(loss_params['alphas'])
                B_BETA.append(loss_params['betas'])
                B_GAMMA.append(loss_params['gammas'])
                B_MMD.append(loss_params['mmd'])
            #----keep mean of batch losses...
            m_elbo = np.mean(B_ELBO)
            m_recon_loss = np.mean(B_RECON_LOSS)
            m_kl = np.mean(B_KL_DIV)
            m_alpha = np.mean(B_ALPHA)
            m_beta = np.mean(B_BETA)
            m_gamma = np.mean(B_GAMMA)
            m_mmd = np.mean(B_MMD)
            self.ELBO.append(m_elbo)
            self.RECON_LOSS.append(m_recon_loss)
            self.KL_DIV.append(m_kl)
            self.ALPHA.append(m_alpha)
            self.BETA.append(m_beta)
            self.GAMMA.append(m_gamma)
            self.MMD.append(m_mmd)
            #----
            print(f'epoch {self.epoch} - ELBO: {m_elbo:.3f} - RECON. LOSS: {m_recon_loss:.3f} - '+\
                  f'KL: {m_kl:.3f} - alpha: {m_alpha:.3f} - beta: {m_beta:.3f} - gamma: {m_gamma:.3f} - mmd: {m_mmd:.3f}')
            
            #---save checkpoints....
            if not self.save_model:
                pass
            else:
                if not (self.epoch % self.save_model_iter) == 0:
                    pass
                else:
                    #store all model metric in a dictionary...
                    self.loggers = {
                                    'elbo': self.ELBO,
                                    'reconstruction': self.RECON_LOSS,
                                    'kl_div': self.KL_DIV,
                                    'alpha': self.ALPHA,
                                    'beta': self.BETA,
                                    'gamma': self.GAMMA,
                                    'mmd': self.MMD
                                    }
                    model_saver_2d(self,\
                                self.test,\
                                self.hidden_dim,\
                                self.latent_dim,\
                                self.batch_size,\
                                self.beta,\
                                self.gamma,\
                                self.dist,\
                                self.vloss,\
                                self.lr,\
                                self.epoch,\
                                self.architecture,\
                                self.mmd_type,\
                                self.datatype)
            
            #save intermediate latent space if requested
            if not self.save_latent:
                pass
            else:
                _, _, z = self.model.encoder.predict(self.test, batch_size = self.batch_size) #predict the test data
                self.int_z[f'epoch{self.epoch}'] = z
            if self.intermediate:
                #----save intermediate latent space generated...
                if len(self.ELBO) == 1:
                    #set the latent space to the first index ELBO
                    if self.ELBO[-1] > 0:
                        _, _, z = self.model.encoder.predict(self.data, batch_size = self.batch_size)
                        self.z_latent_pos = z
                    else:
                        _, _, z = self.model.encoder.predict(self.data, batch_size = self.batch_size)
                        self.z_latent_neg = z
                elif len(self.ELBO) > 1:
                    if self.ELBO[-1] > 0:
                        _, _, z = self.model.encoder.predict(self.data, batch_size = self.batch_size)
                        self.z_latent_pos = z
                    elif self.ELBO[-1] < 0 and self.ELBO[-2] > 0:
                        _, _, z = self.model.encoder.predict(self.data, batch_size = self.batch_size)
                        self.z_latent_neg = z
                    else:
                        pass
            else:
                pass
            #---stopping criterion
            if not self.stopping:
                pass
            else:
                if self.pid_a and not self.pid_b:
                    if not len(self.ALPHA) == 1:
                        if np.abs(self.ALPHA[-1] - self.ALPHA[-2]) < self.epsilon_a:
                            #store all model metric in a dictionary...
                            self.loggers = {
                                            'elbo': self.ELBO,
                                            'reconstruction': self.RECON_LOSS,
                                            'kl_div': self.KL_DIV,
                                            'alpha': self.ALPHA,
                                            'beta': self.BETA,
                                            'gamma': self.GAMMA,
                                            'mmd': self.MMD
                                            }
                            return self
                    else:
                       pass
                elif self.pid_b and not self.pid_a:
                    if not len(self.BETA) == 1:
                        if np.abs(self.BETA[-1] - self.BETA[-2]) < self.epsilon_b:
                            #store all model metric in a dictionary...
                            self.loggers = {
                                            'elbo': self.ELBO,
                                            'reconstruction': self.RECON_LOSS,
                                            'kl_div': self.KL_DIV,
                                            'alpha': self.ALPHA,
                                            'beta': self.BETA,
                                            'gamma': self.GAMMA,
                                            'mmd': self.MMD
                                            }
                            return self
                    else:
                       pass
                elif self.pid_a and self.pid_b:
                    if not len(self.ALPHA) == 1:
                        if np.abs(self.ALPHA[-1] - self.ALPHA[-2]) < self.epsilon_a and np.abs(self.BETA[-1] - self.BETA[-2]) < self.epsilon_b:
                            #store all model metric in a dictionary...
                            self.loggers = {
                                            'elbo': self.ELBO,
                                            'reconstruction': self.RECON_LOSS,
                                            'kl_div': self.KL_DIV,
                                            'alpha': self.ALPHA,
                                            'beta': self.BETA,
                                            'gamma': self.GAMMA,
                                            'mmd': self.MMD
                                            }
                            return self
                    else:
                       pass
        #store all model metric in a dictionary...
        self.loggers = {
                        'elbo': self.ELBO,
                        'reconstruction': self.RECON_LOSS,
                        'kl_div': self.KL_DIV,
                        'alpha': self.ALPHA,
                        'beta': self.BETA,
                        'gamma': self.GAMMA,
                        'mmd': self.MMD
                        }
        #return train model as model
        return self













