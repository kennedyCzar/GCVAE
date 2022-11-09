## GCVAE: Generalized-Controllable Variational Autoencoder

### ABSTRACT: 
```
Variational autoencoders (VAEs) have recently been used for unsupervised disentanglement learning of complex density
distributions. Numerous variants exist to encourage disen tanglement in latent space while improving reconstruction.
However, none have simultaneously managed the trade-off between attaining extremely low reconstruction error and a
high disentanglement score. We present a generalized framework to handle this challenge under constrained optimization
and demonstrate that it outperforms state-of-the-art existing models as regards disentanglement while balancing recon-
struction. We introduce three controllable Lagrangian hyperparameters to control reconstruction loss, KL divergence loss
and correlation measure. We prove that maximizing information in the reconstruction network is equivalent to information
maximization during amortized inference under reasonable assumptions and constraint relaxation.
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
[3] Set parameters ad run (The current parameters are optmimal for experiment)
```python
python main_mnist_2d.py
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
