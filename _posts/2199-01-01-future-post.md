---
title: 'VAE with Normalizing Flows'
date: 2021-03-22
permalink: /posts/2021/03/blog-post-001/
tags:
  - Variational Autoencoder
  - Normalizing Flows
---
## Summary
	The simple families of posterior approximations employed by many Variational Autoencoder (VAE) applications could be too limited to model the true posterior distributions properly. In this post we will explore one way to overcome this limitation- approximation through a normalizing flow.

## Why is VAE so popular?

Before the idea of VAE was first introduced by Kingma & Welling in 2014, Autoencoder(AE) was promoted by Hinton & Salakhutdinov in 2006. AE is a neural network capable of compressing inputs (usually with high dimensionality) into compact latent representations, and reconstructing the original input using such representations. 
In contrast to AE, VAE maps the input into a distribution q instead of a vector with no extra constraints. Approximating the posterior p(z|x) with q(z|x), VAE becomes a powerful generative model as the decoder is optimized to recover latent representations from a probability distribution from which we can sample data at inference time, rather than discrete, possibly non-interpolatable vectors.

## One limitation of VAE: choice of q


## VAE with normalizing Flows: posterior approximations with controllable complexity at run time
 from dist to dist
how, loss, some flow methods, applications

  1. Invertible linear-time transformations
  2. Autoregressive flows
  3. Glow: generative flow with invertible 1x1 convolutions

## Conclusion

## References
