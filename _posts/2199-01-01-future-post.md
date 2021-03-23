---
title: 'VAE with Normalizing Flows'
date: 2021-03-22
permalink: /posts/2021/03/blog-post-001/
tags:
  - variational autoencoder
  - normalizing flows
---


## Summary
The simple families of posterior approximations employed by many Variational Autoencoder (VAE) applications could be too limited to model the true posterior distributions properly. In this post we will explore one way to overcome this limitation- approximation through a normalizing flow.

## From AE to VAE: Variational Bayes

Before the idea of VAE was first introduced by Kingma & Welling in 2014, Autoencoder(AE) was promoted by Hinton & Salakhutdinov in 2006. AE is a neural network capable of compressing inputs (usually with high dimensionality) into compact latent representations, and reconstructing the original input using such representations. 
In contrast to AE, VAE incorporates the idea of variational Bayes and maps the input into a distribution q instead of a vector. Approximating the posterior p(z|x) with q(z|x), VAE becomes a powerful generative model: the decoder is now optimized to recover latent representations from a probability distribution from which we can sample data at inference time.

## One limitation of VAE: choice of q

Despite the great successes, variational methods have several disadvanages that limit their power in statistical inference. One of these limitations is the choice of q, or more precisely, the class of probability distributions we choose to approximate the true posterior. In a vanilla VAE, fully factorized diagonal Gaussians are used, the core concept of which is :

$$ p(\theta|X) \approx q(\phi) = \prod_{i=1}^{n} q_i(\phi _i) $$

Obviously, it can't model every distribution- the fully factorization assumption limits its ability to match complicated true posteriors- e.g., multimodal distributions, which cannot be modelled with basic Gaussians. As a result, we might be optimizing our network at the expense of higher reconstruction errors when approximating the true posterior with mismatching proposed distributions.

Theoretically, better posterior approximations will result in better performance. How to choose a more complex distribution? We want it to also have the good properties of factorized Gaussians: 

  1. computationally efficient to differentiate
  2. easy to sample mini batches from

## VAE with normalizing Flows: posterior approximations with controllable complexity at run time
 from dist to dist
how, loss, some flow methods, applications

  1. Invertible linear-time transformations
  2. Autoregressive flows
  3. Glow: generative flow with invertible 1x1 convolutions

## Conclusion

## References
