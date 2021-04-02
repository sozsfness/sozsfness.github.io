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

Before the idea of VAE was first introduced by [Kingma & Welling in 2014](https://arxiv.org/abs/1312.6114), Autoencoder (AE) was promoted by [Hinton & Salakhutdinov in 2006](https://www.cs.toronto.edu/~hinton/science.pdf). AE is a neural network capable of compressing inputs (usually with high dimensionality) into compact latent representations, and reconstructing the original input using such representations. 
In contrast to AE, VAE incorporates the idea of variational Bayes and maps the input into a distribution q instead of a vector. Approximating the posterior p(z|x) with q(z|x), VAE becomes a powerful generative model: the decoder is now optimized to recover latent representations from a probability distribution from which we can sample data at inference time.

## One limitation of VAE: choice of q

Despite the great successes, variational methods have several disadvanages that limit their power in statistical inference. One of these limitations is the choice of q, or more precisely, the class of probability distributions we choose to approximate the true posterior. In a vanilla VAE, fully factorized diagonal Gaussians are used, the core concept of which is :

$$ p(\theta|X) \approx q(\phi) = \prod_{i=1}^{n} q_i(\phi _i) $$

Obviously, it can't model every distribution- the fully factorization assumption limits its ability to match complicated true posteriors, e.g., multimodal distributions, which cannot be modelled with basic diagonal Gaussians. 

Theoretically, better posterior approximations will result in better performance, so we need to choose a more flexible distribution while keeping the good properties of factorized Gaussians: 

  1. computationally efficient to differentiate
  2. easy to sample mini batches from


## Variational inference with normalizing Flows: posterior approximations with controllable complexity at run time

The idea of approximating posterior distributions using normalizing flows was fist introduced by [Rezende & Mohamed in 2015](https://arxiv.org/abs/1505.05770). In short, it transforms a probability density through a sequence of invertible mappings, and the density "flows" through the sequence, resulting in a more flexible distribution that hopefully could better match the true posterior.

### How it works
Given a random variable $$z$$ with density function $q_z(z)$, we can transform it into another random variable $z'$ with the same dimensionality using an invertible mapping $f$ with inverse $f^{-1} = g$. $z'$ has a density function: 

$$ q_{z'}(z') = \frac{\partial Pr(Z' \leqslant z')}{\partial z'} $$

Which is equal to:

$$ \frac{\partial Pr(f(z) \leqslant z')}{\partial f(z)} $$
      
Applying the inverse mapping, we get:

$$ \frac{\partial Pr(z \leqslant g(z'))}{\partial f(z)} $$

Applying the chain rule,

$$ = \frac{\partial Pr(z \leqslant g(z'))}{\partial z} * \frac{1}{f'(z)} $$

$$ = q_z(g(z')) * |det(\frac{\partial f(z)}{\partial z})^{-1}| $$
      
$$ = q_z(z) * |det({\frac{\partial f(z)}{\partial z}})|^{-1} $$
 
The last line implies that we don't need to compute the inverse of our mappings explicitly.

### Incorporating a normalizing flow into a vanilla VAE
To see how it work in practice, let's start with a simple posterior approximation- fully factorized Gaussians used in vanilla VAEs:

$$ p(\theta|X) \approx q(\phi) = \prod_{i=1}^{n} {q_i}(\phi_i) $$
 
where $ {q_i}(\phi_i)$ follows $ N(\mu_i, \sigma_i) $ i.i.d. for all i.

Now, we will apply a series of invertible transformations $f_t$s with sequence length T.

$ z_0 \sim q(\phi) $, $ z_t = f_t(z_{t-1}) \ \forall t=1...T $

For simplicity, we'll use log expressions onwards. Recall from derivations above that after applying transformation f to z, log of the density function for the new variable becomes:

$$ log(q_z(z)) - log(|det({\frac{\partial f(z)}{\partial z}})|) $$

So after applying T transformations, the log density of the final variable becomes:

$$log(q_{\phi}(z_0)) - \sum_{t=1}^{T} log(|det({\frac{\partial f_t(z_{t-1})}{\partial z_{t-1}}})|) $$

Note that the above function is differentiable and does not require computation of the inverse functions explicitly. Moreover, since the original density $q(\phi)$ remains unchanged, we can still sample from a known Gaussian while being able to model more complex posteriors. The best part is that we don't need to modify our loss function, as the KL divergence we calculate is still between $q(\phi)$ and a known Gaussian. 

### So what normalizing flow should we use?
There are two types of flows: infinitesimal and finite. A infinitesimal flow is a flow with length that tends to infinity. In this case, it is not described as a sequence of transformations, but as a partial differential equation that shows how the initial density function evolves over time. In this post, we will be focusing on the finite flows as they are more straightforward and have been introduced in the example above. 

### Invertible linear-time transformations

This family of transformation takes the form:

$$ f(z) = z\ +\ u*h(w^T z + b) $$

where $w ,\ u \in \mathbf{R}^D$ , $b \in \mathbf{R}$ are the learnable parameters and $h$ is a smooth non-linearity with derivative $h'$ .

$$det(|\frac{\partial f(z)}{\partial z}|) = |det(\mathbf{I} + uw^T h'(w^T z+b)^T)|$$

Using the matrix determinant lemma, we get:

$$|det(\mathbf{I} + uw^T h'(w^T z+b)^T)| = |1 + w^T h'(w^T z+b)^T u|$$

Therefore, starting with a fully factorized Gaussian: 

$ z_0 \sim q(\phi) $, $ z_t = f_t(z_{t-1}) \ \forall t=1...T $

Applying T invertible linear-time transformations $f_t$s, the density function for our new random variable becomes:

$$ log({q_T}(z_T)) = log(q_{\phi}(z_0)) - \sum_{t=1}^{T} log(|1 + {w_t}^T h'({w_t}^T z_{t-1}+b_t)^T u_t|) $$

## Conclusion

Learning transformations of simple density functions, e.g., fully factorized Gaussians, could help the existing variational inference approaches to model complex true posteriors more precisely. Meanwhile, it's straightforward to combine VAEs with normalizing flows, as the transformation is done in the latent space with the density of the simple approximation distribution flowing through the sequence, so stochastic backpropagations and monte carlo sampling can still be used as they are in vanilla VAEs. 

There are more types of flows that allow for different charasteristics of posteriors, i.e., the Hamiltonian flow, and I hope to explore them in future posts.

## References
Rezende, Danilo, and Shakir Mohamed. "Variational inference with normalizing flows." International Conference on Machine Learning. PMLR, 2015.

Salimans, Tim, Diederik Kingma, and Max Welling. "Markov chain monte carlo and variational inference: Bridging the gap." International Conference on Machine Learning. PMLR, 2015. (section 5 for Hamiltonian variational inference)

Dinh, Laurent, David Krueger, and Yoshua Bengio. "Nice: Non-linear independent components estimation." arXiv preprint arXiv:1410.8516 (2014).

Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

Hinton, Geoffrey E., and Ruslan R. Salakhutdinov. "Reducing the dimensionality of data with neural networks." science 313.5786 (2006): 504-507.
