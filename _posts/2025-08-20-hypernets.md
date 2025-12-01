---
layout: post
title: New POVs on hypernetworks
date: 2025-08-20
description: Hypernetworks from the POV of INRs and functional similarity 
tags: ml
categories: ml
featured: false
---

### TLDR
**hypernetworks can be interpreted to**
- **generate implicit neural representations, and**
- **quantify functional similarity between models**

<hr>

I've worked on hypernetworks (neural networks that parameterize other neural networks) for a little while, that has led to <a href="https://openreview.net/forum?id=dyRHRxcgXX">a workshop paper at ICLR 2025</a> and <a href="https://arxiv.org/pdf/2507.10015">a main conference paper at EMNLP 2025</a>. While working on these papers, I've had the time to think about hypernetworks in a couple of fascinating ways that I've described below.

### Hypernetworks as generators of INRs

*Implicit neural representations* (INRs) are a clever and rather under-appreciated class of representations. Contrary to the popular approach of predicting representations, INRs *are* representations themselves: given $f(x)=y$, if we train a neural network $g_\theta$ to predict $y$ from $x$, the parameters of this network $\theta$ *implicitly represent* the function $f(\cdot)$.

Hence, when learning several functions $f_1, \dots, f_n$, we can decide to predict weights (or INRs) for $f_i$, using a generating function $H(\cdot)$ as $H(i) = \theta_i$ such that $g_{\theta_i}(x) = f_i(x)$. From the deep learning perspective, this generating function $H(\cdot)$ is called a hypernetwork.

This ability to conditionally generate INRs is what makes hypernetworks strongly applicable in physics-informed machine learning. In particular, we can see how the process is just another formuation of partial differential equations (PDEs), which denote time-varying functions of space $\nu_t(x)$ in differential form. 

Hypernetworks can very well predict spatial functions given a timestep: $H(t) = \theta_t$ such that $g_{\theta_t}(x) = \nu_t(x)$ as they can conditionally generate an INR corresponding to the function at tilmestep $t$, and can be used to forecast PDEs. Indeed they've been used to do so in a very interesting <a href="https://arxiv.org/abs/2209.14855">NeurIPs paper</a> that I've presented in a tutorial <a href="https://jaisidhsingh.github.io/assets/DINo_continuous_pde_forecasting_with_INRs.html">here</a>.

### Hypernetworks as quantifiers of functional similarity

Let's say that we wish to know the functional similarity between two encoders $f_A, f_B: \mathbb{R}^{m} \to \mathbb{R}^{d}$. The most straightforward way to do this would be to collect a stack of encodings for $N$ inputs for each encoder and use a similarity function like Centered-Kernel-Alignment (CKA) on these features: $$s_1 = CKA(O_A, O_B) \ ; \quad O_A, O_B \in \mathbb{R}^{N\times d}$$
where $s_1 \in [0, 1]$ is a score denoting the functional similarity between the two models, and $O_A, O_B$ are the stacks of encoders for $N$ inputs.

However, let's think of another way to quantify the functional similarity between $f_A$ and $f_B$, specifically, by using a hypernetwork $H$ that predicts linear classifiers $W_A, W_B \in \mathbb{R}^{d\times k}$ on top of the two encoders $f_A, f_B$. Here, $k$ is the number of classes. 

Writing, the composite function $g_j(\bullet) = W_j f_j(\bullet), \ j \in \{A, B\}$ to classify the inputs, we outline the scheme that let's the hypernetwork depict functional similarity between $f_A$ and $f_B$ as follows:

- Given a dataset $\mathcal{D}$ and the current state of the hypernetwork's parameters $\phi^0$, predict $W^{0}_A$, $W^{0}_B$ = $H_{\phi^0}(c_A)$, $H_{\phi^0}(c_B)$
- Obtain the classification loss of $g^0_A$ and $g^0_B$ on $\mathcal{D}$ as $\ell^0_A$ and $\ell^0_B$.
- Train the hypernetwork's parameters $\phi$ only on encoder $f_A$ as $$\phi^1 \leftarrow \phi^0 - \eta \nabla_{\phi^0}\ell^0_A$$
- Predict $W^{1}_A, W^{1}_B = H_{\phi^1}(c_A), H_{\phi^1}(c_B)$.
- Obtain the classification loss of  $g^1_A, g^1_B \text{ as } \ell^1_A, \ell^1_B$.
- Then, the magnitude of $\Delta = \ell^0_B - \ell^1_B$ depicts the functional similarity between encoders $f_A$ and $f_B$.

In other words, if training the hypernetwork *only* using network A lowers the loss of network B as well, then network A and B can be called functionally similar.

<hr>

The above interpretations of hypernetworks are avenues that I think are rather under-utilised in the current literature, and can offer creative and potentially powerful ways of modulating neural networks. I'd be happy to know what you think of the above, or about parameter prediction in general. Hit my up for a chat on <a href="https://x.com/jaisidhsingh"> X/twitter</a> if you like.
