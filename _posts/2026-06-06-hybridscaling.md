---
layout: post
title: What Hybrid Models Mean for Scaling 
date: 2026-06-06
description: Understanding how hybrid models lie in various spectra between dense and linear attention.  
tags: ml
categories: architecture
---

## Introduction
<br>
---

The bitter lesson that we get from scaling is that you can gain more loss reduction from scaling up your model as opposed to relying on domain-specific heuristics. This directly allows us to define notions of algorithmic "goodness" and "generality" - a method is "good" if it leverages computation well within the current paradigm of hardware, and it is "general" if it allows one to leverage compute maximally in the limit (better hardware and different paradigms of computation becoming accessible over time).

But what does it mean to leverage compute well? Can we study this with simple mathematics in a way that is easy to understand? How can we use these mathematics as context to interpret the most dominant computational primitive in neural networks, i.e, attention, with the lens of compute? Can this interpretation explain recent breakthroughs in LLM architecture subject to present-day constraints of training LLMs? 

This blog seeks to provide lightweight answers to these questions, given the recent advent of *hybrid LLMs*. First, we will take a look at the two most popular attention mechanisms (dense and linear) using asymptotic forms of the compute required for each. These forms will allow us to define quantities that will present a spectrum of properties between dense and linear attention. Finally, we will explain how hybrid LLMs allow us to traverse this spectrum, and beat the worst case every time. 

## Attention from the Perspective of Compute
<br>
---

To better understand what it means to leverage compute well, let us briefly bring our attention to the core unit responsible for compute: GPUs. At a high level, GPUs perform arithmetic operations in massive parallelism using streaming multiprocessors (SM) that each have low-latency on-chip caches via an L1 cache and SRAM (shared memory). SMs access data from a special kind of memory called high bandwidth memory (HBM) that is physically bonded to the GPU, and this is where data is initially stored and ultimately written back. However, accessing HBM is quite costly which is why an L2 cache is used an intermediate level between each SM's on-chip memory and HBM (which is off-chip). Typically, accessing data from HBM is about $10×$ and $5×$ slower than using L1 cache/SRAM and L2 cache respectively.

This tells us that along with the number and peak rate of operations performed, one must also consider how operands are placed in order to understand intensity of compute utilization. Moreover, since model architecture determines these patterns of computation and data movement, understanding how LLMs can be efficiently scaled requires studying the main architectural primitive, i.e., the attention mechanism under the lens of compute.

We now describe the two main types of attention mechanisms investigated in this thesis: (i) dense, and (ii) linear attention.

Given a $\mathbf{X} ∈ \mathbb{R}^(n\times d)$ sequence of $n$ tokens embedded in $d$ dimensions, one uses three weight matrices $\mathbf{W}^q, \mathbf{W}^k, \mathbf{W}^v \in \mathbb{R}^{d\times h\times (d/h)}$ to project $\mathbf{X}$ to $\mathbf{Q}, \mathbf{K}, \text{ and } \mathbf{V}$ as
%
$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^{q},
\qquad
\mathbf{K} = \mathbf{X}\mathbf{W}^{k},
\qquad
\mathbf{V} = \mathbf{X}\mathbf{W}^{v}.
$$
%
Subsequently, these are fed into a formula $\mathcal{F}(\bullet,\circ,\cdot)$ that computes the attention output. It is the expression of this formula in which dense and linear attention differ, given respectively by

$$
\mathcal{F}_{\text{dense}}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\operatorname{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}
$$

$$
\mathcal{F}_{\text{linear}}(\mathbf{Q},\mathbf{K},\mathbf{V})
=
\phi(\mathbf{Q})
\left(
\phi(\mathbf{K})^{\top}\mathbf{V}
\right)
$$

where $\phi(·)$ is a learnable kernel with its own parameter $\mathbf{W}^{\phi} \in \mathbb{R}^{d\times d}$. Notice how only dense attention computes the product of queries and keys $\mathbf{Q}\mathbf{K}^\top$, an operation that is quadratic in $n$. Asymptotically, the compute $C$, measured by number of floating point operations (FLOPs), required by each formula follows

$$
C_{\text{dense}}
\sim
n^2 d + n d^2
\quad \text{and} \quad
C_{\text{linear}}
\sim
n d^2.
$$

We now introduce three quantities to understand the differences between dense and linear attention.

1. **Compute per Parameter (CPP).** Dividing the asymptotic expression for the FLOPs needed by $\mathcal{F}$ by that of the parameter count, i.e. $d^2$, yields effective computations performed per parameter.

$$
CPP_{\text{dense}}
=
\frac{n^2 d + n d^2}{d^2}
=
n + \frac{n^2}{d}
$$

$$
CPP_{\text{linear}}
=
\frac{n d^2}{d^2}
=
n
$$

2. **Model Capacity under Fixed Compute (MCC).** This is defined as the model dimension $d$ written as a function of $n$ under fixed pre-defined compute $C$.

$$
C \sim n^2 d + n d^2 \implies d \sim \frac{C}{n^2} \ \text{so } \ MCC_{\text{dense}} = n^{-2} \quad \text{assuming} n \gg d
$$

$$
C \sim n d^2 \implies d \sim \sqrt{\frac{C}{n}} \ \text{so } MCC_{\text{linear}} = n^{-1/2}.
$$

3. **Arithmetic Intensity (AR).** This is defined as the ratio of the FLOPs required by an operation to the bytes of memory accessed by it.

$$
AR_{\text{dense}}
\propto
\frac{n^2 d + n d^2}
     {n d + d^2 + n^2 h}
\approx
\frac{d}{h}
\quad \text{assuming } n \gg d.
$$

$$
AR_{\text{linear}}
\propto
\frac{n d^2}
     {n d + d^2 + d^2/h}
=
\frac{n}
     {(n/d)+1+(1/h)}
\approx
d.
$$

CPP and MCC show us a spectrum on which dense and linear attention lie at opposite ends: dense attention leverages more compute at the same number of parameters, while linear attention affords a higher effective model dimension given the same compute and context length. Also, since linear attention has an arithmetic intensity $h×$ that of dense attention, it implies significant memory savings while performing the same amount of compute.

## How the Scaling Era Informs Architecture Design
<br>
---

Today, we face different constraints than we did 2 years ago because compute is more available than ever and now the commodity that constrains us is high-quality data. Therefore in 2026, the strategy for scaling has changed: instead of scaling up all of $N, D$, and $C$ according to known scaling laws, the frontier of LLMs faces the incentive to find architectures that meet our data availability constraints. Therefore, practitioners are now motivated towards architectures that can match the performance of dense attention transformers with less data. Recently, *hybrid LLMs* have shown strong results in this space, now arriving at the forefront of popular open-source language models.

Hybrid LLMs earn their namesake because they consist of both dense attention layers as well as linear attention layers. Specifically, one dense attention layer is interleaved after every $r$ linear attention layers, where $r$ is known as the *hybridization ratio*.

Let us use the quantities defined in the previous section to understand the benefits afforded by hybrid LLMs. First, we write the asymptotic dependence of the number of FLOPs required by a hybrid LLM under hybridization ratio $r$:

$$
C_{\text{hybrid}}^{(r)}
\sim
\frac{
n^2 d + n d^2 + r n d^2
}
{r+1}
=
n d^2 + \frac{n^2 d}{r+1}.
$$

Hence, compute per parameter for the hybrid LLM can be written as $CPP_{\text{hybrid}}^{(r)} = n + \frac{n^2}{d(r+1)}$. Clearly, the LLM relies purely on linear attention as $r$ increases, and $CPP_{\text{hybrid}}^{(r)} \to CPP_{\text{linear}} \text{ as } r \to \infty$. Furthermore, $CPP_{\text{hybrid}}^{(r)} \to CPP_{\text{dense}}$ as $r \to 0$.

Similarly, $MCC_{\text{hybrid}}^{(r)} = (r+1)\,n^{-2}$, which can be found by solving a quadratic equation in $d$ and using a first-order Taylor expansion of the square root function (assuming $n \gg d$). This is quite significant: hybridization ratio $r$ gives us an $(r+1)\times$ larger effective model dimension than dense attention under fixed compute and context length.

For arithmetic intensity, we suppose that the depth of the model is $L$. Then, $L/(r+1)$ layers will use dense attention and $rL/(r+1)$ layers will use linear attention. Thus, $AR_{\text{hybrid}}^{(r)}$ defined as the following weight average will approach $AR_{\text{dense}}$ as $r\to0$ and $AR_{\text{linear}}$ as $r\to\infty$. Clearly, the hybridization ratio $r$ affords a $(1+r\cdot h)/(1+r) \times$ gain over the worst.

$$
AR_{\text{hybrid}}^{(r)} = \frac{\frac{L}{r+1}\frac{d}{h} + \frac{rL}{r+1}d}{L}= \frac{d}{h}\left(\frac{1+r h}{1+r} \right).
$$

Therefore, hybrid LLMs allow us to traverse the spectrum of compute-relevant properties defined between dense and linear attention, and let us improve over the worst case in all three defined properties. Particularly, hybrid LLMs can allow us to trade off the weaknesses of linear attention in terms of CPP as well as the data-hunger of dense attention via MCC under fixed compute and long contexts. In practice, hybrid LLMs perform similarly: we find them to be approximately $40\%$ more data efficient than dense LLMs while outperforming them by a significant margin. Predictably, hybrid LLMs still do not scale as well with parameters as dense LLMs do; however, projections show strong parameter savings up to $1T$ tokens for hybrid models, as well as increasing data savings across all parameter scales.
