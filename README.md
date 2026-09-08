# RandAR on MNIST

This project implements a simplified **RandAR-style random-order autoregressive Transformer** on the MNIST dataset.

The goal is to understand the first principles behind random-order autoregressive generation and test whether a model trained only on randomly ordered complete images can generalize to tasks such as partial-image completion and inpainting and outpainting.

## Core idea

For an image represented by pixels

$$
x=(x_1,x_2,\ldots,x_N),
$$

the joint probability can be factorized using the probability chain rule:

$$p(x)=
\prod_{i=1}^{N}
p
\left(
x_{\pi_i}
\mid
x_{\pi_{\lt i}}
\right)
$$

where $\(\pi\)$ is a permutation of the pixel locations.

A conventional autoregressive image model normally uses one fixed order, such as raster order:

$$
x_1\rightarrow x_2\rightarrow x_3\rightarrow\cdots
$$

RandAR instead samples a different random permutation during training:

$$
x_{17}\rightarrow x_{503}\rightarrow x_{82}\rightarrow\cdots
$$

The joint distribution is unchanged, but the model learns many different conditional distributions.

## Position instructions

Because pixel order is random, the model must know which location it is currently being asked to predict.

For each pixel $\(x_i\)$, we introduce a position instruction $\(P_i\)$.

A training sequence therefore looks like

$$
P_{\pi_1},x_{\pi_1},
P_{\pi_2},x_{\pi_2},
\ldots,
P_{\pi_N},x_{\pi_N}.
$$

Conceptually,

$$
P_i
$$

means:

> What is the pixel value at location \(i\)?

The model predicts $\(x_i\)$ from the position instruction and all previously revealed pixels.

## Training objective

At each position instruction, the Transformer predicts a probability distribution over possible pixel values.

The RandAR loss is negative log-likelihood:

$$\mathcal L=
-\sum_i
\log
p_\theta
\left(
x_{\pi_i}
\mid
P_{\pi_1},x_{\pi_1},
\ldots,
P_{\pi_i}
\right).
$$

In the MNIST implementation, grayscale values are discretized into intensity bins. Therefore the loss is implemented using categorical cross-entropy:

$$\boxed{\text{Cross Entropy}=
\text{Negative Log-Likelihood}
}
$$

for the discrete pixel-value distribution.

The complete training objective can be viewed as

$$\mathcal{L}=
\mathbb{E}_{\pi}
\left[
-\sum_i
\log
p_\theta
\left(
x_{\pi_i}
\mid
x_{\pi_{\lt i}},
P_{\pi_i}
\right)
\right]
$$

## Why random order matters

Random ordering exposes the model to many different conditioning sets.

For example, the same pixel $\(x_3\$) may be learned under

$$
p(x_3\mid x_1,x_2),
$$

or

$$
p(x_3\mid x_2),
$$

or other combinations depending on the sampled permutation.

Therefore the model learns the more general task

$$
\boxed{
p(x_i\mid x_S)
}
$$

where \(S\) can be many different subsets of known pixels.

This is the main source of RandAR's flexibility.

## Inference and image completion

At inference, any known subset of pixels can be placed first in the autoregressive context.

Suppose $\(S\)$ contains the visible pixels and $\(U\)$ contains the unknown pixels.

The model generates

$$
p(x_U\mid x_S)
$$

autoregressively:

$$p(x_U\mid x_S)=
\prod_j
p(
x_{u_j}
\mid
x_S,
x_{u_1},
\ldots,
x_{u_{j-1}}
).
$$

Each newly sampled pixel becomes additional context for the next prediction.

This allows the same model to perform:

* random missing-pixel completion,
* center-region inpainting,
* left/right-half completion,
* structured-mask completion,
* stochastic generation of multiple plausible reconstructions.

Importantly, the model is trained only with random-order complete images. Structured masks are introduced only during inference.

## Difference from a conventional Transformer

A standard autoregressive Transformer normally assumes a fixed sequence order:

$$
x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_N.
$$

RandAR separates two concepts:

$$
\text{generation order}
$$

from

$$
\text{original spatial location}.
$$

The generation order is determined by the random permutation $\(\pi\)$, while the position instruction $\(P_i\)$ tells the model which spatial location is being predicted.

Thus RandAR is still fully autoregressive and causal, but it is not restricted to one fixed autoregressive factorization.

## Model architecture

The MNIST implementation uses a small decoder-only causal Transformer.

Main components:

* MNIST \(28\times28\) grayscale images;
* 784 pixel locations;
* discretized grayscale pixel-value tokens;
* row and column embeddings for spatial location;
* explicit position-instruction embeddings;
* pixel-value embeddings;
* causal self-attention;
* Transformer blocks with LayerNorm and MLP layers;
* categorical output head over pixel-intensity bins;
* KV caching for faster autoregressive sampling.

The model sequence is approximately

$$
P_1,x_1,P_2,x_2,\ldots
$$

under a different random ordering for every training example.

## Main principle

The entire model can be summarized as

$$\boxed{\text{RandAR}=
\text{chain rule}
+
\text{random factorization order}
+
\text{position instruction}
+
\text{causal probabilistic sampling}
}
$$

The important consequence is that one model learns many conditional views of the same joint image distribution rather than being tied to one fixed generation order.
