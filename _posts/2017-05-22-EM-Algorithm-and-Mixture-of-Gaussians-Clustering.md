---
layout: post
title: "The Gaussian Mixture Model and the EM Algorithm"
date: 2017-05-22
comments: false
categories: 
---

This post is about the Gaussian mixture model which is a generative probabilistic model with hidden variables and the EM algorithm which is the algorithm used to compute the maximum likelihood estimate of its parameters.

<h2><font size="5">The Gaussian Mixture Model</font></h2>

As stated earlier the Gaussian mixture model is a probabilistic model that assumes that the observed data points can be expressed as a convex combination (mixture) of hidden Gaussian distributions.  You can also look at it from a clustering perspective - all data points belong to clusters (expressed as Gaussian distributions) with different probabilities.  However, all that is observed is the data points themselves and the underlying Gaussian distributions from which they are assumed to be generated are hidden/latent variables that we must infer.  It is also important to note that we are NOT trying to infer the number of Gaussian distributions - we assume that that is known before hand and we call that $$K$$.

So to express this more concretely, let's assume we have $$N$$ i.i.d. datapoints

$$D=\{\mathbf{x}_1,...,\mathbf{x_N}\}, \qquad \mathbf{x_i} \in \mathbb{R}^d$$

and also assume there are $$K$$ hidden Gaussian distributions.

$$\mathcal{N}(\mathbf{\mu_k}, \mathbf{\Sigma_k}) \qquad \forall \enspace k=1,...,K$$

Each with hidden mean $$\mathbf{\mu_k}$$ and covariance $$\mathbf{\mathbf{\Sigma_k}}$$.  These are the hidden parameters that we are aiming to infer.

Let's also introduce a hidden variable $$\mathbf{z}$$ which governs the assignment of data points to Gaussian distributions.  So the probability that a data point $$\mathbf{x_i}$$ is assigned to the $$k^{th}$$ distribution is

$$p(z_i=k)=\alpha_k, \qquad \sum_{k=1}^K \alpha_k = 1$$

and $$\alpha_k$$ are called the *mixing coefficients*.  Intuitively, these values say which Gaussian distribution the data point is *closest* to.

Now we can express the likelihood of a single data point $$\mathbf{x_i}$$ as


$$\begin{align}
p(\mathbf{x_i}) &= \sum_z p(\mathbf{x_i}|z)p(z) \\
&= \sum_z p(\mathbf{x_i} | z_i=k)p(z_i=k) \\
&= \sum_z \alpha_k \mathcal{N}(\mathbf{x_i}|\mathbf{\mu_k},\mathbf{\Sigma_k})
\end{align}$$

From this you can see how we are expressing each data point as a convex combination of Gaussians.

Also, the joint distribution of a datapoint and it's assignment can be expressed as 

$$p(\mathbf{x},z)=\prod_{k=1}^K \alpha_k^{z_k}\mathcal{N}(\mathbf{x}|\mathbf{\mu_k},\mathbf{\Sigma_k})^{z_k}$$

and it is important to note that $$z_k$$ is being used here rather than $$p(z_k)$$ and $$z_k$$ is binary valued (it is either assigned to that Gaussian or it isn't), so the Gaussians that aren't assigned to that datapoint will just contribute a $$1$$ to the product.  

So for the entire dataset, the likelihood is

$$p(D,\mathbf{z})=\prod_{i=1}^N\prod_{k=1}^K \alpha_k^{z_k}\mathcal{N}(\mathbf{x_i}|\mathbf{\mu_k},\mathbf{\Sigma_k})^{z_k}$$

<h2><font size="5">The Goal</font></h2>

As stated earlier, the goal here is to compute the maximum likelihood estimate of the hidden Gaussian parameters $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and  $$\alpha_k$$ for all $$K$$.  The typical procedure is to take the derivative of the log of the likelihood function with respect to the parameter, set it to zero, then try to solve for the parameter.  If we do this we get the following expressions for $$\mathbf{\mu_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$.

$$\mathbf{\mu_k} = \frac{\sum_{i=1}^N p(z_i = k | \mathbf{x_i})\mathbf{x_i}}{\sum_{i=1}^N p(z_i = k | \mathbf{x_i})}, \qquad \mathbf{\Sigma_k}= \frac{\sum_{i=1}^N p(z_i = k | \mathbf{x_i})(\mathbf{x_i} - \mathbf{\mu_k})(\mathbf{x_i} - \mathbf{\mu_k})^T}{\sum_{i=1}^N p(z_i = k | \mathbf{x_i})}, \qquad \mathbf{\alpha_k} = \frac{1}{n} \sum_{i=1}^N p(z_i = k | \mathbf{x_i})$$

As you can see, each expression contains the term $$p(z_i = k\mid\mathbf{x_i})$$.  Using Bayes rule, this can be written as

$$p(z_i = k\mid\mathbf{x_i}) = \frac{\alpha_k \mathcal{N}(\mathbf{x_i}|\mathbf{\mu_k},\mathbf{\Sigma_k})}{\sum_{k'}\alpha_{k'} \mathcal{N}(\mathbf{x_i}|\mathbf{\mu_{k'}},\mathbf{\Sigma_{k'}})}$$

So the problem is that $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$ are written in terms of the unknown $$p(z_i = k\mid\mathbf{x_i})$$ and conversely $$p(z_i = k\mid\mathbf{x_i})$$ is written in terms of the unknowns $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$.  So we cannot solve for anything because every unknown is written in terms of another unknown so we are stuck.  In order to get unstuck we use the EM algorithm.

<h2><font size="5">The EM Algorithm</font></h2>

The EM (*Expectation-Maximization*) algorithm proposes a solution to this problem.  Starting from a random initialization of the parameters $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$ the EM algorithm iterates between the following steps until convergence.

<hr>
&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<b><font size="5">E-Step:</font></b> Evaluate $$p(z_i = k\mid\mathbf{x_i})$$ based on the latest values of $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$.

&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;<b><font size="5">M-Step:</font></b> Evaluate $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$ based on the latest value of $$p(z_i = k\mid\mathbf{x_i})$$.
<hr>

Upon convergence you will get a maximum likelihood estimate of the parameters $$\mathbf{u_k}$$, $$\mathbf{\Sigma_k}$$ and $$\alpha_k$$, but it is not guaranteed to be the global optimum.  It is, however, guaranteed that the likelihood will increase at every step.

In this way the EM algoritham can be viewed as a kind of coordinate ascent algorithm - starting with a completely random guess, we are re-estimating an unknown parameter based on our best guess of the parameters it depends on and we iteratively move in the right direction.

Formally, the E-step is defined as evaluating the expected value of the log likelihood function (hence the name *expectation-maximization*) but in many cases such as ours, all that is really necessary is evaluating the simpler expression of $$p(z_i = k\mid\mathbf{x_i})$$.  In general, the EM algorithm is applicable whenever you are trying to find the maximum likelihood estimate of the parameters within models with hidden variables and particularly those that involve distributions from the exponential family.  Another well-known example of the EM algorithm is the [Baum-Welch algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm) which is the EM algorithm applied to the hidden Markov model.

Thank you for reading.

<h2><font size="5">References</font></h2>
* [The EM Algorithm](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm)
* [The Gaussian Mixture Model](http://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf)