---
layout: post
title: "Multilevel Linear Models"
date: 2020-03-14
comments: false
categories: 
---

Multilevel models (also called hierarchical models) are a class of statistical models that are applied to data that have a natural hierarchical or nested structure.  They are useful because they can often out-perform models that don't take this structure into account.  This post will cover multilevel models applied to linear regression however it can be easily extended to logistic regression and generalized linear models.

## Nested Data

Many datasets have a naturally nested or hierarchical structure.  Taking the [radon dataset](http://www.stat.columbia.edu/~gelman/arm/examples/radon/) as an example, the goal is to predict radon levels for a given house with the county and state of that house provided as potential input features.  The nested structure of these features is clear - a house is within a county and a county is within a state.  Since it is known that radon levels are highly correlated with location, these nested features are very important.  Intuitively, a model that takes into account the hierarchical structure of these features should outperform one that does not.

In terms of notation, when dealing with nested data, the training example is usually referred to as the _individual_ and given a subscript of _i_ and the larger heirarchical level that the training example falls into is called the _group_ which is given a subscript of _j_.  So in the radon example, the house is considered the _individual_ and the county and state are two _groups_.  Also, the notation subscript _j[i]_ refers to the group corresponding to individual _i_.

## Simple Multilevel Models

The question now becomes, how do we let the model know that our features have a nested structure? There are various ways of attempting this - some simple (and less effective) and some more complex (and more effective).

### 1. Total Pooling

The simplest approach is to ignore group-level features altogether and train your model with the remaining individual-level features.  This is called _total pooling_ where all the training examples are "pooled" together regardless of the groups they belong to.  The clear problem with this approach is that it throws away group-level features that might be useful - in terms of model selection, this approach tends to significantly underfit your model.

### 2. No Pooling

No pooling is the opposite idea which is to build a separate model for each group where each model is trained with only the training examples that are part of that group.  There are several problems with this approach as well such as

<ol style="margin-left: 25px">
  <li style="font-size:18px">What if a group has only a small number of training examples?</li>
  <li style="font-size:18px">What if the number of groups is so large such that training that many models becomes unscalable?</li>
  <li style="font-size:18px">Predictive power between groups is ignored.</li>
</ol>

In constrast to total pooling, no pooling tends to significantly overfit your model.

### 3. Indicator Variables

Using indicator variables (also called dummy variables) is a good compromise between total pooling and no pooling.  With indicator variables, a binary-valued feature is added to your model for each group indicating that individual's assignment to that group (essentially one-hot encoding).  The main advantage with this method is that you can use all training examples for a single model that takes into account the individual's group assignment.

However this approach has some drawbacks as well such as

<ol style="margin-left: 25px">
  <li style="font-size:18px">It requires a separate feature for each group.</li>
  <li style="font-size:18px">If some groups are small, weights corresponding to these indicator variables could be difficult to train.</li>
  <li style="font-size:18px">It ignores variance between groups.  Consider the radon dataset again, some counties are closer together than others yet with indicator variables they would be treated independently.  Wouldn't it make more sense for the model to take this similarity into account?</li>
</ol>

## Multilevel Model Intuition

Multilevel models aim to mitigate some of the drawbacks of the indicator variable approach.  Consider a simple linear regression model

$$y = X \beta$$

where $$y$$ is the output, $$X$$ is the input _design matrix_ of features and $$\beta$$ is the weight vector.  The features can be either at the individual level or the group level.  For example, in the radon dataset an individual-level feature could be the size of the house and a group-level feature could be the average Uranium level of the county that the house is in.  The idea behind multilevel models is to train separate weight coefficients for each group. With a multilevel model we would rewrite the the regression formula as

$$y_i = \alpha_{j[i]} + X_i \beta, \quad \forall \thinspace i = \{1, ...,n \}$$

$$\alpha_j = U_j \gamma, \quad \forall \thinspace j = \{1, ..., J \}$$

Here the $$X_i$$ matrix is still a design matrix but now it is only for the individual-level features.  In the bottom equation $$U_j$$ is the design matrix for the group-level features and $$\gamma$$ is the associated weight vector.  Finally $$\alpha_{j[i]}$$ represents the group-level contribution to the final output for the group $$j$$ that individual $$i$$ is a part of. Intuitively it looks like there are two regressions going on here - one that predicts the group-level contributions and one that predicts the final output (which is itself a function of the group-level contributions).

Another important aspect of multilevel models is that even though coefficients will be different for each group, they are all taken from the same "feature distribution". It is this aspect that mitigates the group data size and group variance problems associated with the indicator variable approach (we will see why later).

## Putting it in a Bayesian Framework

Multilevel models are most commonly discussed within a Bayesian framework. Traditional linear regression can also be thought of within a Bayesian framework by treating the model output like a normal distribution centered around $$X \beta$$ with some variance $$\sigma^2$$ i.e.

$$y \sim \mathcal{N}(X \beta, \sigma^2)$$

likewise the multilevel regression equations can be written as

$$y_i \sim \mathcal{N}(\alpha_{j[i]} + X_i \beta, \sigma^2_y), \quad \forall \thinspace i = \{1, ...,n \}$$

$$\alpha_j \sim \mathcal{N}(U_j \gamma, \sigma^2_{\alpha}), \quad \forall \thinspace j = \{1, ..., J \}$$

Placing the model in a Bayesian framework is crucial because it allows us to take into account individual and group-level variances ($$\sigma^2_y$$ and $$\sigma^2_{\alpha}$$, respectively) which we could not do with the indicator variable approach.

## Learning Parameters

Now that the problem has been placed in a Bayesian framework, the parameters can be learned via Gibbs sampling.  The goal is to learn the following parameters given the model and the data: the group-level contributions $$\alpha_1, ..., \alpha_J$$, the individual-level coefficients $$\beta$$, the group-level coefficients $$\gamma$$, the individual-level variance hyperparameter $$\sigma^2_y$$, and the group-level variance hyperparameter $$\sigma^2_{\alpha}$$.

The general idea behind the Gibbs sampling procedure is to start with a random initialization of all the parameters then update each of the parameters via that parameter's particular update equation while holding the other parameters constant.  In this way each parameter gets updated based on the most recent update of the other parameters.  If you run this long enough it should converge (how to know if it has converged is another issue and beyond the scope of this blog post).

The Gibbs sampling procedure for the multilevel model that we have been focusing on will be given without proof but the intuition should be somewhat clear.

## Gibbs Sampler Procedure

The following are the Gibbs sampling updates for the parameters of interest which are repeated until convergence.

### Step 1: Update $$\alpha$$

For each group $$j$$, the group-level contribution can be sampled from $$\mathcal{N}(\hat{\alpha}_j, V_j)$$ where

$$ \hat{\alpha}_j  = \frac{\frac{n_j}{\sigma^2_y}\bar{y}_j + \frac{1}{\sigma^2_{\alpha}}\mu_{\alpha}}{\frac{n_j}{\sigma^2_y}+ \frac{1}{\sigma^2_{\alpha}}}, \quad V_j = \frac{1}{\frac{n_j}{\sigma^2_y} + \frac{1}{\sigma^2_{\alpha}}}$$

where $$\bar{y}_j$$ is the mean $$y$$ value for group $$j$$ (think no pooling) and $$\mu_{\alpha}$$ is the mean $$y$$ value across all groups (think total pooling).  Think of this as a weighted average between the two where the weights depend on the sample size of the group ($$n_j$$) as well as the individual and group variance ($$\sigma^2_y$$ and $$\sigma^2_{\alpha}$$, respectively).  

However, we are not interested in $$y$$'s we are interested in $$\alpha$$'s so this doesn't make sense.  For this to make sense we must replace the $$y$$'s with $$y^{temp}$$'s where

$$y^{temp}_i = y_i - X_i\beta - U_{j[i]} \gamma$$ 

This should be equivalent to the group-level error $$\eta_j$$ where

$$\alpha_j = U_j \gamma + \eta_j$$

So once the $$y^{temp}$$ replacement has been made, the sample from $$\mathcal{N}(\hat{\alpha}_j, V_j)$$ will represent $$\eta_j$$.  With that sample, $$\alpha_j$$ can be recovered from the above equation.  It is important to keep in mind that we are holding all other parameters constant in this step.

### Step 2: Update $$\beta$$

For each individual $$i$$, compute

$$y^{temp}_i = y_i - \alpha_{j[i]}$$

which is the contribution of the individual-level features.  Then regress $$y^{temp}$$ on $$X$$ to get $$\hat{\beta}$$ and covariance matrix $$V_{\beta}$$ i.e.

$$\hat{\beta} = (X^TX)^{-1}Xy^{temp}$$

$$V_{\beta} = (X^TX)^{-1} \sigma^2_y$$

Then sample $$\beta \sim \mathcal{N}(\hat{\beta}, V_{\beta})$$.

### Step 3: Update $$\gamma$$

Similary, to update $$\gamma$$ we regress $$\alpha$$ on $$U$$ to get $$\hat{\gamma}$$ and $$V_{\gamma}$$ i.e.

$$\hat{\gamma} = (U^TU)^{-1}U \alpha$$

$$V_{\gamma} = (U^TU)^{-1} \sigma^2_{\alpha}$$

Then sample $$\gamma \sim \mathcal{N}(\hat{\gamma}, V_{\gamma})$$.

### Step 4: Update $$\sigma^2_y$$

Updating the individual-level variance is just taking the sample variance using the current parameters i.e.

$$\sigma^2_y = \frac{1}{n} \sum_{i=1}^n (y_i - \alpha_{j[i]} - X_i \beta)^2$$

### Step 5: Update $$\sigma^2_{\alpha}$$

And similarly for $$\sigma^2_{\alpha}$$,

$$\sigma^2_{\alpha} = \frac{1}{n} \sum_{i=1}^n (\alpha_j - U_j \gamma)^2$$

## Gibbs Sampler Intuition

The should be no intuition necessary for steps 4 and 5 of the Gibbs sampler - they are just two sample variance computations.  Updates 2 and 3 are also fairly intuitive - the coefficients come from the normal equations as with linear regression.  Another way of looking at it is taking the MLE of the coefficients given the current versions of the other parameters.  It is also important that we sample from a normal distribution around these estimates in order to take the variance into account.  Step 1, however, is not very intuitive but generally it estimates the group-level errors given the group-level and individual-level variances then it uses the errors to estimate the $$\alpha_j$$'s' (which are then used for the subsequent steps).

## References

* _Data Analysis Using Regression and Multilevel/Hierarchical Models_, Gelman & Hill.

