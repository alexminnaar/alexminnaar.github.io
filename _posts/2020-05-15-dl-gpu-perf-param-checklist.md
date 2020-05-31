---
layout: post
title: "Deep Learning GPU Performance Analysis: Optimal Parameter Checklist"
date: 2020-05-15
comments: false
categories: 
---

Typically the parameters used for deep learning models are chosen to optimize some objective function that measures predictive performance, however it important to understand that the chosen parameters can also have a significant affect on the time it takes to train that model.  Ideally you would want to choose parameters that both optimize predictive performance as well as minimize the training time. [NVIDIA's deep learning performance guide](https://docs.nvidia.com/deeplearning/performance/index.html) reveals some simple and in some cases unintuitive tips and tricks for choosing parameters that can result in significant training speedups without negatively affecting predictive performance.

## Optimal Parameters for Matrix Multiplication

All the the proceeding layer types use matrix multiplication in some way so we should first make ourselves aware of the optimal parameters for GEMMs (General Matrix Multiplications).  Also, since GEMMs are the building blocks of the proceeding layers you will notice that there is a lot of overlap in optimal parameter rules.

Consider the following simple matrix multiplicaton

$$C = AB$$

where $$A$$ is an $$M \times K$$ matrix, $$B$$ is a $$K \times N$$ matrix, and consequently $$C$$ is a $$M \times N$$ matrix. $$M$$, $$K$$, and $$N$$ should be multiples of 8 in order to utilize Tensor Cores (assuming you are using a device with Tensor Cores).  This could account for a speedup of up to 6x.  If the matrix dimensions do not naturally lend themselves to multiples of 8 then it would be beneficial to pad the matrices.

Also, in general, the larger the matrix sizes the better - it leads to better GPU utilization (through reduced quantization effects) and lower bandwidth usage.  Often a matrix multiply using matrices of twice the size can take less than twice the time.

## Optimal Parameters for Fully-Connected Layers 

Like in the previous section, the batch size, the number of inputs, and the number of outputs of the fully connected layer should be divisible by 8 in order to utilize Tensor Cores.  Again, the input and output sizes can be padded if they are not naturally a multiple of 8.  As you would expect, this is because these values become dimensions for GEMMs under the hood.

Also in the case where a fully-connected layer is small enough such that it is memory-bound, increasing the batch size (while making sure it is still a multiple of 8) can increase the arithmetic intensity.  Conversely, if the batch size is too small then the fully-connected layer will always be memory-bound no matter the size of the inputs and outputs.  

## Optimal Parameters for Convolutional Layers 

For convolutional layers, the sizes of the input and output channels should be divisible by 8, again to enable Tensor Cores.  Alternatively, since it is common for input channels in the first layer to be of size 3, a padded input of size 4 with a stride of 2 can also enable Tensor Cores.

Additionally, parameters like the size of the input and output channels as well as the batch size should be divisible by 256 in order to reduce quantization effects (as well as enabling Tensor Cores).

Also, in general, larger parameter sizes (while obeying the divisibility rules) tend to improve parallelization since it increases the size of the underlying GEMM.


### Resources
* [Nvidia Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)