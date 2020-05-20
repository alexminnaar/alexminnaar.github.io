---
layout: post
title: "Deep Learning GPU Performance Analysis: Mixed Precision Training"
date: 2020-05-02
comments: false
categories: 
---

There are several meaningful benefits to training deep neural networks using a precision format that is lower than 32-bit floating point.  Lower precision requires less memory which enables us to train larger networks and/or training with larger minibatches.  Furthermore lower precision requires less memory bandwidth which means training is faster.  And thirdly lower precision allows for faster math operations.  Mixed precision training exploits these benefits and is possible if you are working with a Volta Nvidia GPU or newer.

## What does "Lower Precision" Mean?

_Single precision_ refers to 32-bit floating point (FP32) format.  Lower precision usually means _half precision_ which refers to 16-bit floating point (FP16) format.  As the name suggests FP32 format uses 32 bits of memory and FP16 format uses 16 bits of memory.  As a consequence, the number of bytes accessed is also halfed.  

However, going from FP32 to FP16 also reduces the precision which, in terms of deep learning, comes into play when you are dealing with small activation gradient values.  For example, a gradient value can be small enough such that it is representable by a FP32 number but not a FP16 number (in which case it becomes zero) which results in significant problems with training accuracy using FP16.  However this loss in accuracy can be mitigated by scaling the gradient values such that they are shifted into FP16's representable range.


## Using Scaling to Prevent Loss of Accuracy

As stated previously, if a value is too small to be represented in FP16 (and hence problematically represented as zero) the value can be recovered by scaling it by a factor such that it can be represented in FP16.  The small values that are of concern in deep learning applications are the activation gradients.  In practice, the scaling is typically done between the forward propagation and backpropagation stages.  If the input of backpropagation is scaled by a given factor then all of the resulting activation gradients will also be scaled by that factor hence only the input needs to be scaled and the rest of the backpropagation procedure can remain unchanged.  Once backpropagation has finished, the weight gradients can be unscaled and passed to the optimizer to update the weights.  In general, the procedure is

<ol style="margin-left: 25px">
  <li style="font-size:18px">Perform forward propagation.</li>
  <li style="font-size:18px">Scale the output by a factor X.</li>
  <li style="font-size:18px">Perform backpropagation.</li>
  <li style="font-size:18px">Unscale the weight gradients by a factor 1/X.</li>
  <li style="font-size:18px">Optimizer updates weights.</li>
</ol>

The scaling factor can be constant or dynamic.  One possible procedure for choosing a constant scaling factor is to determine the maximum gradient value for your particular network and dataset and choose a scaling factor such that the scaled maximum gradient value is below the maximum representable FP16 value (i.e. 65,504).  In this way you will ensure that you will not run into the opposite problem of the scaled gradients being so large that they overflow.

## What is Mixed Precision Training?

Finally, the term _mixed precision training_ refers to a deep neural network training procedure that uses half precision whenever possible and full precision when it is not (for example, reduction operations typically require full precision).  Mixed precision training can result in an up to 3x speedup for arithmetically intense model architectures. In addition, even memory-bound architectures can see a speedup due to the memory bandwidth advantages of using half precision.


## How to Enable Mixed Precision Training in the Deep Learning Frameworks

Mixed precision training is possible in both the PyTorch and TensorFlow frameworks as long as you are working with a Volta Nvidia GPU or newer.

### PyTorch

In the PyTorch framework, mixed precision training is available through the [AMP (automatic mixed precision) API](https://pytorch.org/docs/stable/amp.html).  With this API you create your own gradient scaler object like

```python
scaler = GradScaler()
```

Then, as descibed earlier, the scaler is applied to the output of the forward pass and the resulting scaled value is used for backpropagation.

```python
scaler.scale(loss).backward()
```
The scaler object is then applied to the optimizer such that the scaled weights can then be unscaled in order to update the weights.

```python
scaler.step(optimizer)
```
Finally the scaler is updated such that the scaling factor can be adjusted dynamically.

```python
scaler.update()
```

### TensorFlow

In the TensorFlow framework, mixed precision training can be achieved by wrapping the optimizer object in a `tf.train.experimental.enable_mixed_precision_graph_rewrite()` object like

```python
opt = tf.train.AdamOptimizer()
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
```

## Resources

* [Nvidia Deep Learning Performance Documentation](https://docs.nvidia.com/deeplearning/performance/index.html)
* [PyTorch AMP API](https://pytorch.org/docs/stable/amp.html)
* [TensorFlow Mixed Precision](https://www.tensorflow.org/guide/keras/mixed_precision) 