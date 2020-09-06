---
layout: post
title: "Google JAX Overview"
date: 2020-08-15
comments: false
categories: 
---

According to it's [README](https://github.com/google/jax) JAX is "_Autograd and XLA, brought together for high-performance machine learning research_" from Google.  Autograd is a reference to an [automatic differentiation library](https://github.com/hips/autograd) which was originally maintained by the Harvard Intelligent Probabilistic Systems Group (HIPS).  XLA is a reference to Tensorflow's [XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla) compiler.  JAX also says "_At its core, JAX is an extensible system for transforming numerical functions. Here are four of primary interest: grad, jit, vmap, and pmap_".  At this point, these four functions make up the bulk of JAX so this blog post will go through each of them and doing so should provide a good overview of JAX in general.


## grad (Autograd)

```grad``` is JAX's automatic differentiation function.  It accepts ([nearly](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)) any function and returns a function that is it's derivative.  For example, say you want the derivative of the $$tanh$$ function, you can do this with ```grad``` in one line as

```python
import jax
  
grad_tanh = jax.grad(jax.numpy.tanh)
```
Autograd works in a similar way to the method descibed in my [old computational graph post](http://alexminnaar.com/2018/07/14/simple-computational-graph-engine.html).  The basic steps are

<ol style="margin-left: 25px">
  <li style="font-size:18px"><b>Tracing the Function:</b>  This means parsing the the input function into it's individual ops and building a computational graph where each node is an op.</li>
  <li style="font-size:18px"><b>Topological Sort:</b>  This means putting the ops in order based on their dependencies in the computational graph.</li>
  <li style="font-size:18px"><b>Implement VJPs (vector-Jacobian products) for each Op:</b>  One way to effectively compute an op's derivative in the context of computational graphs is to use it's VJP.  Check out <a href="https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf">these slides</a> for a more in-depth explanation.</li>
  <li style="font-size:18px"><b>Backpropagate:</b>  Using the computational graph and the VJP's for each node/op you can backpropagate through the graph to obtain a derivative value at any point.</li>
</ol>

Since the above process involves going backwards through the computational graph it is called _reverse-mode_ differentiation.  

## jit (XLA)

The ```jit``` function exposes the XLA compiler.  For example, say you want a function "jited" - all you need to do is pass the function through the ```jit``` higher order function and the output is the "jited" version of that function i.e.

```python
import jax

 
def selu(x, alpha=1.67, lmbda=1.05):
   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)

fast_selu = jax.jit(selu)
```

But what does this actually do?  In terms of XLA, the jit compiler uses GPU kernel fusion to apply significant performance improvements to the input function.  Without XLA, typically a function would be parsed into it's constituent ops where each op has it's own kernel that needs to be launched.  The problem with this is that each kernel launch requires it's own memory operations which can be particularly costly in memory-bound computations.  XLA side-steps this problem by fusing the ops into one kernel requiring the costly memory opertations to only happen once.

Of course XLA is it's [own project](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla) within TensorFlow so you will not find it's implementation in the JAX project.  Behind the scenes JAX interacts with XLA via a downloaded XLA shared object.


## vmap

```vmap``` is short for _vectorizing map_ and it does just that.  A map is a particular kind of higher-order function which applies some function argument to each element of a sequential data structure then returns the result ([more about maps](https://en.wikipedia.org/wiki/Map_(higher-order_function))).  ```vmap``` does the same thing except it vectorizes the process for better performance.

As a concrete example, consider the example from JAX's README.  Let's start with a ```predict``` function which does a forward pass through a standard MLP neural network taking a vector input.

```python
def predict(params, input_vec):
  assert input_vec.ndim == 1
  for W, b in params:
    output_vec = jnp.dot(W, input_vec) + b
    input_vec = jnp.tanh(output_vec)
  return output_vec
```

The important thing to note about the above function is that ```input_vec``` is a vector so if you wish to predict a batch of input vectors then you would need to call the ```predict``` function on each individual vector sequentially i.e. a map function.  ```vmap``` can vectorize this map function in the following way.

```python
from jax import vmap

vectorized_predict = vmap(predict, in_axis=(None, 0))
```
The ```in_axis``` argument is a tuple that indicates which axis of the inputs to map over.  We don't want to map over the ```params``` argument so that is given a ```None``` value.  We do want to map over ```input_vec``` though.  Assume the vectorized input is a sequence of input vectors i.e. a matrix.  We would want to vectorize over the rows of that matrix i.e. the ```0``` dimension.  Now we can call the vectorized ```predict``` function on an entire batch of of inputs.

```python
fast_predictions = vectorized_predict(params, input_batch)
```


## pmap


```pmap``` is a higher-order map function that executes functions across multiple GPUs in parallel.  The semantics of ```pmap``` are relatively simple.  ```pmap```'s arugment is the function that is to be parallelized.  What is returned is a function that executes across the devices and takes an argument for each devices.  For example, from the README, we create a parallelized function that creates random matrices across devices based on some key argument.

```python
from jax import random, pmap

parallel_matrix_create = pmap(lambda key: random.normal(key, (5000, 6000)))
```

Now we create inputs corresponding to each device (let's say there are 8) and apply them to the parallelized function.

```python
keys = random.split(random.PRNGKey(0), 8)
mats = parrallel_matrix_create(keys)
```

That's it - thanks for reading.

## References

* [JAX](https://github.com/google/jax)
* [XLA](https://www.tensorflow.org/xla)
* [Map functions](https://en.wikipedia.org/wiki/Map_(higher-order_function))
* [Automatic Differentiation](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf)
