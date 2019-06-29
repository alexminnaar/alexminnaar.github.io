---
layout: post
title: "Reinforcement Learning Notes Part 1: Dynamic Programming"
date: 2019-06-29
comments: false
categories: 
---

This series of blog posts is intended to be a collection of short, concise, cheat-sheet-like notes on different topics relating to reinforcement learning.  This first one will cover dynamic programming methods applied to reinforcement learning.

<h2><font size="5">Markov Decision Processes (MDPs)</font></h2>
 The methods outlined in this blog post all assume a Markov Decisin Process.  An MDP involves complete knowledge of the environment as defined by a state transition probability matrix where 

 $$P^a_{s,s'} = P(s_{t+1}=s' | s_t = s, a_t = a)$$ 

 represents the probability of moving from state $$s$$ to state $$s'$$ after taking action $$a$$ (note: sometimes actions can be deterministic i.e. the probability of moving from one state to another after a given action can equal one).  An MDP also assumes complete knowledge of rewards as defined by the reward function

 $$R^a_{s,s'} = E[r_{t+1}|a_t=a,s_t=s,s_{t+1}=s']$$

 which is the expected reward associated with moving from state $$s$$ to $$s'$$ after taking action $$a$$.

<h2><font size="5">The Bellman Equation</font></h2>

The value assocated with a given state and a given action taken from that state can be thought of as the expected cumulative discounted reward for that state and action.  This can be expressed as the Bellman equation

$$V^a(s) = E[r_{t+1} + \gamma V(s_{t+1})|s_t=s,a_t=a]$$

where $$\gamma$$ is the discount factor.  This is also called the value function.  Since we have access to the transition probabilities and the reward function we can write this expectation explicitly as

$$V^a(s) = \sum_{s'} P^a_{s,s'}[R^a_{s,s'}+\gamma V(s')]$$


<h2><font size="5">Finding Optimal Policies</font></h2>

The goal here is to find an optimal policy where a policy is a function $$\pi(s,a)$$ that determines the action to take at a given state that will maximize the expected cumulative discounted reward ($$V^a(s)$$).  Note that policies can be stochastic where different actions can be taken with different probabilities from the same state.  In this case finding the optimal policy is equivalent to finding the optimal value function - the optimal action is the one that moves you to the state with the best value.  The optimal value function $$V^*(s)$$ can be expressed as

$$V^*(s) = \max_a \sum_{s'} P^a_{s,s'}[R^a_{s,s'}+\gamma V^*(s')]$$

There are analytical ways to solve this problem however typically iterative solutions are more convenient.  Two common iterative methods are policy iteration and value iteration.

<h2><font size="5">Policy Iteration</font></h2>

Policy iteration consists of two steps which are repeated until convergence.
<ul style="margin-left: 20px">
  <li style="font-size:19px"><b>Policy Evaluation: </b> Given a policy, evaluate the value function associated with it.</li>
  <li style="font-size:19px"><b>Policy Improvement: </b> Given the value function that you just evaluated, improve your policy in a greedy fashion.</li>
</ul>

This procedure is initialized with a random value function and eventually it is guaranteed to converge to the optimal policy.  Convergence occurs when no more greedy policy improvements can be made.  This procedure might remind you of the [EM algorithm](http://alexminnaar.com/2017/05/22/EM-Algorithm-and-Mixture-of-Gaussians-Clustering.html).

However it must be noted that the policy evaluation step is not very straightforward because it is itself an iterative procedure.  Given a policy $$\pi(s,a)$$ we again use the Bellman equation to iteratively compute the value function associated with it

$$V_{k+1}(s) = \sum_a \pi(s,a) \sum_{s'} P^a_{s,s'}[R^a_{s,s'}+\gamma V_{k}(s')]$$

Again, the value function is initialized randomly and eventually converges to the value function for the given policy.

The policy improvement step is more straightforward.  With the updated value function that was just computed, the policy is improved in a greedy fashion.  This simply means that the action is taken which leads to a state with the largest value.

<h2><font size="5">Value Iteration</font></h2>
Value iteration is very similar to policy iteration except that it does not wait for full convergence during the policy evaluation step.  During the policy evaluation step only one update sweep is performed before the policy is improved again. It is essentially just turning the Bellman equation into an update rule.

$$V_{k+1}(s) = \max_a \sum_{s'} P^a_{s,s'}[R^a_{s,s'}+\gamma V_{k}(s')]$$

As you can see, we are not explicitly keeping track of the policy here, but we are keeping track of the value function.  When the value function converges to it's optimal value the optimal policy can be obtained trivially.  Value iteration has the same convergence guarantees as policy iteration.

Thank you for reading.

## References
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)