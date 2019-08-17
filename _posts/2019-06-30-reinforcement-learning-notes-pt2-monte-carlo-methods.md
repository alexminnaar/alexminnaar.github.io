---
layout: post
title: "Reinforcement Learning Notes Part 2: Monte Carlo Methods"
date: 2019-06-30
comments: false
categories: 
---

In the [last reinforcement learning blog post](http://alexminnaar.com/2019/06/29/reinforcement-learning-pt1-dynamic-programming.html) we covered dynamic programming methods.  In this blog post we will cover Monte Carlo (MC) methods.  The biggest difference between these two methods is that dynamic programming methods assume a complete knowledge of the environement (via a MDP), but Monte Carlo methods do not.  Instead, with Monte Carlo methods, knowledge of the environment is learned through experience.  Another significant difference is that Monte Carlo methods can only learn from _episodic_ tasks i.e. ones that start and terminate.

However, dynamic programming methods and Monte Carlo methods handle the control problem in generally the same way.  In both methods it is an iterative procedure that alternates between policy evaluation and policy improvement (i.e. policy iteration).  In fact the policy improvement step is the same in that it is performed in a greedy fashion.  The main difference is in the policy evaluation step.

<h2><font size="5">Monte Carlo Policy Evaluation</font></h2>

Recall that policy evalution is the problem of computing the value function for a given policy i.e. $$V^{\pi}(s)$$.  And the value function evaluated at a given state is defined as the _expected cumulative future discounted reward_ starting from that state.  A very simple way of doing this is to just average the rewards that have been seen for the episodes in which that state was visited.  This is essentially what Monte Carlo policy evaluation does. 

There are two general methods for Monte Carlo policy evaluation - first-visit MC and every-visit MC.  
<ul style="margin-left: 20px">
  <li style="font-size:19px"><b>First-Visit MC: </b> Uses the average of the returns following all visits to the state in a set of episodes.</li>
  <li style="font-size:19px"><b>Every-Visit MC: </b> Uses averaged returns only after the first vist to the state in a set of episodes.</li>
</ul>

Both methods converge to the $$V^{\pi}(s)$$ as the number of visits go to infinity.

<h2><font size="5">The Role of Exploration</font></h2>
The problem with Monte Carlo policy evaluation is that you can only determine the value of a given state if your policy visits it, otherwise there will never be a reward associated with it.  So if you begin the process with a deterministic policy that never visits certain states then the values associated with those states will never be known.  However, it is also possible that the optimal policy would visit those states.  So we need a way to insure that all states will be visited.  There are two strategies to insure this - they are called on-policy MC and off-policy MC.

<h2><font size="5">On-Policy Monte Carlo</font></h2>

On policy methods are the simpler of the two.  With on-policy methods you are still following the policy during policy evaluation however you are adding some random exploration as well.  For example, $$\epsilon$$-greedy is a common on-policy method where at every step you choose the action determined by your policy with probability $$1-\epsilon$$ and you choose a random action with probability $$\epsilon$$.  With this added randomness you can insure that exploration occurs.

<h2><font size="5">Off-Policy Monte Carlo</font></h2>
The idea for off-policy methods is to evaluate your policy while following a different policy.  This has obvious practical advantages for cases where you can observe other policies but are unable to try your own.  Say you want to evaluate policy $$\pi$$ but can only observe a different policy $$\pi'$$.  As long as every action taken under $$\pi$$ is also taken under $$\pi'$$ you can still evaluate $$V^{\pi}$$.  Let's see how this is possible.

Let $$p_i(s)$$ represent the probability of a complete episode occurring under policy $$\pi$$ and similarly let $$p_i'(s)$$ represent the probability of a complete episode occurring under policy $$\pi'$$.  Also, let $$R_i(s)$$ represent the reward observed after the first occurrence of $$s$$.  Then the value at state $$s$$ under policy $$\pi$$ can be estimated as

$$V(s) = \frac{\sum_{i=1}^n \frac{p_i(s)}{p_i'(s)}R_i(s)}{\sum_{i=1}^n \frac{p_i(s)}{p_i'(s)}}$$

Where $$n$$ is the number of episodes.  Intuitively, the way to look at this is that even though the reward $$R_i(s)$$ is associated with the other policy $$\pi'$$, we are weighting it by the relative probability of it occurring $$\frac{p_i(s)}{p_i'(s)}$$ i.e. if the reward is more likely to occur under $$\pi$$ then it is up-weighted and if it is less likely to occur under $$\pi$$ then it is down-weighted.  This is very similar to [importance sampling](https://en.wikipedia.org/wiki/Importance_sampling).  Now the question is how to determine the probabilities $$p_i(s)$$ and $$p_i'(s)$$.  Luckily we don't need to compute them individually rather we only need their ratio.  This can be computed by

$$\frac{p_i(s_t)}{p_i'(s_t)} = \prod^{T_i(s)-1}_{k=t} \frac{\pi(s_k,a_k)}{\pi'(s_k,a_k)}$$ 

where $$T_i(s)$$ is the time of termination of the $$i^{th}$$ episode involving $$s$$.  The advantage here is that to compute the desired ratio all we need are the two policies and not the environment dynamics.

Thank you for reading.

## References
* [Reinforcement Learning Notes Part 1: Dynamic Programming](http://alexminnaar.com/2019/06/29/reinforcement-learning-pt1-dynamic-programming.html)
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)