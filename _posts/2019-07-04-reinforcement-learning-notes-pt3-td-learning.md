---
layout: post
title: "Reinforcement Learning Notes Part 3: Temporal Difference Learning"
date: 2019-07-02
comments: false
categories: 
---

Temporal difference learning shares many of the benefits of both [dynamic programming methods](http://alexminnaar.com/2019/06/29/reinforcement-learning-pt1-dynamic-programming.html) and [Monte Carlo methods](http://alexminnaar.com/2019/06/30/reinforcement-learning-notes-pt2-monte-carlo-methods.html) without many their disadvantages.  Like dynamic programming methods, policy evaluation can be updated at each time step but unlike dynamic programming you do not need a model of the environment.  Like Monte Carlo methods, you do not need a model of the environemt but unlike Monte Carlo methods you do not need to wait til the end of an episode to make a policy evaluation update.  All three of these methods use the same policy iteration strategy which iterates between policy evaluation (different for each method) and policy improvement (in a greedy fashion for each method).

The value-function update formula for temporal difference learning is

$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

where $$r_{t+1}$$ is the observed reward of the next time step and $$V(s_{t+1})$$ is the estimate of the value of the next time step.  Since TD learning uses an existing estimate $$V(s_{t+1})$$ it is known as a _bootstrapping_ method (like DP methods). This TD update algorithm is guaranteed to converge to the value function of a given policy $$V^\pi(s)$$ as long as the step size parameter $$\alpha$$ is sufficiently small.
<h2><font size="5">Sarsa (On-Policy TD Learning)</font></h2>

Recall that for on-policy methods, the value function is learned by following the associated policy (with some exploration strategy like $$\epsilon$$-greedy).  Sarsa is concerned with learning an action-value function $$Q^\pi(s,a)$$ for a given policy $$\pi$$ rather than just the value function, however essentially the same update is used.

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]$$

If you take a close look at the update formula you will see that it involves the current state $$s_t$$, the current action $$a_t$$, the reward observed from that action $$r_{t+1}$$, the next state $$s_{t+1}$$, and the next action $$a_{t+1}$$.  If you combine these symbols together you get $$s_ta_tr_{t+1}s_{t+1}a_{t+1} \rightarrow Sarsa$$ which is how you get the name.



<h2><font size="5">Q-Learning (Off-Policy TD Learning)</font></h2>

Recall the for off-policy methods, the value function for a given policy is learned by following a different policy.  The Q-learning update function is

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_{t+1} + \gamma \max_aQ(s_{t+1},a) - Q(s_t,a_t)]$$

So this update formula will end up estimating $$Q^\pi(s,a)$$ no matter what policy is being observed.  The only requirement is that the policy being observed updates all state-action pairs.

Thank you for reading.
## References
* [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)