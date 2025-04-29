# cs285-assignment-2-policy-gradients-solved
**TO GET THIS SOLUTION VISIT:** [CS285 Assignment 2-Policy Gradients Solved](https://www.ankitcodinghub.com/product/cs285-assignment-2-policy-gradients-solved-4/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;113329&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS285 Assignment 2-Policy Gradients Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
1 Introduction

The goal of this assignment is to experiment with policy gradient and its variants, including variance reduction tricks such as implementing reward-to-go and neural network baselines. The startercode can be found at

https://github.com/berkeleydeeprlcourse/homework_fall2020/tree/master/hw2

2 Review

2.1 Policy gradient

Recall that the reinforcement learning objective is to learn a θ∗ that maximizes the objective function:

J(θ) = Eτ∼πθ(τ) [r(τ)] (1)

where each rollout τ is of length T, as follows:

T−1

πθ(τ) = p(s0,a0,…,sT−1,aT−1) = p(s0)πθ(a0|s0) Y p(st|st−1,at−1)πθ(at|st)

t=1

and

T−1

r(τ) = r(s0,a0,…,sT−1,aT−1) = X r(st,at).

t=0

The policy gradient approach is to directly take the gradient of this objective:

Z

∇θJ(θ) = ∇θ πθ(τ)r(τ)dτ (2)

Z

= πθ(τ)∇θ logπθ(τ)r(τ)dτ. (3)

= Eτ∼πθ(τ) [∇θ logπθ(τ)r(τ)] (4)

(5) In practice, the expectation over trajectories τ can be approximated from a batch of N sampled trajectories:

) (6)

. (7)

Here we see that the policy πθ is a probability distribution over the action space, conditioned on the state. In the agent-environment loop, the agent samples an action at from πθ(·|st) and the environment responds with a reward r(st,at).

2.2 Variance Reduction

2.2.1 Reward-to-go

One way to reduce the variance of the policy gradient is to exploit causality: the notion that the policy cannot affect rewards in the past. This yields the following modified objective, where the sum of rewards here does not include the rewards achieved prior to the time step at which the policy is being queried. This sum of rewards is a sample estimate of the Q function, and is referred to as the “reward-to-go.”

. (8)

2.2.2 Discounting

The first way applies the discount on the rewards from full trajectory:

!

(9)

and the second way applies the discount on the “reward-to-go:”

. (10)

.

2.2.3 Baseline

Another variance reduction method is to subtract a baseline (that is a constant with respect to τ) from the sum of rewards:

∇θJ(θ) = ∇θEτ∼πθ(τ) [r(τ) − b]. (11)

This leaves the policy gradient unbiased because

∇θEτ∼πθ(τ) [b] = Eτ∼πθ(τ) [∇θ logπθ(τ) · b] = 0.

In this assignment, we will implement a value function Vφπ which acts as a state-dependent baseline. This value function will be trained to approximate the sum of future rewards starting from a particular state:

T−1

Vφπ(st) ≈ XEπθ [r(st0,at0)|st],

t0=t

so the approximate policy gradient now looks like this: (12)

. (13)

3 Overview of Implementation

3.1 Files

To implement policy gradients, we will be building up the code that we started in homework 1. All files needed to run your code are in the hw2 folder, but there will be some blanks you will fill with your solutions from homework 1. These locations are marked with # TODO: get this from hw1 and are found in the following files:

• infrastructure/rl trainer.py

• infrastructure/utils.py

• policies/MLP policy.py

After bringing in the required components from the previous homework, you can begin work on the new policy gradient code. These placeholders are marked with TODO, located in the following files:

• agents/pg agent.py

• policies/MLP policy.py

The script to run the experiments is found in scripts/run hw2.py (for the local option) or scripts/run hw2.ipynb (for the Colab option).

3.2 Overview

As in the previous homework, the main training loop is implemented in infrastructure/rl trainer.py.

The policy gradient algorithm uses the following 3 steps:

1. Sample trajectories by generating rollouts under your current policy.

2. Estimate returns and compute advantages. This is executed in the train function of pg agent.py

3. Train/Update parameters. The computational graph for the policy and the baseline, as well as the update functions, are implemented in policies/MLP policy.py.

4 Implementing Policy Gradients

You will be implementing two different return estimators within pg agent.py. The first (“Case 1” within calculate q vals) uses the discounted cumulative return of the full trajectory and corresponds to the “vanilla” form of the policy gradient (Equation 9):

T−1

0

r(τi) = X γt r(sit0,ait0).

t0=0

The second (“Case 2”) uses the “reward-to-go” formulation from Equation 10: (14)

T−1

X t0−t

r(τi) = γ r(sit0,ait0). (15)

t0=t

Note that these differ only by the starting point of the summation.

5 Small-Scale Experiments

After you have implemented all non-baseline code from Section 4, you will run two small-scale experiments to get a feel for how different settings impact the performance of policy gradient methods.

Experiment 1 (CartPole). Run multiple experiments with the PG algorithm on the discrete CartPole-v0 environment, using the following commands:

python cs285/scripts/run_hw2.py –env_name CartPole-v0 -n 100 -b 1000

-dsa –exp_name q1_sb_no_rtg_dsa

python cs285/scripts/run_hw2.py –env_name CartPole-v0 -n 100 -b 1000

-rtg -dsa –exp_name q1_sb_rtg_dsa

python cs285/scripts/run_hw2.py –env_name CartPole-v0 -n 100 -b 1000

-rtg –exp_name q1_sb_rtg_na

python cs285/scripts/run_hw2.py –env_name CartPole-v0 -n 100 -b 5000

-dsa –exp_name q1_lb_no_rtg_dsa

python cs285/scripts/run_hw2.py –env_name CartPole-v0 -n 100 -b 5000

-rtg -dsa –exp_name q1_lb_rtg_dsa

python cs285/scripts/run_hw2.py –env_name CartPole-v0 -n 100 -b 5000

-rtg –exp_name q1_lb_rtg_na

What’s happening here:

• -n : Number of iterations.

• -b : Batch size (number of state-action pairs sampled while acting according to the current policy at each iteration).

• -dsa : Flag: if present, sets standardize_advantages to False. Otherwise, by default, standardizes advantages to have a mean of zero and standard deviation of one.

• -rtg : Flag: if present, sets reward_to_go=True. Otherwise, reward_to_go=False by default.

• –exp_name : Name for experiment, which goes into the name for the data logging directory.

Various other command line arguments will allow you to set batch size, learning rate, network architecture, and more. You can change these as well, but keep them fixed between the 6 experiments mentioned above.

Deliverables for report:

• Create two graphs:

– In the first graph, compare the learning curves (average return at each iteration) for the experiments prefixed with q1_sb_. (The small batch experiments.)

– In the second graph, compare the learning curves for the experiments prefixed with q1_lb_. (The large batch experiments.)

• Answer the following questions briefly:

– Which value estimator has better performance without advantage-standardization: the trajectorycentric one, or the one using reward-to-go?

– Did advantage standardization help?

– Did the batch size make an impact?

• Provide the exact command line configurations (or #@params settings in Colab) you used to run your experiments, including any parameters changed from their defaults.

What to Expect:

• The best configuration of CartPole in both the large and small batch cases should converge to a maximum score of 200.

Experiment 2 (InvertedPendulum). Run experiments on the InvertedPendulum-v2 continuous control environment as follows:

python cs285/scripts/run_hw2.py –env_name InvertedPendulum-v2

–ep_len 1000 –discount 0.9 -n 100 -l 2 -s 64 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg

–exp_name q2_b&lt;b*&gt;_r&lt;r*&gt;

Deliverables:

• Provide the exact command line configurations you used to run your experiments.

6 Implementing Neural Network Baselines

You will now implement a value function as a state-dependent neural network baseline. This will require filling in the remaining TODO sections skipped in Section 4. In particular:

• This neural network will be trained in the update method of MLPPolicyPG along with the policy gradient update.

• In pg agent.py:estimate advantage, the predictions of this network will be subtracted from the reward-to-go to yield an estimate of the advantage. This implements

7 More Complex Experiments

Note: The following tasks take quite a bit of time to train. Please start early! For all remaining experiments, use the reward-to-go estimator.

Experiment 3 (LunarLander). You will now use your policy gradient implementation to learn a controller for LunarLanderContinuous-v2. The purpose of this problem is to test and help you debug your baseline implementation from Section 6.

Run the following command:

python cs285/scripts/run_hw2.py

–env_name LunarLanderContinuous-v2 –ep_len 1000

–discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005

–reward_to_go –nn_baseline –exp_name q3_b40000_r0.005

Deliverables:

• Plot a learning curve for the above command. You should expect to achieve an average return of around 180 by the end of training.

Experiment 4 (HalfCheetah). You will be using your policy gradient implementation to learn a controller for the HalfCheetah-v2 benchmark environment with an episode length of 150. This is shorter than the default episode length (1000), which speeds up training significantly. Search over batch sizes b ∈ [10000,30000,50000] and learning rates r ∈ [0.005,0.01,0.02] to replace &lt;b&gt; and &lt;r&gt; below.

python cs285/scripts/run_hw2.py –env_name HalfCheetah-v2 –ep_len 150

–discount 0.95 -n 100 -l 2 -s 32 -b &lt;b&gt; -lr &lt;r&gt; -rtg –nn_baseline

–exp_name q4_search_b&lt;b&gt;_lr&lt;r&gt;_rtg_nnbaseline

Deliverables:

• Provide a single plot with the learning curves for the HalfCheetah experiments that you tried. Describe in words how the batch size and learning rate affected task performance.

Once you’ve found optimal values b* and r*, use them to run the following commands (replace the terms in angle brackets):

python cs285/scripts/run_hw2.py –env_name HalfCheetah-v2 –ep_len 150

–discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt;

–exp_name q4_b&lt;b*&gt;_r&lt;r*&gt;

python cs285/scripts/run_hw2.py –env_name HalfCheetah-v2 –ep_len 150

–discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg

–exp_name q4_b&lt;b*&gt;_r&lt;r*&gt;_rtg

python cs285/scripts/run_hw2.py –env_name HalfCheetah-v2 –ep_len 150

–discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; –nn_baseline

–exp_name q4_b&lt;b*&gt;_r&lt;r*&gt;_nnbaseline

python cs285/scripts/run_hw2.py –env_name HalfCheetah-v2 –ep_len 150

–discount 0.95 -n 100 -l 2 -s 32 -b &lt;b*&gt; -lr &lt;r*&gt; -rtg –nn_baseline

–exp_name q4_b&lt;b*&gt;_r&lt;r*&gt;_rtg_nnbaseline

Deliverables: Provide a single plot with the learning curves for these four runs. The run with both rewardto-go and the baseline should achieve an average score close to 200.

8 Bonus!

Choose any (or all) of the following:

• A serious bottleneck in the learning, for more complex environments, is the sample collection time. In infrastructure/rl trainer.py, we only collect trajectories in a single thread, but this process can be fully parallelized across threads to get a useful speedup. Implement the parallelization and report on the difference in training time.

• In PG, we collect a batch of data, estimate a single gradient, and then discard the data and move on. Can we potentially accelerate PG by taking multiple gradient descent steps with the same batch of data? Explore this option and report on your results. Set up a fair comparison between single-step PG and multi-step PG on at least one MuJoCo gym environment.

9 Submission

1 https://arxiv.org/abs/1506.02438

9.1 Submitting the PDF

Your report should be a document containing

(a) All graphs and answers to short explanation questions requested for Experiments 1-4.

(b) All command-line expressions you used to run your experiments.

(c) (Optionally) Your bonus results (command-line expressions, graphs, and a few sentences that comment on your findings).

9.2 Submitting the code and experiment runs

In order to turn in your code and experiment logs, create a folder that contains the following:

• The cs285 folder with all the .py files, with the same names and directory structure as the original homework repository (excluding the cs285/data folder). Also include any special instructions we need to run in order to produce each of your figures or tables in the form of a README file.

agents bc agent.py

…

policies

…

9.3 Turning it in
