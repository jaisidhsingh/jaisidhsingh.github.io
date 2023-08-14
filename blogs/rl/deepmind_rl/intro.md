# Lesson 1: Introduction to RL concepts

************General definition of AI: the ability of a machine to learn to make decisions************

************Reinforcement learning: learning to make decisions by understanding incentives and penalties given for some interaction with the world.************

# Interaction Loop

- This describes the stages of interaction of an agent with its environment.
- At each timestep $t$ the agent:
    - Receives an observation $O_t$ and a reward $R_t$
    - Infers a state $s_t$ (the agent state) from $O_t$
    - It then executes an action $a_t$
- Then the environment:
    - Receives action $a_t$
    - Emits another observation $O_{t+1}$ and $R_{t+1}$ for the action taken i.e. $a_t$

# Agent State

- Every considered attribute that the agent takes from one timestep to another is called the agent state. The agent state is a subset of the observations, i.e., $s_t \subseteq O_t$.
- It is denoted by $s$ for a general state, and $s_t$ for the agent state at timestep $t$.
- In general, when we refer to a “state” it is implied that we refer to the agent state.

# Environment State

- Firstly, we define the environment of the agent as the region (or entity) or “world” that it interacts with. For example, a robot is an agent and the real world of Earth is the environment.
- The environment state is the internet state of the environment. In the above example, the environment state is the physical state of the entire world, its atoms and their orientation, everything.
- In this learning, and in general, we do not talk much about the environment state in learning to make decisions.

# Reward

- The reward at a timestep, or a reward signal over the entire series of interactions, provides the incentive or penalty to an agent to learn to perform desirable actions.
- The reward at a general timestep $t$ is denoted by $R_t$, which is a scalar quantity.
- It is also beneficial to define the ******return******, i.e. the cumulative rewards that an agent will get over the course of its action from timestep $t$ to $\infin$. This is denoted by $G_t$ formally defined as:
    
    $$
    G_t = R_{t+1} + R_{t+2} + R_{t+3} + ...\\ G_t = R_{t+1} + G_{t+1} \text{ (recurrent form) }
    $$
    
- Reinforcement learning is based on the *********reward hypothesis*********: 
*****************************************Any goal can be formalized as maximizing a cumulative reward.*****************************************

# Policy

- A mapping from state to actions
- Deterministic policy: $\pi(s) = a$, i.e., directly gives us an action $a$ by considering state $s$ as input
- Stochastic policy: $\pi(a|s) = p(a|s)$ , i.e., gives us the probability that action $a$ should be chosen given that we receive state $s$ as input.
    - Typically considered as a probability distribution over actions.
    - One may think of this policy exemplified by a simple linear neural network $\pi_{nn}(\cdot)$ given by:
        
        $$
        \pi_{nn}(s) = softmax(W_{s \rightarrow a}s)\\ p(a_i|s) = \pi_{nn}(s)_i
        $$
        
        where $W_{s \rightarrow a} \in \mathbb{R}^{s \times a}$ and $\pi_{nn}(s) \in \mathbb{R}^{a}$. 
        
- We aim to learn optimal policies or perform policy optimization to learn to make the best decisions.

# Value Function

- We call the expected cumulative reward from a state $s$ the *****value*****, given by:

$$
V(s) = \mathbb{E}[G_t|s_t = s]\\ = \mathbb{E}[R_{t+1} + R_{t+2} + ... | s_t = s]\\ = \mathbb{E}[R_{t+1} + V(s_{t+1})| s_t = s] \text{ (recurrent form) }
$$

- Value denotes the desirability of states. Goal is to *maximize value*, by picking suitable actions.
- Rewards and values define the utility of actions, which in turn define action preference in a certain state. Recall game theory which says that an action $A$ is preferred over an action $B$ if and only if $U(A) \geq U(B)$, where $U(\cdot)$ is the utility function.
- To this end, we introduce a discount factor $\gamma \in [0, 1]$ for the subsequent rewards in order to establish a customizable tradeoff between short-term and long-term consequences during implementation. Thus the value function is defined as:
    
    $$
    V(s) = \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | s_t=s, \pi]
    $$
    
    This signifies that the agent now maximizes the cumulative discounted reward, or the discounted return. Since this is a very beneficial and popular choice, in most cases, it will be implied that our rewards are discounted.
    
- It is necessary to understand that actions may have long-term consequences as well, and since the value of a state depends on the actions that we choose starting from that state, this is reflected in the values as well. It may thus be beneficial to sacrifice short-term rewards for long-term rewards.
- The value $V(s)$  depends on what action the agent chooses, hence it also depends on the policy of the agent $\pi(s)$ which denotes the actions that it shall take given a state $s$ as input.
Thus we can write the value function as
    
    $$
    V_{\pi}(s) = \mathbb{E}[R_{t+1} + R_{t+2} + ...| s_t = s, a_t  \sim \pi(s)]
    $$
    
- The recurrent form of the above equation is given below, which is called the *****************Bellman Equation*****************, first described by Richard Bellman in 1957*****************.*****************
    
    $$
    V_{\pi}(s) = \mathbb{E}[R_{t+1} + \gamma V_{\pi}(s_{t+1})| s_t=s, a_t \sim \pi(s)]
    $$
    
- This form is useful as we can turn this into algorithm. A similar equation for the optimal value function is given by:
    
    $$
    V_{*}(s) = \max_{a}\mathbb{E}[R_{t+1} + \gamma V_{*}(s_{t+1})|s_t=s, a]
    $$
    
    Note that this does *********************not********************* depend on a policy.
    
- We heavily exploit such equations and use them to create algorithms.
- Agents often approximate value functions. With an accurate value function we can behave optimally. With suitable approximations we can behave well, even in large and intractable domains.

# Model

- A model predicts what the environment will do next.
- For example $\mathcal{P}$ which predicts the next state according to
    
    $$
    \mathcal{P}(s, a, s') \approx p(s_{t+1}=s'|s_t=s, a_t=a)
    $$
    
- For example $\mathcal{R}$ which predicts the next rewards according to
    
    $$
    \mathcal{R}(s, a) \approx \mathbb{E}[R_{t+1}|s_t=s, a_t=a]
    $$
    
- For a model we do not have a policy to get actions so we use some planning methods. Models require additional computation to extract a good policy.
- We could also consider a ***stochastic (generative) model.***

# Agent Categories

- We can consider multiple configurations or categories for our reinforcement learning agents.
- Policy and/or value function:
    - Value based:
        - No policy
        - Value function
    - Policy based:
        - No value function
        - Explicit policy
    - Actor-critic:
        - Explicit value function (the critic which critiques the actions that the policy takes. Used to update the policy)
        - Explicit policy (takes the actions or acts)
- Model-free:
    - Policy and/or value function
    - No model
- Model-based:
    - Optionally policy and/or value function
    - Explicit model

# Subproblems of Reinforcement Learning

- Prediction: evaluating the future (for a given policy)
- Control: optimizing the future (finding the best policy)
- These are strongly related, as if we have good predictions, then we can use that to pick policies
    
    $$
    \pi_*(s) = \argmax_{\pi}V_{\pi}(s)
    $$
    
- An interesting question: if we could predict everything do we need anything else?
    - Answer: YES. If we can predict everything then we know “what” everything is, or will lead to. But we need to have an idea of “how” to do things, i.e., take the decisions leading to best option.
- Two fundamental problems in reinforcement learning:
    - Learning:
        - The environment is initially unknown
        - The agent interacts with the environment to learn
    - Planning:
        - The model of the environment is given or learnt
        - Agent plans in this model without external interaction
        - Involves reasoning about the model, pondering, searching, planning.
        - Planning is determining the steps in order to execute a process to reach a goal.