# Value-based
## Off-policy
### 1. [Double Q learning(DDQN)](!https://arxiv.org/pdf/1509.06461.pdf)
- `model/double_dqn.py`
- The idea of double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation.

<p style="align:center">
    <img src='https://cugtyt.github.io/blog/rl-notes/R/fix-target-update-param.png' width="550">
</p>

<br><br>

# Policy-based

<br><br>

# Exploration
### 1. [Hindsight Experience Replay(HER)](!https://arxiv.org/pdf/1707.01495.pdf)
- `src/replay.py HerSampler`
- The HER buffer attempts to overcome this by copying each trajectory experienced and replacing the actual rewards with rewards calculated assuming the goals are the steps achieved at the end of the trajectories. The idea is this will help the agent explore better and also learn intermediate goals that build up to the actual desired goal.