import torch
from torch.distributions import Categorical


def generalized_advantage_estimation(rewards, values, dones, gamma=0.99, lam=0.95):
    last_part = 0
    advantages = torch.zeros(rewards.size(0))
    for t in reversed(range(rewards.size(0))):
        if dones[t]:
            delta = rewards[t] - values[t]
            last_part = 0
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = last_part = delta + gamma * lam * last_part
    return advantages


def ppo_loss(old_logits, new_logits, actions, advantages, clip_param=0.2):
    old_distribution = Categorical(logits=old_logits)
    old_log_prob = old_distribution.log_prob(actions)

    new_distribution = Categorical(logits=new_logits)
    new_log_prob = new_distribution.log_prob(actions)

    ratio = torch.exp(new_log_prob - old_log_prob)
    surr1 = -advantages * ratio
    surr2 = -advantages * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
    policy_loss = torch.max(surr1, surr2).mean()

    entropy = new_distribution.entropy().mean()

    return policy_loss, entropy
