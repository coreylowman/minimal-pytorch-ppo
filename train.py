from torch.distributions import Categorical
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
import datetime
import torch
import gym

from memory import Memory
from policy import MLPPolicy, GRUPolicy, LSTMPolicy
from ppo import generalized_advantage_estimation, ppo_loss


def gather_training_data(env, policy, num_games=50, max_steps=100):
    policy.eval()

    memory = Memory()
    for _ in range(num_games):
        obs = env.reset()
        hx = policy.get_init_hx()
        for _ in range(max_steps):
            with torch.no_grad():
                step_logits, step_value, new_hx = policy(torch.from_numpy(obs).float(), hx)

            action = Categorical(logits=step_logits).sample().item()  # TODO add gaussian noise to policy to explore?

            new_obs, reward, done, info = env.step(action)

            memory.store(torch.from_numpy(obs).float(), action, reward, done, step_logits, step_value, hx)

            obs = new_obs
            hx = new_hx
            if done:
                break

        memory.dones[-1] = True
    return memory.get()


def evaluate(env, policy, num_games=10, max_steps=100):
    policy.eval()
    total_reward = 0
    for _ in range(num_games):
        obs = env.reset()
        hx = policy.get_init_hx()
        for _ in range(max_steps):
            with torch.no_grad():
                logits, value, hx = policy(torch.from_numpy(obs).float(), hx)
            obs, reward, done, info = env.step(logits.argmax(dim=-1).item())
            total_reward += reward
            if done:
                break
    return total_reward


def train(num_epochs=4, mini_batch_size=32, entropy_coef=0.01, value_coef=0.5):
    env = gym.make('CartPole-v0')

    policy: torch.nn.Module = LSTMPolicy(env.observation_space.shape[0], env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, eps=1e-6)

    while True:
        states, actions, rewards, dones, logits, values, hxs = gather_training_data(env, policy)

        advantages = generalized_advantage_estimation(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        target_values = advantages + values.squeeze()

        n = states.shape[0]

        policy.train()
        for i_epoch in range(num_epochs):
            for indices in BatchSampler(SubsetRandomSampler(list(range(n))), mini_batch_size, drop_last=True):
                new_logits, new_values, new_hxs = policy(states[indices], hxs[indices])

                policy_loss, entropy = ppo_loss(logits[indices], new_logits, actions[indices], advantages[indices])
                value_loss = (target_values[indices] - torch.squeeze(new_values)).pow(2).mean()

                loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f'[{datetime.datetime.now()}] steps={n} eval={evaluate(env, policy)}')


if __name__ == '__main__':
    train()
