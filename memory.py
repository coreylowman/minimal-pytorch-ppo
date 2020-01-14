import torch


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.logits = []
        self.values = []
        self.hxs = []

    def store(self, state, action, reward, done, logits, value, hx):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logits.append(logits)
        self.values.append(value)
        self.hxs.append(hx)

    def combine(self, other):
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.dones.extend(other.dones)
        self.logits.extend(other.logits)
        self.values.extend(other.values)
        self.hxs.extend(other.hxs)
        return self

    def get(self):
        return (
            torch.stack(self.states),
            torch.tensor(self.actions, dtype=torch.int),
            torch.tensor(self.rewards, dtype=torch.int),
            torch.tensor(self.dones, dtype=torch.bool),
            torch.stack(self.logits),
            torch.tensor(self.values, dtype=torch.float),
            torch.stack(self.hxs),
        )
