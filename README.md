# Minimal Pytorch PPO

The intent of this repo is to implement PPO in Pytorch in a minimal amount of code.

## What's included?

- Network Structures
  - [x] Feedforward
  - [x] LSTM
  - [x] GRU
  - [ ] Conv
- Action Spaces
  - [x] Categorical Action Spaces
  - [ ] Continuous Action Spaces
- [x] Generic PPO loss function
- [x] Generalize Advantage Estimate function
- [x] Sample training script that uses all the above.

The code was tested on the `Cartpole-v0` OpenAI Gym Environment.
