from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and GPU driver.")
    env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = PPO("MlpPolicy", env, device="cuda", verbose=1)
    Path("models").mkdir(exist_ok=True)
    model.learn(total_timesteps=15_000)
    model.save("models/ppo_cartpole_partial")
    model.learn(total_timesteps=85_000)
    model.save("models/ppo_cartpole")
    env.close()


if __name__ == "__main__":
    main()