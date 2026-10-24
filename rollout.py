import argparse
from pathlib import Path

import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed


DEFAULT_CHECKPOINT = Path("models/ppo_cartpole.zip")
DEFAULT_OUTPUT = Path("data/rollouts.csv")
DEFAULT_EPISODES = 3_000
DEFAULT_SEED = 42
CHECKPOINT_EVERY = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Collect clean PPO CartPole rollouts.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_rows(rows, output, write_header):
    data = pd.DataFrame(rows)
    data["near_failure"] = data["near_failure"].astype(bool)
    data.to_csv(output, mode="w" if write_header else "a", header=write_header, index=False)


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {args.checkpoint}")

    # CPU inference is deterministic and efficient for this small policy network.
    set_random_seed(args.seed, using_cuda=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    model = PPO.load(args.checkpoint, device="cpu")
    env = gym.make("CartPole-v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)

    buffered_rows = []
    total_states = 0
    successful_episodes = 0
    failed_episodes = 0
    write_header = True

    try:
        for episode_id in range(args.episodes):
            episode_seed = args.seed + episode_id
            state, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            episode_rows = []
            done = False
            terminated = False
            truncated = False

            while not done:
                action, _ = model.predict(state, deterministic=True)
                action = int(action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_rows.append(
                    {
                        "episode_id": episode_id,
                        "step_idx": len(episode_rows),
                        "state_0": state[0],
                        "state_1": state[1],
                        "state_2": state[2],
                        "state_3": state[3],
                        "action": action,
                        "reward": reward,
                    }
                )
                state = next_state
                done = terminated or truncated

            episode_steps = len(episode_rows)
            outcome = "success" if truncated and not terminated else "fail"
            successful_episodes += outcome == "success"
            failed_episodes += outcome == "fail"
            for row in episode_rows:
                row["outcome"] = outcome
                row["steps_to_end"] = episode_steps - row["step_idx"] - 1
                row["near_failure"] = outcome == "fail" and row["steps_to_end"] <= 10

            buffered_rows.extend(episode_rows)
            total_states += episode_steps

            should_flush = (episode_id + 1) % CHECKPOINT_EVERY == 0
            is_last_episode = episode_id + 1 == args.episodes
            if should_flush or is_last_episode:
                write_rows(buffered_rows, temporary_output, write_header)
                write_header = False
                buffered_rows.clear()
                print(
                    f"episodes: {episode_id + 1}/{args.episodes}, "
                    f"states: {total_states}, successes: {successful_episodes}, "
                    f"failures: {failed_episodes}",
                    flush=True,
                )
    finally:
        env.close()

    temporary_output.replace(args.output)
    success_rate = successful_episodes / args.episodes
    print(f"checkpoint: {args.checkpoint}")
    print(f"seed: {args.seed}")
    print(f"episodes: {args.episodes}")
    print(f"total states: {total_states}")
    print(f"successful episodes: {successful_episodes}")
    print(f"failed episodes: {failed_episodes}")
    print(f"success rate: {success_rate:.4%}")
    print(f"dataset: {args.output}")


if __name__ == "__main__":
    main()
