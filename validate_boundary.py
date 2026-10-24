import argparse
from pathlib import Path

import gymnasium as gym
import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CHECKPOINT = Path("models/ppo_cartpole.zip")
DEFAULT_TREE = Path("models/tree_outcome.joblib")
DEFAULT_CONTROLLED_DATA = Path("data/controlled_rollouts.csv")
DEFAULT_OUTPUT = Path("data/boundary_validation.csv")
DEFAULT_PLOT = Path("boundary_validation.png")
DEFAULT_SEED = 42
DEFAULT_POOL_SIZE = 100_000

FEATURES = ["state_0", "state_1", "state_2", "state_3"]
FEATURE_NAMES = [
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
]
NEAR_TOLERANCE = 0.05
NEAR_PER_SIDE = 60
CONTROL_PER_CLASS = 40
CLEAR_SUCCESS_MAX = 0.25
CLEAR_FAILURE_MIN = 0.90
CLASS_LABELS = ["fail", "success"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the learned failure boundary on new CartPole states."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--controlled-data", type=Path, default=DEFAULT_CONTROLLED_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    return parser.parse_args()


def controlled_initial_states(data):
    initial_states = data[data["step_idx"] == 0].copy()
    if initial_states["episode_id"].duplicated().any():
        raise ValueError("Every controlled episode must have one timestep-0 row")
    return initial_states


def sample_indices(rng, eligible, count, name):
    indices = np.flatnonzero(eligible)
    if len(indices) < count:
        raise RuntimeError(
            f"Candidate pool has only {len(indices)} states for {name}; need {count}"
        )
    return rng.choice(indices, size=count, replace=False)


def generate_validation_states(tree, initial_states, seed, pool_size):
    rng = np.random.default_rng(seed)
    ranges = np.column_stack(
        ([initial_states[feature].min() for feature in FEATURES],
         [initial_states[feature].max() for feature in FEATURES])
    )
    candidates = rng.uniform(ranges[:, 0], ranges[:, 1], size=(pool_size, 4))

    # Exclude all original grid points. Rounding only makes the exact-membership
    # check robust to CSV floating-point representation.
    original_keys = {
        tuple(np.round(row, 12)) for row in initial_states[FEATURES].to_numpy()
    }
    new_state_mask = np.array(
        [tuple(np.round(row, 12)) not in original_keys for row in candidates]
    )
    candidates = candidates[new_state_mask]
    candidate_frame = pd.DataFrame(candidates, columns=FEATURES)

    failure_index = list(tree.classes_).index("fail")
    scores = tree.predict_proba(candidate_frame[FEATURES])[:, failure_index]
    distance = np.abs(scores - 0.5)

    selections = [
        (
            sample_indices(
                rng,
                (distance <= NEAR_TOLERANCE) & (scores < 0.5),
                NEAR_PER_SIDE,
                "near-boundary predicted success",
            ),
            "near_boundary",
        ),
        (
            sample_indices(
                rng,
                (distance <= NEAR_TOLERANCE) & (scores >= 0.5),
                NEAR_PER_SIDE,
                "near-boundary predicted failure",
            ),
            "near_boundary",
        ),
        (
            sample_indices(
                rng,
                scores <= CLEAR_SUCCESS_MAX,
                CONTROL_PER_CLASS,
                "clear predicted-success control",
            ),
            "clear_success_control",
        ),
        (
            sample_indices(
                rng,
                scores >= CLEAR_FAILURE_MIN,
                CONTROL_PER_CLASS,
                "clear predicted-failure control",
            ),
            "clear_failure_control",
        ),
    ]

    selected_parts = []
    selected_indices = set()
    for indices, group in selections:
        overlap = selected_indices.intersection(indices.tolist())
        if overlap:
            raise RuntimeError("Validation sampling groups unexpectedly overlap")
        selected_indices.update(indices.tolist())
        part = candidate_frame.iloc[indices].copy()
        part["sampling_group"] = group
        part["surrogate_failure_score"] = scores[indices]
        selected_parts.append(part)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected.insert(0, "validation_id", np.arange(len(selected)))
    selected["is_near_boundary"] = selected["sampling_group"].eq("near_boundary")
    selected["surrogate_prediction"] = np.where(
        selected["surrogate_failure_score"] >= 0.5, "fail", "success"
    )

    selected_keys = {
        tuple(np.round(row, 12)) for row in selected[FEATURES].to_numpy()
    }
    if selected_keys & original_keys:
        raise RuntimeError("Validation states overlap the original controlled grid")
    if len(selected_keys) != len(selected):
        raise RuntimeError("Validation states must be unique")
    return selected, ranges, len(candidate_frame)


def evaluate_frozen_policy(states, checkpoint, seed):
    set_random_seed(seed, using_cuda=False)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device="cpu")
    env = gym.make("CartPole-v1")

    actual_outcomes = []
    episode_steps = []
    terminated_values = []
    truncated_values = []
    try:
        for row in states.itertuples(index=False):
            episode_seed = seed + 100_000 + row.validation_id
            env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            initial_state = np.array(
                [row.state_0, row.state_1, row.state_2, row.state_3],
                dtype=np.float64,
            )
            env.unwrapped.state = initial_state
            state = initial_state.astype(np.float32)
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(state, deterministic=True)
                state, _, terminated, truncated, _ = env.step(int(action))
                steps += 1

            actual_outcomes.append(
                "success" if truncated and not terminated else "fail"
            )
            episode_steps.append(steps)
            terminated_values.append(bool(terminated))
            truncated_values.append(bool(truncated))
            completed = row.validation_id + 1
            if completed % 25 == 0 or completed == len(states):
                successes = actual_outcomes.count("success")
                failures = actual_outcomes.count("fail")
                print(
                    f"evaluated: {completed}/{len(states)}, "
                    f"actual successes: {successes}, actual failures: {failures}",
                    flush=True,
                )
    finally:
        env.close()

    results = states.copy()
    results["actual_outcome"] = actual_outcomes
    results["agreement"] = results["surrogate_prediction"].eq(
        results["actual_outcome"]
    )
    results["episode_steps"] = episode_steps
    results["terminated"] = terminated_values
    results["truncated"] = truncated_values
    return results


def metric_summary(data, name):
    actual = data["actual_outcome"]
    predicted = data["surrogate_prediction"]
    matrix = confusion_matrix(actual, predicted, labels=CLASS_LABELS)
    print(f"\n{name} ({len(data)} states)")
    print(f"actual outcomes: {actual.value_counts().to_dict()}")
    print(f"accuracy/agreement: {accuracy_score(actual, predicted):.4f}")
    print(f"balanced accuracy: {balanced_accuracy_score(actual, predicted):.4f}")
    print(
        f"failure precision: "
        f"{precision_score(actual, predicted, pos_label='fail', zero_division=0):.4f}"
    )
    print(
        f"failure recall: "
        f"{recall_score(actual, predicted, pos_label='fail', zero_division=0):.4f}"
    )
    print(
        f"failure F1: "
        f"{f1_score(actual, predicted, pos_label='fail', zero_division=0):.4f}"
    )
    print("confusion matrix (rows=actual, columns=predicted; order=fail, success):")
    print(matrix)
    print(
        classification_report(
            actual,
            predicted,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            digits=4,
            zero_division=0,
        )
    )


def print_examples(results, agreement, count=3):
    label = "agreement" if agreement else "disagreement"
    examples = results[results["agreement"].eq(agreement)].head(count)
    print(f"{label} examples:")
    for row in examples.itertuples(index=False):
        state = ", ".join(
            f"{name}={value:.4f}"
            for name, value in zip(
                FEATURE_NAMES,
                [row.state_0, row.state_1, row.state_2, row.state_3],
            )
        )
        print(
            f"  id={row.validation_id}, group={row.sampling_group}, "
            f"score={row.surrogate_failure_score:.4f}, "
            f"predicted={row.surrogate_prediction}, actual={row.actual_outcome}, "
            f"{state}"
        )


def plot_validation(results, output, seed):
    rng = np.random.default_rng(seed + 1)
    actual_numeric = results["actual_outcome"].map({"success": 0.0, "fail": 1.0})
    x_jitter = rng.uniform(-0.006, 0.006, size=len(results))
    y_jitter = rng.uniform(-0.075, 0.075, size=len(results))
    x_values = results["surrogate_failure_score"].to_numpy() + x_jitter
    y_values = actual_numeric.to_numpy() + y_jitter

    fig, axis = plt.subplots(figsize=(10, 6.5))
    axis.axvspan(0.0, 0.45, color="#d9f0d3", alpha=0.45, label="Predicted success region")
    axis.axvspan(0.45, 0.55, color="#fee08b", alpha=0.45, label="Near-boundary score band")
    axis.axvspan(0.55, 1.0, color="#f4a6a6", alpha=0.35, label="Predicted failure region")
    axis.axvline(0.5, color="black", linestyle="--", linewidth=1.2)

    success_mask = results["actual_outcome"].eq("success").to_numpy()
    failure_mask = ~success_mask
    axis.scatter(
        x_values[success_mask],
        y_values[success_mask],
        s=45,
        marker="o",
        c="#16834a",
        edgecolors="white",
        linewidths=0.6,
        label="Actual PPO success",
        zorder=3,
    )
    axis.scatter(
        x_values[failure_mask],
        y_values[failure_mask],
        s=50,
        marker="X",
        c="#242424",
        edgecolors="white",
        linewidths=0.5,
        label="Actual PPO failure",
        zorder=3,
    )
    disagreement = ~results["agreement"].to_numpy()
    axis.scatter(
        x_values[disagreement],
        y_values[disagreement],
        s=115,
        marker="o",
        facecolors="none",
        edgecolors="#c51b8a",
        linewidths=1.4,
        label="Surrogate disagreement",
        zorder=4,
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.18, 1.18)
    axis.set_yticks([0, 1], labels=["Actual success", "Actual failure"])
    axis.set_xlabel("Decision Tree predicted failure score (not calibrated)")
    axis.set_title(
        "Boundary validation on new controlled initial states\n"
        "True outcomes from deterministic frozen-PPO rollouts"
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95)
    fig.text(
        0.5,
        0.015,
        "Points are vertically jittered for readability; magenta rings mark prediction/outcome disagreements.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    for path in (args.checkpoint, args.tree, args.controlled_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")

    controlled_data = pd.read_csv(args.controlled_data)
    initial_states = controlled_initial_states(controlled_data)
    tree = joblib.load(args.tree)
    validation_states, ranges, candidate_count = generate_validation_states(
        tree, initial_states, args.seed, args.pool_size
    )
    print(
        f"selected {len(validation_states)} new states from {candidate_count} "
        f"seeded candidates; evaluating frozen PPO policy...",
        flush=True,
    )
    results = evaluate_frozen_policy(validation_states, args.checkpoint, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    plot_validation(results, args.plot, args.seed)

    print(f"seed: {args.seed}")
    print(f"candidate pool after excluding original states: {candidate_count}")
    for name, bounds in zip(FEATURE_NAMES, ranges):
        print(f"{name} range: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
    print(f"validation states: {len(results)}")
    print(f"near-boundary states: {int(results['is_near_boundary'].sum())}")
    print(
        "near-boundary definition: "
        f"abs(surrogate failure score - 0.5) <= {NEAR_TOLERANCE:.2f}"
    )
    print(f"sampling groups: {results['sampling_group'].value_counts().to_dict()}")
    print("group outcomes and agreement:")
    print(
        results.groupby("sampling_group")
        .agg(
            states=("validation_id", "size"),
            actual_failures=("actual_outcome", lambda values: (values == "fail").sum()),
            actual_successes=("actual_outcome", lambda values: (values == "success").sum()),
            agreement_rate=("agreement", "mean"),
        )
        .to_string()
    )

    metric_summary(results, "Overall validation")
    metric_summary(results[results["is_near_boundary"]], "Near-boundary validation")
    print_examples(results[results["is_near_boundary"]], agreement=True)
    print_examples(results[results["is_near_boundary"]], agreement=False)
    print(f"saved validation data: {args.output}")
    print(f"saved validation plot: {args.plot}")


if __name__ == "__main__":
    main()
