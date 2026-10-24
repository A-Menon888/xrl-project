from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("data/controlled_rollouts.csv")
MODEL_PATH = Path("models/tree_outcome.joblib")
FEATURES = ["state_0", "state_1", "state_2", "state_3"]
LABELS = {
    "state_0": "Cart position",
    "state_1": "Cart velocity",
    "state_2": "Pole angle (radians)",
    "state_3": "Pole angular velocity",
}
SHORT_LABELS = {
    "state_0": "cart position",
    "state_1": "cart velocity",
    "state_2": "pole angle",
    "state_3": "pole angular velocity",
}


def initial_states_from_rollouts(data):
    initial_states = data[data["step_idx"] == 0].copy()
    if initial_states["episode_id"].duplicated().any():
        raise ValueError("Every episode must have exactly one timestep-0 row")
    if initial_states["episode_id"].nunique() != data["episode_id"].nunique():
        raise ValueError("Every episode must contain a timestep-0 row")
    return initial_states


def plot_probability_slice(
    initial_states,
    tree,
    x_feature,
    y_feature,
    fixed_values,
    filename,
    title,
):
    x_values = np.linspace(
        initial_states[x_feature].min(), initial_states[x_feature].max(), 300
    )
    y_values = np.linspace(
        initial_states[y_feature].min(), initial_states[y_feature].max(), 300
    )
    grid_x, grid_y = np.meshgrid(x_values, y_values)

    grid = pd.DataFrame(0.0, index=np.arange(grid_x.size), columns=FEATURES)
    grid[x_feature] = grid_x.ravel()
    grid[y_feature] = grid_y.ravel()
    for feature, value in fixed_values.items():
        grid[feature] = value

    failure_class_index = list(tree.classes_).index("fail")
    failure_probability = tree.predict_proba(grid)[:, failure_class_index]
    failure_probability = failure_probability.reshape(grid_x.shape)

    slice_mask = np.ones(len(initial_states), dtype=bool)
    for feature, value in fixed_values.items():
        slice_mask &= np.isclose(initial_states[feature], value)
    observed_slice = initial_states[slice_mask]
    successes = observed_slice[observed_slice["outcome"] == "success"]
    failures = observed_slice[observed_slice["outcome"] == "fail"]

    fig, axis = plt.subplots(figsize=(9, 7.2))
    probability_plot = axis.contourf(
        grid_x,
        grid_y,
        failure_probability,
        levels=np.linspace(0.0, 1.0, 21),
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=1.0,
    )
    if failure_probability.min() < 0.5 < failure_probability.max():
        axis.contour(
            grid_x,
            grid_y,
            failure_probability,
            levels=[0.5],
            colors="white",
            linewidths=1.2,
            linestyles="--",
        )

    axis.scatter(
        successes[x_feature],
        successes[y_feature],
        s=65,
        marker="o",
        c="#167a3f",
        edgecolors="white",
        linewidths=0.9,
        label="Observed success",
        zorder=3,
        clip_on=False,
    )
    axis.scatter(
        failures[x_feature],
        failures[y_feature],
        s=70,
        marker="X",
        c="#202020",
        edgecolors="white",
        linewidths=0.7,
        label="Observed failure",
        zorder=3,
        clip_on=False,
    )

    fixed_description = ", ".join(
        f"{SHORT_LABELS[feature]} = {value:.2f}"
        for feature, value in fixed_values.items()
    )
    axis.set_title(
        f"{title}\n2D surrogate slice with {fixed_description}",
        fontsize=13,
        pad=12,
    )
    axis.set_xlabel(LABELS[x_feature])
    axis.set_ylabel(LABELS[y_feature])
    axis.legend(title="Controlled rollout outcome", loc="best", framealpha=0.95)
    axis.grid(color="white", alpha=0.2, linewidth=0.6)

    colorbar = fig.colorbar(probability_plot, ax=axis, pad=0.02)
    colorbar.set_label("Decision Tree predicted P(failure)")
    fig.text(
        0.5,
        0.012,
        "Background: model prediction. Markers: observed initial-state outcomes. "
        "This is a 2D slice of the learned 4D approximation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        f"saved {filename}: varied {x_feature}, {y_feature}; "
        f"fixed {fixed_values}; observed slice points: {len(observed_slice)}"
    )


def main():
    data = pd.read_csv(DATA_PATH)
    initial_states = initial_states_from_rollouts(data)
    tree = joblib.load(MODEL_PATH)

    plot_probability_slice(
        initial_states,
        tree,
        x_feature="state_2",
        y_feature="state_3",
        fixed_values={"state_0": 0.0, "state_1": 0.0},
        filename="failure_envelope.png",
        title="Learned failure envelope: pole dynamics",
    )
    plot_probability_slice(
        initial_states,
        tree,
        x_feature="state_0",
        y_feature="state_1",
        fixed_values={"state_2": 0.0, "state_3": 0.0},
        filename="failure_envelope_cart.png",
        title="Learned failure envelope: cart dynamics",
    )


if __name__ == "__main__":
    main()
