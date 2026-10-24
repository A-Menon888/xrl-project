from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


DATA_PATH = Path("data/controlled_rollouts.csv")
MODEL_PATH = Path("models/tree_outcome.joblib")
RULES_PATH = Path("rules_outcome.txt")
RANDOM_SEED = 42
FEATURES = ["state_0", "state_1", "state_2", "state_3"]
FEATURE_NAMES = [
    "cart_position",
    "cart_velocity",
    "pole_angle",
    "pole_angular_velocity",
]
CLASS_LABELS = ["fail", "success"]


def extract_initial_states(data):
    outcome_counts = data.groupby("episode_id")["outcome"].nunique()
    if not outcome_counts.eq(1).all():
        raise ValueError("Every episode must have exactly one outcome label")

    initial_states = data[data["step_idx"] == 0].copy()
    if initial_states["episode_id"].duplicated().any():
        raise ValueError("Every episode must have exactly one timestep-0 row")
    if initial_states["episode_id"].nunique() != data["episode_id"].nunique():
        raise ValueError("Every episode must contain a timestep-0 row")

    unique_state_count = initial_states[FEATURES].drop_duplicates().shape[0]
    if unique_state_count != len(initial_states):
        raise ValueError("Controlled initial states must be unique")
    return initial_states.sort_values("episode_id").reset_index(drop=True)


def split_by_episode(initial_states):
    train_ids, test_ids = train_test_split(
        initial_states["episode_id"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=initial_states["outcome"],
    )
    if set(train_ids) & set(test_ids):
        raise RuntimeError("Train and test episode IDs overlap")

    train_data = initial_states[initial_states["episode_id"].isin(train_ids)]
    test_data = initial_states[initial_states["episode_id"].isin(test_ids)]
    return train_data, test_data


def train_tree(train_data):
    tree = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    tree.fit(train_data[FEATURES], train_data["outcome"])
    return tree


def evaluate(tree, test_data):
    actual = test_data["outcome"]
    prediction = tree.predict(test_data[FEATURES])
    matrix = confusion_matrix(actual, prediction, labels=CLASS_LABELS)

    print(f"accuracy: {accuracy_score(actual, prediction):.4f}")
    print(f"balanced accuracy: {balanced_accuracy_score(actual, prediction):.4f}")
    print(f"failure precision: {precision_score(actual, prediction, pos_label='fail'):.4f}")
    print(f"failure recall: {recall_score(actual, prediction, pos_label='fail'):.4f}")
    print(f"failure F1: {f1_score(actual, prediction, pos_label='fail'):.4f}")
    print(f"macro F1: {f1_score(actual, prediction, average='macro'):.4f}")
    print("confusion matrix (rows=actual, columns=predicted; order=fail, success):")
    print(matrix)
    print("per-class metrics:")
    print(
        classification_report(
            actual,
            prediction,
            labels=CLASS_LABELS,
            target_names=CLASS_LABELS,
            digits=4,
            zero_division=0,
        )
    )


def main():
    data = pd.read_csv(DATA_PATH)
    initial_states = extract_initial_states(data)
    train_data, test_data = split_by_episode(initial_states)
    tree = train_tree(train_data)

    print(f"unique initial states: {len(initial_states)}")
    print(f"train size: {len(train_data)}")
    print(f"test size: {len(test_data)}")
    print(f"train class counts: {train_data['outcome'].value_counts().to_dict()}")
    print(f"test class counts: {test_data['outcome'].value_counts().to_dict()}")
    evaluate(tree, test_data)
    print(f"tree depth: {tree.get_depth()}")
    print(f"tree leaves: {tree.get_n_leaves()}")

    importances = dict(zip(FEATURE_NAMES, tree.feature_importances_))
    print("feature importances:")
    for name, importance in sorted(importances.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {importance:.4f}")

    rules = export_text(tree, feature_names=FEATURE_NAMES)
    print("tree rules:")
    print(rules)

    RULES_PATH.write_text(
        "Controlled-initial-state outcome tree\n"
        "Interpretable approximation of the frozen PPO policy over the controlled "
        "initial-state distribution; not a universal or exact failure boundary.\n\n"
        + rules,
        encoding="utf-8",
    )
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(tree, MODEL_PATH)
    print(f"saved model: {MODEL_PATH}")
    print(f"saved rules: {RULES_PATH}")


if __name__ == "__main__":
    main()
