# XRL CartPole Failure-Envelope Project

## Purpose and current status

This project explains **where a frozen PPO policy succeeds or fails** in the four-dimensional CartPole state space. It deliberately studies trajectory-level outcomes from controlled initial states rather than explaining individual actions. The current pipeline is complete through:

1. PPO training.
2. Clean nominal-policy evaluation.
3. Controlled initial-state evaluation.
4. Interpretable Decision Tree surrogate training.
5. Two-dimensional failure-envelope visualization.
6. Independent boundary-focused validation against fresh PPO rollouts.

The principal result is that the final deterministic PPO policy succeeds on all tested standard CartPole resets, but fails on many broader valid initial states. A depth-4 Decision Tree provides a useful global approximation over the chosen controlled range. Its agreement is strong in clear-score control regions and substantially weaker near its predicted boundary, so it must not be presented as a universal, exact, or calibrated failure boundary.

> **Important artifact warning:** `REPORT.md`, `models/tree_near_failure.joblib`, `rules_near_failure.txt`, and `near_failure_envelope.png` are legacy outputs from an earlier perturbed-rollout experiment. They are stale and must not be used to describe the current methodology or results. The current outcome model is `models/tree_outcome.joblib`; the current results are documented here.

## System architecture

```text
train.py
  └─ PPO + CartPole-v1
      ├─ models/ppo_cartpole_partial.zip       (legacy partial checkpoint)
      └─ models/ppo_cartpole.zip               (frozen final checkpoint)
             │
             ├─ rollout.py
             │    └─ data/rollouts.csv         (3,000 nominal-reset episodes)
             │
             └─ controlled_rollout.py
                  └─ data/controlled_rollouts.csv
                       │                         (625 full-factorial initial states)
                       └─ train_surrogate.py
                            ├─ models/tree_outcome.joblib
                            ├─ rules_outcome.txt
                            └─ visualize.py
                                 ├─ failure_envelope.png
                                 └─ failure_envelope_cart.png

models/tree_outcome.joblib + data/controlled_rollouts.csv
  └─ validate_boundary.py
       ├─ generates 200 new continuous states before observing PPO outcomes
       ├─ evaluates the frozen PPO on those states
       ├─ data/boundary_validation.csv
       └─ boundary_validation.png
```

The PPO checkpoint and Decision Tree are frozen during evaluation and validation. All current policy evaluations use deterministic PPO actions with no random-action injection.

## Environment and state semantics

The environment is Gymnasium `CartPole-v1`. Its observation/state ordering is:

| CSV field | Semantic name | Meaning |
| --- | --- | --- |
| `state_0` | cart position | Horizontal cart position |
| `state_1` | cart velocity | Horizontal cart velocity |
| `state_2` | pole angle | Pole angle in radians |
| `state_3` | pole angular velocity | Pole angular velocity |

The action space is discrete:

- `0`: push cart left.
- `1`: push cart right.

Relevant CartPole mechanics and labels:

- Standard reset samples every state coordinate uniformly from `[-0.05, 0.05]`.
- An episode terminates when cart position leaves `[-2.4, 2.4]` or pole angle leaves approximately `[-0.20944, 0.20944]` radians (plus or minus 12 degrees).
- Cart velocity and pole angular velocity are unbounded in the Gymnasium observation space.
- `CartPole-v1` has a 500-step `TimeLimit`.
- This project labels an episode `success` only when it is truncated at the 500-step limit without termination.
- It labels an episode `fail` when the environment returns `terminated=True`. If termination and truncation ever coincide, the code treats that episode as failure.
- CartPole supplies reward `1.0` for every recorded step, including the terminating step under its default reward mode.
- Rollout CSV rows contain the **pre-action state**, selected action, and resulting step reward. The final post-action terminal observation is not added as a separate row.

## Algorithms and methodology

### 1. PPO policy training

`train.py` creates one `CartPole-v1` environment inside Stable-Baselines3 `DummyVecEnv` and trains `PPO("MlpPolicy", ...)` on CUDA.

Training sequence:

- Request 15,000 timesteps, then save `models/ppo_cartpole_partial.zip`.
- Continue training the same in-memory policy for another requested 85,000 timesteps, then save `models/ppo_cartpole.zip`.
- PPO collects 2,048-step rollout batches, so the saved partial checkpoint reports 16,384 steps and the final archive reports 86,016 for the second `learn` call. The policy weights received approximately 102,400 cumulative environment steps.
- The second `learn` call uses Stable-Baselines3's default `reset_num_timesteps=True`, which resets the logged timestep counter but not the learned weights.

Saved PPO configuration:

- Algorithm: Proximal Policy Optimization (PPO).
- Policy/value networks: separate two-layer MLP paths, each `64 → 64` with `Tanh` activations.
- Observation dimension: 4.
- Policy action logits: 2.
- Rollout length: 2,048.
- Batch size: 64.
- Epochs per update: 10.
- Discount `gamma`: 0.99.
- GAE lambda: 0.95.
- PPO clip range: 0.2.
- Entropy coefficient: 0.0.
- Value-function coefficient: 0.5.
- Maximum gradient norm: 0.5.
- Advantage normalization: enabled.

Important limitation: `train.py` does not set a training seed (`seed=None` in the saved model), so retraining PPO is not guaranteed to recreate the current checkpoint. All downstream experiments are reproducible because they use the already-saved frozen checkpoint. Do not rerun `train.py` before a demonstration unless replacing the policy and regenerating every downstream artifact is intentional.

### 2. Clean nominal rollout evaluation

`rollout.py` evaluates the final checkpoint, not the partial checkpoint.

- Default checkpoint: `models/ppo_cartpole.zip`.
- Default output: `data/rollouts.csv`.
- Default episodes: 3,000.
- Default seed: 42.
- Episode reset seed: `42 + episode_id`.
- Inference device: CPU.
- `torch.use_deterministic_algorithms(True)` is enabled.
- PPO action selection uses `model.predict(..., deterministic=True)` on every step.
- There is no random action injection or action override.
- Data is streamed to a temporary CSV every 100 episodes and atomically moved into place after successful completion.

Observed nominal result:

| Quantity | Result |
| --- | ---: |
| Episodes | 3,000 |
| State rows | 1,500,000 |
| Successful episodes | 3,000 |
| Failed episodes | 0 |
| Success rate | 100% |

Every episode reached 500 steps. This dataset accurately represents nominal behavior under Gymnasium's narrow default reset distribution, but it cannot train a binary failure classifier because it has no failures.

### 3. Controlled initial-state evaluation

`controlled_rollout.py` keeps the same frozen checkpoint and deterministic policy but starts episodes from a broader reproducible full-factorial grid.

For each episode, the script first calls the normal seeded `env.reset(...)` so the base environment, wrapper, termination bookkeeping, and time-limit counter are reset. It then sets `env.unwrapped.state` to the controlled state. Direct state assignment is the simplest supported local approach because CartPole stores its four physical variables in `env.unwrapped.state`.

Five evenly spaced values are used per dimension:

| Variable | Values |
| --- | --- |
| Cart position | `[-2.0, -1.0, 0.0, 1.0, 2.0]` |
| Cart velocity | `[-1.5, -0.75, 0.0, 0.75, 1.5]` |
| Pole angle | `[-0.18, -0.09, 0.0, 0.09, 0.18]` radians |
| Pole angular velocity | `[-1.5, -0.75, 0.0, 0.75, 1.5]` |

This is a `5^4 = 625` state grid. Initial cart positions and pole angles are strictly inside the termination limits, so no episode begins in an already-failed position. Failures occur only after the unchanged dynamics and frozen policy advance the environment.

Observed controlled result:

| Quantity | Result |
| --- | ---: |
| Controlled episodes / unique initial states | 625 |
| State rows | 93,572 |
| Successful episodes | 178 |
| Failed episodes | 447 |
| Success rate | 28.48% |
| Successful episode length | 500 steps |
| Failed episode length range | 1–30 steps |

All 447 failures are genuine `terminated=True` results. No labels were manufactured and no actions were perturbed.

### 4. Interpretable outcome surrogate

`train_surrogate.py` reads `data/controlled_rollouts.csv` and extracts exactly the `step_idx == 0` row from every episode. Consequently, each initial state contributes one example, regardless of trajectory length.

Only these four features are used:

- `state_0`: cart position.
- `state_1`: cart velocity.
- `state_2`: pole angle.
- `state_3`: pole angular velocity.

The model does **not** use episode ID, timestep, action, reward, trajectory length, time-to-termination, `near_failure`, or any outcome-derived field as a feature.

Split and model:

- Unique examples: 625.
- Stratified 80/20 split of episode IDs with seed 42.
- Training set: 500 states (`358 fail`, `142 success`).
- Test set: 125 states (`89 fail`, `36 success`).
- Episode-ID overlap between train and test: zero.
- Model: `sklearn.tree.DecisionTreeClassifier`.
- `max_depth=4`.
- `class_weight="balanced"`.
- `random_state=42`.
- Fitted depth: 4.
- Leaves: 13.
- Saved model: `models/tree_outcome.joblib`.
- Exported human-readable rules: `rules_outcome.txt`.

Held-out test results:

| Metric | Value |
| --- | ---: |
| Accuracy | 70.40% |
| Balanced accuracy | 72.60% |
| Failure precision | 88.24% |
| Failure recall | 67.42% |
| Failure F1 | 76.43% |
| Macro F1 | 68.32% |

Per-class results:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Failure | 0.8824 | 0.6742 | 0.7643 | 89 |
| Success | 0.4912 | 0.7778 | 0.6022 | 36 |

Confusion matrix, with rows as actual labels and columns as predicted labels in `[fail, success]` order:

```text
[[60, 29],
 [ 8, 28]]
```

Feature importances:

| Feature | Importance |
| --- | ---: |
| Pole angular velocity | 36.80% |
| Cart position | 31.79% |
| Pole angle | 18.62% |
| Cart velocity | 12.80% |

The exported rules are:

```text
cart_velocity <= -1.12
├─ cart_position <= -0.50 → fail
└─ cart_position > -0.50
   ├─ pole_angular_velocity <= 0.38 → fail
   └─ pole_angular_velocity > 0.38
      ├─ pole_angle <= 0.14 → success
      └─ pole_angle > 0.14 → fail

-1.12 < cart_velocity <= 1.12
├─ pole_angular_velocity <= -1.12
│  ├─ pole_angle <= -0.05 → fail
│  └─ pole_angle > -0.05 → success
└─ pole_angular_velocity > -1.12
   ├─ pole_angle <= 0.14 → success
   └─ pole_angle > 0.14 → fail

cart_velocity > 1.12
├─ pole_angular_velocity <= -0.38
│  ├─ cart_position <= 0.50 → success
│  └─ cart_position > 0.50 → fail
└─ pole_angular_velocity > -0.38 → fail
```

Some exported terminal branches split further but lead to the same class; `rules_outcome.txt` is the authoritative exact `export_text` output.

The tree is a compact approximation over the specified controlled grid. Its `predict_proba` values are class-weighted leaf proportions. They are useful surrogate scores but have not been calibrated as probabilities of true PPO failure.

### 5. Failure-envelope visualization

`visualize.py` loads only `models/tree_outcome.joblib` and the timestep-0 states from `data/controlled_rollouts.csv`. It creates two exact two-dimensional slices of the learned four-dimensional model.

Both plots use:

- Background color: Decision Tree `predict_proba` estimate for class `fail` from 0 to 1.
- Dashed white contour: tree score 0.5.
- Green circles: observed successful controlled initial states on the exact slice.
- Black crosses: observed failed controlled initial states on the exact slice.
- Exactly 25 observed points per slice because the fixed coordinates equal values present in the original five-level grid.
- Explicit title and caption stating fixed values and warning that this is a 2D slice, not the complete 4D boundary.

Figures:

- `failure_envelope.png`: varies pole angle on the x-axis and pole angular velocity on the y-axis; fixes cart position `0.0` and cart velocity `0.0`.
- `failure_envelope_cart.png`: varies cart position on the x-axis and cart velocity on the y-axis; fixes pole angle `0.0` and pole angular velocity `0.0`.

These figures show model score regions and the actual outcomes on those slices. They do not show calibrated risk, and they do not prove behavior away from the displayed fixed values.

### 6. Independent boundary validation

`validate_boundary.py` tests whether the saved tree's learned boundary corresponds to actual frozen-PPO behavior on new states. It does not retrain or alter PPO or the tree.

Selection procedure, completed before observing any validation outcomes:

1. Use NumPy RNG seed 42.
2. Draw 100,000 continuous uniform candidate states across the same four controlled ranges.
3. Remove any exact match to one of the original 625 grid states (comparison is rounded to 12 decimal places for floating-point robustness).
4. Score candidates with `models/tree_outcome.joblib`.
5. Select 200 unique states without replacement:
   - 60 states on the predicted-success side with `abs(score - 0.5) <= 0.05`.
   - 60 states on the predicted-failure side with `abs(score - 0.5) <= 0.05`.
   - 40 clear predicted-success controls with score at most 0.25.
   - 40 clear predicted-failure controls with score at least 0.90.
6. Reproducibly shuffle the selected states.
7. Run the actual frozen PPO from each state with deterministic actions.

Because a Decision Tree has piecewise-constant leaf scores, the selected near-boundary states have scores approximately `0.463926` or `0.543367`; this is not a smooth geometric-distance measure. The 200 selected states are all unique and have zero overlap with the original 625 states.

Overall validation results:

| Quantity / metric | Result |
| --- | ---: |
| Validation states | 200 |
| Actual failures | 119 |
| Actual successes | 81 |
| Agreements | 143 |
| Disagreements | 57 |
| Accuracy / agreement | 71.50% |
| Balanced accuracy | 72.31% |
| Failure precision | 81.00% |
| Failure recall | 68.07% |
| Failure F1 | 73.97% |
| Macro F1 | 71.24% |

Overall confusion matrix, rows actual and columns predicted in `[fail, success]` order:

```text
[[81, 38],
 [19, 62]]
```

Near-boundary-only results:

| Quantity / metric | Result |
| --- | ---: |
| Near-boundary states | 120 |
| Actual failures | 74 |
| Actual successes | 46 |
| Accuracy / agreement | 58.33% |
| Balanced accuracy | 58.81% |
| Failure precision | 70.00% |
| Failure recall | 56.76% |
| Failure F1 | 62.69% |
| Macro F1 | 57.76% |

Near-boundary confusion matrix:

```text
[[42, 32],
 [18, 28]]
```

Control-group results:

| Sampling group | States | Actual failures | Actual successes | Agreement |
| --- | ---: | ---: | ---: | ---: |
| Clear predicted-failure | 40 | 39 | 1 | 97.50% |
| Clear predicted-success | 40 | 6 | 34 | 85.00% |
| Near boundary | 120 | 74 | 46 | 58.33% |

Every validation failure has `terminated=True`; every validation success has `truncated=True`. The result supports the modest claim that the surrogate captures clear regions reasonably well, while its approximation is uncertain and often wrong near the learned boundary. Because the sample deliberately over-represents boundary states and high-confidence controls, the 71.50% overall agreement is not an estimate for a natural initial-state distribution.

Example near-boundary cases:

- Agreement: `[-0.4258, 1.0093, 0.0781, -1.1698]`, score `0.4639`, predicted success, actual success.
- Disagreement: `[1.6306, -1.3634, 0.0441, -0.0070]`, score `0.5434`, predicted failure, actual success.
- Disagreement: `[0.1239, -0.3297, 0.0380, -1.1730]`, score `0.4639`, predicted success, actual failure.

`boundary_validation.png` represents:

- X-axis: uncalibrated Decision Tree failure score.
- Y-axis: true PPO outcome (`Actual success` or `Actual failure`).
- Pale green/red regions: surrogate-predicted success/failure sides.
- Yellow band: the `[0.45, 0.55]` near-boundary score interval.
- Green circles: actual PPO successes.
- Black crosses: actual PPO failures.
- Magenta rings: surrogate/actual disagreements.
- Small deterministic vertical and horizontal jitter: display-only separation of overlapping points; it does not alter stored scores or labels.

## Data schemas

### `data/rollouts.csv` and `data/controlled_rollouts.csv`

Both files have the same 11-column downstream interface:

| Column | Meaning |
| --- | --- |
| `episode_id` | Integer trajectory identifier |
| `step_idx` | Zero-based timestep within trajectory |
| `state_0` | Pre-action cart position |
| `state_1` | Pre-action cart velocity |
| `state_2` | Pre-action pole angle in radians |
| `state_3` | Pre-action pole angular velocity |
| `action` | Deterministic PPO action (`0` or `1`) |
| `reward` | Environment step reward |
| `outcome` | Eventual episode label: `success` or `fail`, copied to every row |
| `steps_to_end` | Remaining recorded transitions after this row |
| `near_failure` | True only for failed trajectories with `steps_to_end <= 10`; retained for compatibility but not used by the current outcome surrogate |

### `data/boundary_validation.csv`

This is a one-row-per-validation-state summary, not a trajectory-row dataset:

| Column | Meaning |
| --- | --- |
| `validation_id` | Reproducibly shuffled integer identifier |
| `state_0`–`state_3` | New controlled initial state |
| `sampling_group` | `near_boundary`, `clear_success_control`, or `clear_failure_control` |
| `surrogate_failure_score` | Tree leaf score for class `fail`; not calibrated |
| `is_near_boundary` | Whether the state meets the score-band definition |
| `surrogate_prediction` | `fail` when score is at least 0.5, otherwise `success` |
| `actual_outcome` | True result from the frozen PPO rollout |
| `agreement` | Whether surrogate prediction equals actual outcome |
| `episode_steps` | True rollout length |
| `terminated` | Gymnasium termination flag at episode end |
| `truncated` | Gymnasium time-limit flag at episode end |

## File and artifact inventory

| Path | Status and responsibility |
| --- | --- |
| `train.py` | Trains PPO on CUDA and writes partial/final checkpoints. Training is not seeded. |
| `rollout.py` | Clean nominal-reset evaluation; deterministic final PPO; writes `data/rollouts.csv`. |
| `controlled_rollout.py` | Full-factorial controlled-state evaluation; writes `data/controlled_rollouts.csv`. |
| `train_surrogate.py` | Extracts one initial row per controlled episode, splits episode IDs, trains/evaluates/saves the outcome tree. |
| `visualize.py` | Creates current probability-score slices from the outcome tree. |
| `validate_boundary.py` | Generates fresh score-targeted states, executes PPO, reports validation metrics, writes CSV/plot. |
| `requirements.txt` | Python dependencies and CUDA PyTorch index/pin. Most packages are not version-pinned. |
| `context.md` | Authoritative current technical and experimental context. |
| `models/ppo_cartpole.zip` | **Current frozen final PPO checkpoint.** |
| `models/ppo_cartpole_partial.zip` | Earlier partial checkpoint; not used by current experiments. |
| `models/tree_outcome.joblib` | **Current depth-4 controlled-state outcome surrogate.** |
| `rules_outcome.txt` | Current exact human-readable outcome-tree rules. |
| `data/rollouts.csv` | Current 3,000-episode nominal dataset; all successes. |
| `data/controlled_rollouts.csv` | Current 625-state full-factorial dataset; used to train/test tree. |
| `data/boundary_validation.csv` | Current 200-state independent validation summary. |
| `failure_envelope.png` | Current pole-angle/pole-angular-velocity model slice. |
| `failure_envelope_cart.png` | Current cart-position/cart-velocity model slice. |
| `boundary_validation.png` | Current new-state surrogate-vs-PPO validation plot. |
| `REPORT.md` | **Stale legacy report; do not cite as current.** It describes the old 1,000-episode perturbed dataset. |
| `models/tree_near_failure.joblib` | **Stale legacy model; not used.** |
| `rules_near_failure.txt` | **Stale legacy rules; not used.** |
| `near_failure_envelope.png` | **Stale legacy figure; not used.** |
| `venv/` | Local Windows virtual environment containing installed dependencies. |

## Reproduction and demonstration

Run commands from the repository root on Windows PowerShell.

### Recommended professor demonstration

The boundary validation is the best single end-to-end demonstration because it loads the frozen PPO and tree, generates genuinely new states, executes real episodes, prints metrics/examples, and saves a plot:

```powershell
.\venv\Scripts\python.exe validate_boundary.py
```

Expected behavior:

- Selects 200 states from 100,000 seeded candidates.
- Prints progress every 25 episodes.
- Takes roughly one minute on the current machine.
- Rewrites only `data/boundary_validation.csv` and `boundary_validation.png`.
- Prints the same metrics recorded above when run with the included environment and artifacts.

Exact reproducibility was verified by two complete consecutive runs: both the CSV and PNG were byte-identical.

### Regenerate current downstream artifacts from the saved PPO

```powershell
.\venv\Scripts\python.exe rollout.py
.\venv\Scripts\python.exe controlled_rollout.py
.\venv\Scripts\python.exe train_surrogate.py
.\venv\Scripts\python.exe visualize.py
.\venv\Scripts\python.exe validate_boundary.py
```

`rollout.py` is the longest stage because it records 1.5 million nominal state rows. `controlled_rollout.py` and `validate_boundary.py` also execute real environment episodes. `train_surrogate.py` and `visualize.py` are fast.

Useful CLI options:

```powershell
python rollout.py --episodes 3000 --seed 42 --checkpoint models/ppo_cartpole.zip --output data/rollouts.csv
python controlled_rollout.py --levels 5 --seed 42 --checkpoint models/ppo_cartpole.zip --output data/controlled_rollouts.csv
python validate_boundary.py --seed 42 --pool-size 100000 --checkpoint models/ppo_cartpole.zip --tree models/tree_outcome.joblib --controlled-data data/controlled_rollouts.csv --output data/boundary_validation.csv --plot boundary_validation.png
```

### Retraining PPO

Only `train.py` requires CUDA:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\venv\Scripts\python.exe train.py
```

Retraining will overwrite both PPO checkpoints and will not reproduce the current weights because no training seed is set. If retraining is performed, every rollout, surrogate, figure, validation file, result, and hash below must be regenerated.

## Software environment

`requirements.txt` declares:

- `gymnasium[classic-control]`
- `pandas`
- `stable-baselines3`
- CUDA 12.6 PyTorch wheel index
- `torch==2.8.0+cu126`
- `scikit-learn`
- `joblib`
- `matplotlib`

Versions reported by the included `venv` at the time this context was updated:

| Package | Version |
| --- | --- |
| Python | 3.13.3 |
| Gymnasium | 1.3.0 |
| pandas | 3.0.5 |
| NumPy | 2.5.3 |
| PyTorch | 2.14.0+cu126 |
| scikit-learn | 1.9.1 |
| Stable-Baselines3 | 2.9.0 |
| Matplotlib | 3.11.2 |
| joblib | 1.6.0 |

There is a version discrepancy between the PyTorch pin in `requirements.txt` (`2.8.0+cu126`) and the currently installed environment (`2.14.0+cu126`). The included environment produced the artifacts documented here. Most other dependencies are unpinned, so a fresh install may resolve newer versions.

## Current artifact checksums

SHA-256 checksums provide a compact way to confirm that demonstrations use the documented artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `models/ppo_cartpole.zip` | `AE75801872039714F907B9AD941A1F1DF58D02637654179089512165373E7258` |
| `models/ppo_cartpole_partial.zip` | `1A8E894BFC266FFA714FA6A755F3CE8776E434B4E817CD631ED8B9227AB502AD` |
| `models/tree_outcome.joblib` | `534A38B20F1F19ADF4D9D5A0C213F4021537CB301650D5CA9EF266BE23CAAF17` |
| `data/rollouts.csv` | `6FA6678736C50FDE703AB30702EBF341E9375C490F23A21AACF869338CB603C9` |
| `data/controlled_rollouts.csv` | `EF914648F87E760AD183030A36683469DDB845238B4AC1C06A47063865847B6D` |
| `data/boundary_validation.csv` | `3A70274EF39FBA4DC621FFFB44D547C1316958F12C974246F2E1F737EE47A7ED` |
| `failure_envelope.png` | `8BE0D8A9252B85F6F98723CCC96E0F05B71B32386F58D68DE5886B768C96ACBE` |
| `failure_envelope_cart.png` | `A683D55E96B0890C25B07407D4F48F4773E18D54E9D32C9416B576DCC7FB2C54` |
| `boundary_validation.png` | `3C01D5AA69D345B77A7EF179D8B8818DABD9B1100D4077F9AA8779E31CC86224` |

## Interpretation boundaries and known limitations

- The final PPO is robust under the tested standard reset distribution, not proven robust for every state.
- The controlled grid is deliberately broader than default reset conditions but covers only a finite rectangular range and five levels per dimension.
- Direct initial-state assignment preserves dynamics and termination behavior but evaluates states that the policy may rarely or never encounter under nominal resets.
- The Decision Tree approximates trajectory outcome from initial state only. It does not explain individual PPO actions or intermediate-state causal effects.
- The surrogate test split is episode-disjoint and has one row per episode, but all examples come from the same structured 625-point grid.
- The tree uses class weighting. Its `predict_proba` output is an uncalibrated class-weighted leaf score.
- The two failure-envelope figures are exact 2D slices with other coordinates fixed at zero; they are not marginal views of all four dimensions.
- Boundary validation states are genuinely new and selected without observing their outcomes, but they are score-targeted rather than representative of a natural state distribution.
- Near-boundary agreement of 58.33% demonstrates that the learned boundary is approximate and unstable in its most uncertain regions.
- PPO training itself is not seeded, although every current post-training evaluation is seeded and deterministic.
- There is no automated test suite. Current validation consists of schema/invariant audits, deterministic reruns, checksum comparisons, and direct inspection of generated figures.
- Legacy near-failure files remain in the workspace only for history. They are not part of the current analysis.

The defensible final claim is: **within the specified controlled initial-state range, a small Decision Tree captures meaningful high-level structure in the frozen PPO policy's successes and failures, especially in clear-score regions, but it is only an interpretable approximation and is notably less reliable near its learned boundary.**
