# CartPole Failure-Envelope Report

## Dataset Summary

- Episodes: 1,000
- Fail rows: 130,828
- Success rows: 1,500
- Near-failure rows: 10,965

## Surrogate Performance

| Target | Test accuracy | Test F1 |
| --- | ---: | ---: |
| outcome | 0.8470 | 0.9072 |
| near_failure | 0.8597 | 0.8833 |

## Top-Level Rules

The outcome tree first splits on `cart_pos` (`state_0`) at approximately `0.13`. States with cart position at or below this threshold enter the left branch; states above it enter the right branch and are further separated by pole angle and cart velocity.

The near-failure tree first splits on `pole_angle` (`state_2`) at approximately `-0.11`. States with pole angle at or below this threshold enter the left branch, while larger angles enter the right branch and are further separated mainly by pole velocity and cart position.

## Interpretation

The unsafe region for this policy is concentrated around specific combinations of cart position and pole angle rather than a single state variable. Outcome predictions are dominated by cart position, with pole angle providing the next strongest signal. Near-failure predictions are dominated by pole angle and pole velocity, indicating that the policy becomes vulnerable when the pole is tilted while its angular motion is also unfavorable. The envelope plots show these learned regions while holding the other two state features at their dataset means, so they are useful projections rather than a complete four-dimensional boundary.