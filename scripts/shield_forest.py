"""Fly the forest policy with and without the safety filter (quadjax.shield) through 1000 new
forests of a few densities, and compare crash rates.

Usage: shield_forest.py [forest params], default out/forest_params.pkl. Needs
out/waypoint_params.pkl too: the filter's brake is the waypoint policy.
Reached = alive and within 0.5 m of the goal at the end. Alive but not there means the
filter kept braking and the drone ran out of time.
"""

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from quadjax import forest, ppo, shield

PARAMS = Path(sys.argv[1] if len(sys.argv) > 1 else "out/forest_params.pkl")
N = 1000
KINDS = [("as trained", forest.TREES), ("dense (45)", 45), ("very dense (60)", 60)]

with open(PARAMS, "rb") as f:
    policy = pickle.load(f).actor
with open("out/waypoint_params.pkl", "rb") as f:
    backup = pickle.load(f).actor


def evaluate(trees: int, filtered: str) -> tuple[float, float, float]:
    """Crashed, reached, and share of steps the filter braked. filtered is "off",
    "nominal" (the check simulates the nominal drone) or "true body" (it simulates the
    exact drone flying: the best a check could do)."""
    keys = jax.random.split(jax.random.key(1), N)
    s, obs = jax.vmap(
        lambda k: forest.start(k, forest.random_trees(jax.random.fold_in(k, 1), trees))
    )(keys)
    mode = jax.vmap(lambda _: shield.idle())(keys)

    def act(
        s: forest.EnvState, obs: jax.Array, mode: shield.Mode
    ) -> tuple[jax.Array, shield.Mode]:
        a = ppo.mlp(policy, obs)
        match filtered:
            case "nominal":
                return shield.filter_action(backup, s, a, mode)
            case "true body":
                return shield.filter_action(backup, s, a, mode, s.body)
            case _:
                return a, mode

    def go(
        c: tuple[forest.EnvState, jax.Array, shield.Mode, jax.Array], _step: None
    ) -> tuple[tuple[forest.EnvState, jax.Array, shield.Mode, jax.Array], jax.Array]:
        s, obs, mode, dead = c
        a, mode = jax.vmap(act)(s, obs, mode)
        s2, obs, _, crashed, _ = jax.vmap(forest.step)(s, a)
        return (s2, obs, mode, dead | crashed), mode.braking & ~dead

    (s, _, _, dead), braked = jax.lax.scan(
        go, (s, obs, mode, jnp.zeros(N, bool)), None, forest.MAX_STEPS
    )
    reached = ~dead & (jnp.linalg.norm(s.drone.pos - forest.GOAL, axis=1) < 0.5)
    return float(dead.mean()), float(reached.mean()), float(braked.mean())


print(f"{PARAMS}, {N} new forests per row, margin {shield.MARGIN} m")
print(f"{'forest':18s} {'filter':10s} crashed  reached  steps braked")
for name, trees in KINDS:
    for filtered in ("off", "nominal", "true body"):
        crashed, reached, braked = evaluate(trees, filtered)
        print(
            f"{name:18s} {filtered:10s} {crashed * 100:5.1f}%  "
            f"{reached * 100:5.1f}%   {braked * 100:5.1f}%",
            flush=True,
        )
