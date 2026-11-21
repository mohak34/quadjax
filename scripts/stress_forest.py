"""Fly the trained forest policy through forests unlike the ones it trained on: more or fewer
trunks, thinner or thicker trunks. 1000 new forests per row.

Usage: stress_forest.py [params file], default out/forest_params.pkl.
Prints crash rate and how many got within 0.5 m of the goal by the end, per kind of forest.
"""

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from volatus import forest, ppo

PARAMS = Path(sys.argv[1] if len(sys.argv) > 1 else "out/forest_params.pkl")
N = 1000
KINDS = [
    ("as trained", forest.TREES, forest.TRUNK_RADIUS),
    ("sparse (15 trunks)", 15, forest.TRUNK_RADIUS),
    ("dense (45 trunks)", 45, forest.TRUNK_RADIUS),
    ("very dense (60 trunks)", 60, forest.TRUNK_RADIUS),
    ("thin trunks (4-8 cm)", forest.TREES, (0.04, 0.08)),
    ("thick trunks (25-40 cm)", forest.TREES, (0.25, 0.4)),
]

with open(PARAMS, "rb") as f:
    params: ppo.Params = pickle.load(f)


def evaluate(trees: int, radius: tuple[float, float]) -> tuple[float, float]:
    keys = jax.random.split(jax.random.key(1), N)
    s, obs = jax.vmap(
        lambda k: forest.start(
            k, forest.random_trees(jax.random.fold_in(k, 1), trees, radius)
        )
    )(keys)

    def go(c: tuple[forest.EnvState, jax.Array, jax.Array], _step: None):
        s, obs, dead = c
        s2, obs, _, crashed, _ = jax.vmap(forest.step)(s, ppo.mlp(params.actor, obs))
        return (s2, obs, dead | crashed), None

    (s, _, dead), _ = jax.lax.scan(
        go, (s, obs, jnp.zeros(N, bool)), None, forest.MAX_STEPS
    )
    reached = ~dead & (jnp.linalg.norm(s.drone.pos - forest.GOAL, axis=1) < 0.5)
    return float(dead.mean()), float(reached.mean())


print(f"{PARAMS}, {N} new forests per row")
print(f"{'forest':26s} crashed   reached goal")
for name, trees, radius in KINDS:
    crashed, reached = evaluate(trees, radius)
    print(f"{name:26s} {crashed * 100:5.1f}%   {reached * 100:5.1f}%")
