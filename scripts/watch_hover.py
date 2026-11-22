"""Render the trained hover policy to out/hover.gif: six drones fly in from random starts."""

import pickle
from pathlib import Path

import jax
import numpy as np

from quadjax import hover, ppo, render, sim

N = 6
OUT = Path("out")

with open(OUT / "hover_params.pkl", "rb") as f:
    params: ppo.Params = pickle.load(f)


def policy_step(s: hover.EnvState, obs: jax.Array) -> tuple[hover.EnvState, jax.Array]:
    s, obs, *_ = jax.vmap(hover.step)(s, ppo.mlp(params.actor, obs))
    return s, obs


state, obs = jax.vmap(hover.reset)(jax.random.split(jax.random.key(7), N))
step = jax.jit(policy_step)
target = np.asarray(hover.TARGET)
scene = render.Scene(N, target[None], lookat=target, distance=3.2)

frames = []
for t in range(hover.MAX_STEPS):
    if t % 2 == 0:  # policy runs at 50 Hz, the GIF at 25 fps
        frames.append(
            scene.frame(np.asarray(state.drone.pos), np.asarray(state.drone.quat))
        )
    state, obs = step(state, obs)

render.save_gif(frames, OUT / "hover.gif")
print(
    f"saved {OUT}/hover.gif ({len(frames)} frames, {hover.MAX_STEPS * hover.SUBSTEPS * sim.DT:.0f} s)"
)
