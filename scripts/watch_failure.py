"""Four drones hold a point with the trained failure policy. At 1 s one motor on each dies:
a different motor for each drone. Each drone flies its own copy of the task; they are drawn
side by side so they do not overlap.

Prints how fast each one ends up spinning and how far from its point it ends (or when it
crashed: fell to the ground or drifted more than 3 m away), and saves
out/failure.gif.
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import ImageDraw, ImageFont

from volatus import failure, ppo, render

OUT = Path("out")
N = 4
NAMES = ["blue", "orange", "green", "yellow"]  # render.COLORS
MOTORS = ["front-right", "back-right", "back-left", "front-left"]  # sim motor order
FAIL_AT = failure.HZ  # 1 s
SPREAD = np.array([[x, 0.0, 0.0] for x in (-1.5, -0.5, 0.5, 1.5)])  # drawing offsets

with open(OUT / "failure_params.pkl", "rb") as f:
    params: ppo.Params = pickle.load(f)

state, _ = jax.vmap(failure.reset)(jax.random.split(jax.random.key(3), N))
state = state._replace(
    motor=jnp.arange(N),
    left=jnp.zeros(N),
    fail_at=jnp.full(N, FAIL_AT),
)


@jax.jit
def policy_step(
    s: failure.EnvState, dead: jax.Array
) -> tuple[failure.EnvState, jax.Array]:
    """A crashed drone stays where it crashed."""
    obs = jax.vmap(failure.observe)(s)
    s2, _, _, crashed, _ = jax.vmap(failure.step)(s, ppo.mlp(params.actor, obs))
    s2 = jax.tree.map(
        lambda new, old: jnp.where(dead.reshape(-1, *[1] * (new.ndim - 1)), old, new),
        s2,
        s,
    )
    return s2, dead | crashed


target = np.asarray(failure.TARGET)
scene = render.Scene(N, target + SPREAD, lookat=target, distance=5.0)
font = ImageFont.load_default(size=22)
frames = []
yaw = []
dead = jnp.zeros(N, bool)
crash_time = [None] * N
for t in range(failure.MAX_STEPS):
    state, dead_now = policy_step(state, dead)
    for i in np.nonzero(np.asarray(dead_now & ~dead))[0]:
        crash_time[i] = (t + 1) / failure.HZ
    dead = dead_now
    yaw.append(np.abs(np.asarray(state.drone.omega[:, 2])))
    if t % (failure.HZ // 25):
        continue  # the GIF runs at 25 fps
    img = scene.frame(
        np.asarray(state.drone.pos) + SPREAD, np.asarray(state.drone.quat)
    )
    draw = ImageDraw.Draw(img)
    draw.text(
        (20, 16), f"t = {(t + 1) / failure.HZ:.1f} s", fill=(240, 239, 236), font=font
    )
    if t + 1 >= FAIL_AT:
        draw.text((20, 46), "one motor dead on each", fill=(201, 133, 0), font=font)
    frames.append(img)

spin = np.array(yaw)[-failure.HZ :].mean(axis=0)  # last second
miss = np.linalg.norm(np.asarray(state.drone.pos) - target, axis=1)
for i in range(N):
    head = f"{NAMES[i]} ({MOTORS[i]} motor dead)"
    if crash_time[i]:
        print(f"{head}: crashed at {crash_time[i]:.1f} s")
        continue
    print(
        f"{head}: spins at {spin[i]:.0f} rad/s ({spin[i] / (2 * np.pi):.1f} turns/s), "
        f"ends {miss[i] * 100:.0f} cm from its point"
    )
render.save_gif(frames, OUT / "failure.gif")
print(f"saved {OUT}/failure.gif ({len(frames)} frames)")
