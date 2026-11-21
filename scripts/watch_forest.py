"""Six drones with random bodies fly the same new forest with the trained forest policy.
A crashed drone stays where it crashed. The inset is what the blue drone's depth camera sees.

Prints each drone's outcome and saves out/forest.gif.
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from volatus import depth, forest, ppo, render

OUT = Path("out")
N = 6
NAMES = ["blue", "orange", "green", "yellow", "pink", "purple"]  # render.COLORS
SEED = 2

with open(OUT / "forest_params.pkl", "rb") as f:
    params: ppo.Params = pickle.load(f)

key = jax.random.key(SEED)
trees = forest.random_trees(key)  # everyone flies this one forest
state, _ = jax.vmap(lambda k: forest.start(k, trees))(jax.random.split(key, N))


@jax.jit
def policy_step(
    s: forest.EnvState, dead: jax.Array
) -> tuple[forest.EnvState, jax.Array, jax.Array]:
    obs = jax.vmap(forest.observe)(s)
    s2, _, _, crashed, _ = jax.vmap(forest.step)(s, ppo.mlp(params.actor, obs))
    s2 = jax.tree.map(
        lambda new, old: jnp.where(dead.reshape(-1, *[1] * (new.ndim - 1)), old, new),
        s2,
        s,
    )
    return s2, dead | crashed, obs[0, 18 : 18 + depth.PIXELS]  # blue's newest frame


goal = np.asarray(forest.GOAL)
scene = render.Scene(
    N,
    goal[None],
    lookat=np.zeros(3),
    distance=6.0,
    trees=np.asarray(state.trees[0]),
    floor=30.0,
)
# Nearly straight down: from the side the trunks hide the drones.
scene.camera.distance, scene.camera.azimuth, scene.camera.elevation = 11.0, 90, -75
font = ImageFont.load_default(size=22)
dead = jnp.zeros(N, bool)
crash_time = [None] * N
frames = []
for t in range(forest.MAX_STEPS):
    state, dead_now, seen = policy_step(state, dead)
    for i in np.nonzero(np.asarray(dead_now & ~dead))[0]:
        crash_time[i] = (t + 1) / 50
    dead = dead_now
    if t % 2:
        continue  # policy runs at 50 Hz, the GIF at 25 fps
    pos = np.array(state.drone.pos)
    pos[:, 2] = np.maximum(pos[:, 2], 0.0)  # draw ground hits on the floor
    scene.camera.lookat[:] = [min(pos[:, 0].mean(), goal[0]), 0.0, 1.0]
    img = scene.frame(pos, np.asarray(state.drone.quat))

    # Depth inset: bright = close, black = nothing within MAX_DEPTH.
    gray = (255 * (1 - np.asarray(seen))).reshape(depth.HEIGHT, depth.WIDTH)
    inset = Image.fromarray(gray.astype(np.uint8)).resize(
        (256, 64), Image.Resampling.NEAREST
    )
    img.paste(inset.convert("RGB"), (render.WIDTH - 276, render.HEIGHT - 84))
    draw = ImageDraw.Draw(img)
    draw.text(
        (render.WIDTH - 276, render.HEIGHT - 112),
        "blue drone's depth camera",
        fill=(240, 239, 236),
        font=font,
    )
    draw.text((20, 16), f"t = {(t + 1) / 50:.1f} s", fill=(240, 239, 236), font=font)
    draw.text(
        (20, 46), f"crashed: {int(dead.sum())} of {N}", fill=(201, 133, 0), font=font
    )
    frames.append(img)

final = np.asarray(state.drone.pos)
for i in range(N):
    end = f"crashed at {crash_time[i]:.1f} s" if crash_time[i] else "made it"
    miss = np.linalg.norm(final[i] - goal)
    print(f"{NAMES[i]}: {end}, ends {miss * 100:.0f} cm from the goal")
render.save_gif(frames, OUT / "forest.gif")
print(f"saved {OUT}/forest.gif ({len(frames)} frames)")
