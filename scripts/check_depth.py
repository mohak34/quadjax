"""Check our depth camera against MuJoCo's ray caster on the same trees, then save
out/depth.png: a top-down map of what the camera's middle row sees, next to the full image."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from volatus import depth, sim

OUT = Path("out")
k_xy, k_r, k_pose = jax.random.split(jax.random.key(0), 3)
trees = jnp.concatenate(
    [
        jax.random.uniform(
            k_xy, (30, 2), minval=jnp.array([1.0, -4]), maxval=jnp.array([10.0, 4])
        ),
        jax.random.uniform(k_r, (30, 1), minval=0.1, maxval=0.25),
    ],
    axis=1,
)

# Same trees in MuJoCo: cylinders 100 m tall standing on a ground plane.
cyl = "".join(
    f'<geom type="cylinder" size="{r} 50" pos="{x} {y} 0"/>'
    for x, y, r in np.asarray(trees)
)
xml = (
    f'<mujoco><worldbody><geom type="plane" size="50 50 .1"/>{cyl}</worldbody></mujoco>'
)
model = mujoco.MjModel.from_xml_string(xml)  # ty: ignore[missing-attribute]
data = mujoco.MjData(model)  # ty: ignore[missing-attribute]
mujoco.mj_forward(model, data)  # ty: ignore[missing-attribute]
geom = np.zeros(1, np.int32)


def mujoco_image(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    dirs = (
        np.asarray(depth.RAYS, np.float64)
        @ np.asarray(sim.rotation(jnp.asarray(quat)), np.float64).T
    )
    hits = [mujoco.mj_ray(model, data, pos, d, None, True, -1, geom) for d in dirs]  # ty: ignore[missing-attribute]
    return np.array(
        [depth.MAX_DEPTH if h < 0 else min(h, depth.MAX_DEPTH) for h in hits]
    )


# 50 random poses: anywhere in x from 0 to 4, tilted up to 40 degrees, any heading.
worst = 0.0
keys = jax.random.split(k_pose, 50)
for k in keys:
    k_pos, k_axis, k_angle = jax.random.split(k, 3)
    pos = jax.random.uniform(
        k_pos, (3,), minval=jnp.array([0.0, -3, 0.3]), maxval=jnp.array([4.0, 3, 2])
    )
    if jnp.any(jnp.linalg.norm(pos[:2] - trees[:, :2], axis=1) < trees[:, 2]):
        continue  # started inside a trunk
    axis = jax.random.normal(k_axis, (3,))
    angle = jax.random.uniform(k_angle, maxval=jnp.radians(40))
    quat = jnp.concatenate(
        [jnp.cos(angle / 2)[None], jnp.sin(angle / 2) * axis / jnp.linalg.norm(axis)]
    )
    ours = np.asarray(depth.image(pos, quat, trees))
    worst = max(
        worst,
        float(
            np.abs(
                ours - mujoco_image(np.asarray(pos, np.float64), np.asarray(quat))
            ).max()
        ),
    )
print(
    f"{len(keys)} poses x {depth.PIXELS} rays: largest gap between ours and MuJoCo {worst * 1000:.4f} mm"
)

# Picture: level drone at (0, 0, 1) looking down +x.
pos, quat = jnp.array([0.0, 0, 1]), jnp.array([1.0, 0, 0, 0])
img = np.asarray(depth.image(pos, quat, trees)).reshape(depth.HEIGHT, depth.WIDTH)
middle = depth.HEIGHT // 2 - 1  # the row just above the horizon, ~level with the drone
rays = np.asarray(depth.RAYS).reshape(depth.HEIGHT, depth.WIDTH, 3)[middle]

plt.rcParams.update(
    {
        "figure.facecolor": "#000",
        "axes.facecolor": "#000",
        "axes.edgecolor": "#55554f",
        "text.color": "#f0efec",
        "axes.labelcolor": "#f0efec",
        "xtick.color": "#f0efec",
        "ytick.color": "#f0efec",
    }
)
fig, ax = plt.subplots(
    1, 2, figsize=(12, 5), layout="constrained", width_ratios=[1, 1.4]
)
for x, y, r in np.asarray(trees):
    ax[0].add_patch(plt.Circle((x, y), r, color="#199e70"))
for d, dist in zip(rays, img[middle]):
    hit = dist < depth.MAX_DEPTH
    ax[0].plot(
        [0, d[0] * dist], [0, d[1] * dist], color="#c98500" if hit else "#55554f", lw=1
    )
ax[0].plot(0, 0, "o", color="#3987e5")
ax[0].set_aspect("equal")
ax[0].set_xlim(-0.5, 10)
ax[0].set_ylim(-5, 5)
ax[0].set_title("top view: middle row of rays (orange = hit a trunk)", loc="left")
ax[0].set_xlabel("x, m (forward)")
ax[0].set_ylabel("y, m (left)")
shown = ax[1].imshow(img, cmap="magma_r", vmin=0, vmax=depth.MAX_DEPTH)
ax[1].set_title(
    f"what the policy sees: {depth.WIDTH} x {depth.HEIGHT} depth image", loc="left"
)
ax[1].axhline(middle, color="#3987e5", lw=1, ls=(0, (4, 3)))
fig.colorbar(shown, ax=ax[1], label="distance, m (bright = close)", shrink=0.6)
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "depth.png", dpi=110)
print(f"saved {OUT}/depth.png")
