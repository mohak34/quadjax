"""Fly our sim and MuJoCo with identical forces and plot how far apart they end up.

Both get the same total thrust and body torques every step, so any gap comes from
the physics and integration, not from the motor model.

mujoco is compiled C++ with no type stubs, so ty cannot see its members; those lines carry ty ignores.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from volatus import sim

SECONDS = 3.0
OUT = Path("out/compare.png")

# Same body as our sim. One actuator per force: thrust along body z, then torque about x, y, z.
XML = f"""
<mujoco>
  <option timestep="{sim.DT}" integrator="Euler"/>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <inertial pos="0 0 0" mass="{sim.MASS}" diaginertia="{" ".join(str(float(i)) for i in sim.INERTIA)}"/>
      <site name="center"/>
    </body>
  </worldbody>
  <actuator>
    <motor site="center" gear="0 0 1 0 0 0"/>
    <motor site="center" gear="0 0 0 1 0 0"/>
    <motor site="center" gear="0 0 0 0 1 0"/>
    <motor site="center" gear="0 0 0 0 0 1"/>
  </actuator>
</mujoco>
"""


def wobble_commands(steps: int) -> jax.Array:
    """Each motor swings +-4% around hover at its own rate, so the drone tilts and turns on every axis."""
    t = jnp.arange(steps)[:, None] * sim.DT
    freq = jnp.array([1.0, 1.3, 1.7, 2.1])
    phase = jnp.array([0.0, 1.0, 2.0, 3.0])
    return sim.HOVER_THRUST * (1 + 0.04 * jnp.sin(2 * jnp.pi * freq * t + phase))


steps = round(SECONDS / sim.DT)
_, ours = sim.rollout(
    sim.hover_state(jnp.array([0.0, 0.0, 1.0])), wobble_commands(steps)
)
# Forces our sim actually applied each step (after motor lag), as thrust + 3 torques.
total, torque = jax.vmap(sim.mix)(ours.thrust)
forces = np.asarray(jnp.concatenate([total[:, None], torque], axis=1))

model = mujoco.MjModel.from_xml_string(XML)  # ty: ignore[missing-attribute]
data = mujoco.MjData(model)  # ty: ignore[missing-attribute]
mj_pos = np.zeros((steps, 3))
mj_quat = np.zeros((steps, 4))
for i in range(steps):
    data.ctrl[:] = forces[i]
    mujoco.mj_step(model, data)  # ty: ignore[missing-attribute]
    mj_pos[i] = data.qpos[:3]
    mj_quat[i] = data.qpos[3:7]

pos = np.asarray(ours.pos)
pos_gap = np.linalg.norm(pos - mj_pos, axis=1)
# Angle of the rotation taking one orientation to the other: relative quaternion conj(a) * b,
# angle = 2 atan2(|vector part|, |w|). arccos(a . b) would lose tiny angles to rounding.
a = np.asarray(ours.quat, dtype=np.float64)
rel_w = (a * mj_quat).sum(axis=1)
rel_v = (
    a[:, :1] * mj_quat[:, 1:]
    - mj_quat[:, :1] * a[:, 1:]
    - np.cross(a[:, 1:], mj_quat[:, 1:])
)
att_gap = np.degrees(2 * np.arctan2(np.linalg.norm(rel_v, axis=1), np.abs(rel_w)))

print(
    f"after {SECONDS:.0f} s: position gap {pos_gap[-1] * 1000:.3f} mm, attitude gap {att_gap[-1]:.4f} deg"
)
print(f"ours   ends at {pos[-1].round(4)}")
print(f"mujoco ends at {mj_pos[-1].round(4)}")

# Plot: our sim solid blue, MuJoCo dashed orange, gaps on log scale below.
OURS, MUJOCO, INK, MUTED = "#3987e5", "#d95926", "#f0efec", "#55554f"
plt.rcParams.update(
    {
        "figure.facecolor": "#000",
        "axes.facecolor": "#000",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#222",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
t = np.arange(1, steps + 1) * sim.DT
fig, ax = plt.subplot_mosaic("xyz;ppa", figsize=(13, 7), layout="constrained")
for k, axis in enumerate("xyz"):
    ax[axis].plot(t, pos[:, k], color=OURS, lw=2, label="our sim")
    ax[axis].plot(t, mj_pos[:, k], color=MUJOCO, lw=2, ls=(0, (4, 3)), label="MuJoCo")
    ax[axis].set_title(f"{axis} (m)", loc="left")
ax["x"].legend(frameon=False)
ax["p"].semilogy(t, np.maximum(pos_gap * 1000, 1e-9), color=INK, lw=2)
ax["p"].set_title("position gap (mm)", loc="left")
ax["a"].semilogy(t, np.maximum(att_gap, 1e-9), color=INK, lw=2)
ax["a"].set_title("attitude gap (deg)", loc="left")
for a in ("p", "a"):
    ax[a].set_xlabel("time (s)")
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=110)
print(f"plot saved to {OUT}")
