"""Render the trained hover policy to out/hover.gif: six drones fly in from random starts.

Our sim does the physics; MuJoCo only draws the drones at the positions our sim computes.
mujoco is compiled C++ with no type stubs, so ty cannot see its members; those lines carry ty ignores.
"""

import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # draw on the GPU without opening a window

import jax
import mujoco
import numpy as np
from PIL import Image

from volatus import hover, ppo, sim

N = 6
OUT = Path("out")
COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]


def rgba(hex_color: str) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def drone_xml(i: int) -> str:
    # Drawn at about 3x real size so it shows up from the camera distance.
    return f"""
    <body name="d{i}" mocap="true">
      <geom type="box" size="0.14 0.012 0.008" euler="0 0 45" rgba="{rgba(COLORS[i])}"/>
      <geom type="box" size="0.14 0.012 0.008" euler="0 0 -45" rgba="{rgba(COLORS[i])}"/>
      <geom type="box" size="0.012 0.012 0.008" pos="0.1 0 0.012" rgba="1 1 1 1"/>
      <geom type="cylinder" size="0.04 0.003" pos="0.1 0.1 0.012" rgba=".8 .8 .8 .6"/>
      <geom type="cylinder" size="0.04 0.003" pos="0.1 -0.1 0.012" rgba=".8 .8 .8 .6"/>
      <geom type="cylinder" size="0.04 0.003" pos="-0.1 0.1 0.012" rgba=".8 .8 .8 .6"/>
      <geom type="cylinder" size="0.04 0.003" pos="-0.1 -0.1 0.012" rgba=".8 .8 .8 .6"/>
    </body>"""


XML = f"""
<mujoco>
  <visual><global offwidth="720" offheight="480"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".15 .15 .15" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom type="plane" size="4 4 .1" material="grid"/>
    <site pos="{" ".join(map(str, np.asarray(hover.TARGET)))}" type="sphere" size="0.05" rgba="1 1 1 .35"/>
    {"".join(drone_xml(i) for i in range(N))}
  </worldbody>
</mujoco>
"""

with open(OUT / "hover_params.pkl", "rb") as f:
    params: ppo.Params = pickle.load(f)


def policy_step(s: hover.EnvState, obs: jax.Array) -> tuple[hover.EnvState, jax.Array]:
    s, obs, *_ = jax.vmap(hover.step)(s, ppo.mlp(params.actor, obs))
    return s, obs


state, obs = jax.vmap(hover.reset)(jax.random.split(jax.random.key(7), N))
step = jax.jit(policy_step)

model = mujoco.MjModel.from_xml_string(XML)  # ty: ignore[missing-attribute]
data = mujoco.MjData(model)  # ty: ignore[missing-attribute]
renderer = mujoco.Renderer(model, 480, 720)
camera = mujoco.MjvCamera()  # ty: ignore[missing-attribute]
camera.lookat[:] = np.asarray(hover.TARGET)
camera.distance, camera.azimuth, camera.elevation = 3.2, 120, -20

frames = []
for t in range(hover.MAX_STEPS):
    if t % 2 == 0:  # policy runs at 50 Hz, the GIF at 25 fps
        data.mocap_pos[:] = np.asarray(state.drone.pos)
        data.mocap_quat[:] = np.asarray(state.drone.quat)
        mujoco.mj_forward(model, data)  # ty: ignore[missing-attribute]
        renderer.update_scene(data, camera)
        frames.append(Image.fromarray(renderer.render()))
    state, obs = step(state, obs)

frames[0].save(
    OUT / "hover.gif", save_all=True, append_images=frames[1:], duration=40, loop=0
)
print(
    f"saved {OUT}/hover.gif ({len(frames)} frames, {hover.MAX_STEPS * hover.SUBSTEPS * sim.DT:.0f} s)"
)
