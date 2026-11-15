"""Draw drones from our sim with MuJoCo's offscreen renderer and save GIFs.

MuJoCo only draws here; our sim does the physics. mujoco is compiled C++ with no
type stubs, so ty cannot see some members; those lines carry ty ignores.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # draw on the GPU without opening a window

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
WIDTH, HEIGHT = 720, 480


def _rgba(hex_color: str, alpha: float = 1.0) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return f"{r:.3f} {g:.3f} {b:.3f} {alpha}"


def _drone_xml(i: int) -> str:
    # Drawn at about 3x real size so it shows up from the camera distance.
    c = _rgba(COLORS[i])
    props = "".join(
        f'<geom type="cylinder" size="0.04 0.003" pos="{x} {y} 0.012" rgba=".8 .8 .8 .6"/>'
        for x in (0.1, -0.1)
        for y in (0.1, -0.1)
    )
    return f"""
    <body name="d{i}" mocap="true">
      <geom type="box" size="0.14 0.012 0.008" euler="0 0 45" rgba="{c}"/>
      <geom type="box" size="0.14 0.012 0.008" euler="0 0 -45" rgba="{c}"/>
      <geom type="box" size="0.012 0.012 0.008" pos="0.1 0 0.012" rgba="1 1 1 1"/>
      {props}
    </body>"""


class Scene:
    """A floor, n drones and fixed marker balls (targets). Call frame() once per picture."""

    def __init__(
        self, n: int, markers: np.ndarray, lookat: np.ndarray, distance: float
    ) -> None:
        balls = "".join(
            f'<site pos="{x} {y} {z}" type="sphere" size="0.05" rgba="1 1 1 .35"/>'
            for x, y, z in markers
        )
        xml = f"""
        <mujoco>
          <visual><global offwidth="{WIDTH}" offheight="{HEIGHT}"/></visual>
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .1 .1" rgb2=".15 .15 .15"
                     width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8"/>
          </asset>
          <worldbody>
            <light pos="0 0 5" dir="0 0 -1" diffuse=".8 .8 .8"/>
            <geom type="plane" size="4 4 .1" material="grid"/>
            {balls}
            {"".join(_drone_xml(i) for i in range(n))}
          </worldbody>
        </mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)  # ty: ignore[missing-attribute]
        self.data = mujoco.MjData(self.model)  # ty: ignore[missing-attribute]
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.camera = mujoco.MjvCamera()  # ty: ignore[missing-attribute]
        self.camera.lookat[:] = lookat
        self.camera.distance, self.camera.azimuth, self.camera.elevation = (
            distance,
            120,
            -20,
        )

    def frame(self, pos: np.ndarray, quat: np.ndarray) -> Image.Image:
        """pos (n, 3) and quat (n, 4, w first) of every drone."""
        self.data.mocap_pos[:] = pos
        self.data.mocap_quat[:] = quat
        mujoco.mj_forward(self.model, self.data)  # ty: ignore[missing-attribute]
        self.renderer.update_scene(self.data, self.camera)
        return Image.fromarray(self.renderer.render())


def save_gif(frames: list[Image.Image], path: Path, fps: int = 25) -> None:
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=1000 // fps, loop=0
    )
