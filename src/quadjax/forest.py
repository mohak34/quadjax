"""Forest task: fly from x = 0 to a goal past a random patch of tree trunks, seeing the trees
only through the depth camera. Touching a trunk, the ground, the ceiling or a side wall is a
crash. Every episode has a new forest and a new randomized drone body.

Reward is progress (how much closer to the goal this step got, counted only up to
CRUISE_SPEED so going faster earns nothing but risk) plus the waypoint reward
exp(-distance), which is ~0 in the forest and up to 1 per step for holding at the goal.
A crash ends the episode and forfeits everything still to come; that is the only push to
avoid trees. Progress alone is not enough: with nothing left to earn at the goal, a crash
there is free, and the policy learns to dive into the ground at the finish at full speed.
Without the speed cap it learns to fly at ~10 m/s and can no longer brake at the goal.

Every reading is noisy, like on a real drone (numbers under "Sensor noise" below), so the
policy cannot learn to trust any single pixel or measurement exactly.

The policy sees the last FRAMES depth images, newest first. More than one would let it
see trunks move across the view and remember one it is passing after it leaves the view,
but 3 frames crashed more than 1 at the same training budget (bigger input, same data).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from quadjax import depth, ppo, sim
from quadjax.waypoint import random_body

SUBSTEPS = 10  # 500 Hz physics, 50 Hz policy
MAX_STEPS = 400  # 8 s
CONFIG = ppo.Config(total_steps=300_000_000, hidden=128)
CHECKS = [(f"after {sec} s", sec * 50 - 1) for sec in (2, 4, 8)]

TREES = 30
FOREST_X = (2.0, 14.0)  # m, trunks stand between these x
FOREST_Y = 4.5  # m, and within this distance of the center line
TRUNK_RADIUS = (0.1, 0.25)  # m
GOAL = jnp.array([16.0, 0.0, 1.0])
WALL_Y = 4.0  # m, |y| beyond this is a crash, so the drone cannot fly around the forest
CEILING = 3.0  # m
CRUISE_SPEED = 3.0  # m/s
DRONE_RADIUS = 0.06  # m, a Crazyflie from propeller tip to propeller tip is about 0.1 m

# Sensor noise, one standard deviation. Depth error grows with distance, as on a stereo camera.
POS_NOISE = 0.02  # m, e.g. from a motion-capture or visual odometry estimate
VEL_NOISE = 0.05  # m/s
TILT_NOISE = 0.01  # added to each rotation matrix entry, ~0.6 degrees
GYRO_NOISE = 0.05  # rad/s
DEPTH_NOISE = (0.02, 0.02)  # m: fixed part, then per m of distance

FRAMES = 1  # depth images in the observation
OBS_SIZE = 18 + FRAMES * depth.PIXELS
ACT_SIZE = 4


class EnvState(NamedTuple):
    drone: sim.State
    body: sim.Body
    trees: jax.Array  # (TREES, 3): x, y, radius
    t: jax.Array
    key: jax.Array  # this episode's sensor noise
    frames: (
        jax.Array
    )  # (FRAMES, PIXELS) noisy depth images scaled to 0..1, newest first


def _noise(s: EnvState, i: int, std: float | jax.Array, shape: tuple[int, ...]):
    """Noise for reading i at this step: the same every time it is asked for."""
    k = jax.random.fold_in(jax.random.fold_in(s.key, s.t), i)
    return std * jax.random.normal(k, shape)


def look(s: EnvState) -> jax.Array:
    """(PIXELS,) noisy depth image right now, scaled to 0..1."""
    seen = depth.image(s.drone.pos, s.drone.quat, s.trees)
    seen = seen + _noise(s, 4, DEPTH_NOISE[0] + DEPTH_NOISE[1] * seen, seen.shape)
    return jnp.clip(seen, 0.0, depth.MAX_DEPTH) / depth.MAX_DEPTH


def observe(s: EnvState) -> jax.Array:
    """Same 18 numbers as waypoint with the goal as target, with sensor noise, then the
    stored depth frames."""
    d = s.drone
    return jnp.concatenate(
        [
            GOAL - d.pos + _noise(s, 0, POS_NOISE, (3,)),
            d.vel + _noise(s, 1, VEL_NOISE, (3,)),
            sim.rotation(d.quat).ravel() + _noise(s, 2, TILT_NOISE, (9,)),
            d.omega + _noise(s, 3, GYRO_NOISE, (3,)),
            s.frames.ravel(),
        ]
    )


def distance(s: EnvState) -> jax.Array:
    """True distance to the target, for tests."""
    return jnp.linalg.norm(GOAL - s.drone.pos)


def hits_tree(pos: jax.Array, trees: jax.Array) -> jax.Array:
    gap = jnp.linalg.norm(pos[:2] - trees[:, :2], axis=1) - trees[:, 2]
    return jnp.any(gap < DRONE_RADIUS)


def random_trees(
    key: jax.Array, n: int = TREES, radius: tuple[float, float] = TRUNK_RADIUS
) -> jax.Array:
    """(n, 3) trunks scattered over the forest patch."""
    k_x, k_y, k_r = jax.random.split(key, 3)
    return jnp.stack(
        [
            jax.random.uniform(k_x, (n,), minval=FOREST_X[0], maxval=FOREST_X[1]),
            jax.random.uniform(k_y, (n,), minval=-FOREST_Y, maxval=FOREST_Y),
            jax.random.uniform(k_r, (n,), minval=radius[0], maxval=radius[1]),
        ],
        axis=1,
    )


def reset(key: jax.Array) -> tuple[EnvState, jax.Array]:
    k_start, k_trees = jax.random.split(key)
    return start(k_start, random_trees(k_trees))


def start(key: jax.Array, trees: jax.Array) -> tuple[EnvState, jax.Array]:
    """New body in the given forest. The drone starts near (0, 0, 1), close to level and
    slow. Every depth frame starts as the first view."""
    k_body, k_pos, k_vel, k_axis, k_angle, k_noise = jax.random.split(key, 6)
    axis = jax.random.normal(k_axis, (3,))
    angle = jax.random.uniform(k_angle, maxval=jnp.radians(10))
    drone = sim.State(
        pos=jax.random.uniform(
            k_pos,
            (3,),
            minval=jnp.array([0.0, -1, 0.8]),
            maxval=jnp.array([0.0, 1, 1.5]),
        ),
        vel=jax.random.uniform(k_vel, (3,), minval=-0.5, maxval=0.5),
        quat=jnp.concatenate(
            [
                jnp.cos(angle / 2)[None],
                jnp.sin(angle / 2) * axis / jnp.linalg.norm(axis),
            ]
        ),
        omega=jnp.zeros(3),
        thrust=jnp.full(4, sim.HOVER_THRUST),
    )
    s = EnvState(
        drone,
        random_body(k_body),
        trees,
        jnp.int32(0),
        k_noise,
        jnp.zeros((FRAMES, depth.PIXELS)),
    )
    s = s._replace(frames=jnp.tile(look(s), (FRAMES, 1)))
    return s, observe(s)


def step(
    s: EnvState, action: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    cmd = sim.HOVER_THRUST * (1 + action)

    # Check for trunks every physics step, not every policy step: at 10 m/s the drone moves
    # 20 cm per policy step, enough to pass clean through a thin trunk between checks.
    def physics(
        _: int, carry: tuple[sim.State, jax.Array]
    ) -> tuple[sim.State, jax.Array]:
        d, hit = carry
        d = sim.step(d, cmd, s.body)
        return d, hit | hits_tree(d.pos, s.trees) | (d.pos[2] < 0)

    drone, hit = jax.lax.fori_loop(0, SUBSTEPS, physics, (s.drone, jnp.bool_(False)))
    dist = jnp.linalg.norm(GOAL - drone.pos)
    progress = jnp.linalg.norm(GOAL - s.drone.pos) - dist
    progress = jnp.minimum(progress, CRUISE_SPEED * SUBSTEPS * sim.DT)
    reward = progress + jnp.exp(-dist - 0.1 * jnp.linalg.norm(drone.omega))
    crashed = hit | (jnp.abs(drone.pos[1]) > WALL_Y) | (drone.pos[2] > CEILING)
    s = s._replace(drone=drone, t=s.t + 1)
    s = s._replace(frames=jnp.concatenate([look(s)[None], s.frames[:-1]]))
    return s, observe(s), reward, crashed, s.t >= MAX_STEPS
