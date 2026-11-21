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
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from volatus import depth, sim
from volatus.waypoint import random_body

SUBSTEPS = 10  # 500 Hz physics, 50 Hz policy
MAX_STEPS = 400  # 8 s
TRAIN_STEPS = 300_000_000
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

OBS_SIZE = 18 + depth.PIXELS
ACT_SIZE = 4


class EnvState(NamedTuple):
    drone: sim.State
    body: sim.Body
    trees: jax.Array  # (TREES, 3): x, y, radius
    t: jax.Array


def observe(s: EnvState) -> jax.Array:
    """Same 18 numbers as waypoint with the goal as target, then the depth image scaled to 0..1."""
    d = s.drone
    seen = depth.image(d.pos, d.quat, s.trees) / depth.MAX_DEPTH
    return jnp.concatenate(
        [GOAL - d.pos, d.vel, sim.rotation(d.quat).ravel(), d.omega, seen]
    )


def hits_tree(pos: jax.Array, trees: jax.Array) -> jax.Array:
    gap = jnp.linalg.norm(pos[:2] - trees[:, :2], axis=1) - trees[:, 2]
    return jnp.any(gap < DRONE_RADIUS)


def reset(key: jax.Array) -> tuple[EnvState, jax.Array]:
    """New forest and body. The drone starts near (0, 0, 1), close to level and slow."""
    k_body, k_tx, k_ty, k_tr, k_pos, k_vel, k_axis, k_angle = jax.random.split(key, 8)
    trees = jnp.stack(
        [
            jax.random.uniform(k_tx, (TREES,), minval=FOREST_X[0], maxval=FOREST_X[1]),
            jax.random.uniform(k_ty, (TREES,), minval=-FOREST_Y, maxval=FOREST_Y),
            jax.random.uniform(
                k_tr, (TREES,), minval=TRUNK_RADIUS[0], maxval=TRUNK_RADIUS[1]
            ),
        ],
        axis=1,
    )
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
    s = EnvState(drone, random_body(k_body), trees, jnp.int32(0))
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
    return s, observe(s), reward, crashed, s.t >= MAX_STEPS
