"""Hover task: start somewhere random near a target point, fly to it and hold still.

One call handles one drone; ppo.py vmaps it over thousands.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from quadjax import ppo, sim

TARGET = jnp.array([0.0, 0.0, 1.0])  # m, world frame; the ground is z = 0
SUBSTEPS = 10  # physics steps per policy decision: 500 Hz physics, 50 Hz policy
MAX_STEPS = 250  # policy decisions per episode = 5 s
OBS_SIZE = 18
ACT_SIZE = 4
CONFIG = ppo.Config(total_steps=30_000_000)
CHECKS = [(f"after {sec} s", sec * 50 - 1) for sec in (1, 2, 5)]  # (label, policy step)


class EnvState(NamedTuple):
    drone: sim.State
    t: jax.Array  # policy steps taken this episode


def observe(s: EnvState) -> jax.Array:
    """What the policy sees: where the target is, velocity, orientation, spin. All 18 numbers sit around -3..3."""
    d = s.drone
    return jnp.concatenate(
        [TARGET - d.pos, d.vel, sim.rotation(d.quat).ravel(), d.omega]
    )


def distance(s: EnvState) -> jax.Array:
    """True distance to the target, for tests."""
    return jnp.linalg.norm(TARGET - s.drone.pos)


def reset(key: jax.Array) -> tuple[EnvState, jax.Array]:
    """New episode: within 1 m sideways and 0.5 m up/down of the target, drifting, tilted up to 30 degrees."""
    k_pos, k_vel, k_axis, k_angle, k_omega = jax.random.split(key, 5)
    axis = jax.random.normal(k_axis, (3,))
    axis = axis / jnp.linalg.norm(axis)
    angle = jax.random.uniform(k_angle, maxval=jnp.radians(30))
    drone = sim.State(
        pos=TARGET
        + jax.random.uniform(k_pos, (3,), minval=-1, maxval=1) * jnp.array([1, 1, 0.5]),
        vel=jax.random.uniform(k_vel, (3,), minval=-0.5, maxval=0.5),
        quat=jnp.concatenate([jnp.cos(angle / 2)[None], jnp.sin(angle / 2) * axis]),
        omega=jax.random.uniform(k_omega, (3,), minval=-1, maxval=1),
        thrust=jnp.full(4, sim.HOVER_THRUST),
    )
    s = EnvState(drone, jnp.int32(0))
    return s, observe(s)


def step(
    s: EnvState, action: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    """One policy decision. Returns (state, obs, reward, crashed, timed_out).

    action is 4 numbers, one per motor: 0 means hover thrust, +0.1 means 10% more, -1 means off.
    """
    cmd = sim.HOVER_THRUST * (1 + action)
    drone = jax.lax.fori_loop(0, SUBSTEPS, lambda _, d: sim.step(d, cmd), s.drone)
    s = EnvState(drone, s.t + 1)

    dist = jnp.linalg.norm(TARGET - drone.pos)
    # 1 when on target and not spinning, sliding toward 0 as either grows. Never negative,
    # so staying alive always beats crashing.
    reward = jnp.exp(-dist - 0.1 * jnp.linalg.norm(drone.omega))
    crashed = (drone.pos[2] < 0) | (dist > 3)
    return s, observe(s), reward, crashed, s.t >= MAX_STEPS
