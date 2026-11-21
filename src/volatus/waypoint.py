"""Waypoint task: from any flying state, go to a target point and hold it, on a drone whose
mass, inertia and motors are a little different every episode (domain randomization).

Each episode has one target. A course of waypoints is just a sequence of targets: swap in
the next one and the policy flies there (see scripts/watch_waypoint.py). Training on targets
that jump mid-episode was worse, because the critic cannot see when the next jump is coming.

The policy sees the same 18 numbers as in hover and never sees the body numbers,
so it has to fly any drone in the range.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from volatus import ppo, sim

SUBSTEPS = 10  # 500 Hz physics, 50 Hz policy
MAX_STEPS = 200  # 4 s
OBS_SIZE = 18
ACT_SIZE = 4
CONFIG = ppo.Config(total_steps=150_000_000)
CHECKS = [(f"after {sec} s", sec * 50 - 1) for sec in (1, 2, 4)]

# Targets and starts live in this box, m. Leaving it by more than MARGIN is a crash.
BOX_LOW = jnp.array([-1.5, -1.5, 0.5])
BOX_HIGH = jnp.array([1.5, 1.5, 2.0])
MARGIN = jnp.array([1.0, 1.0, 0.5])  # the low z margin puts the ground at the edge


class EnvState(NamedTuple):
    drone: sim.State
    body: sim.Body
    target: jax.Array  # (3,)
    t: jax.Array


def observe(s: EnvState) -> jax.Array:
    d = s.drone
    return jnp.concatenate(
        [s.target - d.pos, d.vel, sim.rotation(d.quat).ravel(), d.omega]
    )


def distance(s: EnvState) -> jax.Array:
    """True distance to the target, for tests."""
    return jnp.linalg.norm(s.target - s.drone.pos)


def random_body(key: jax.Array) -> sim.Body:
    """Mass and inertia within 20% of the real Crazyflie, each motor within 10%, motor lag 20-50 ms."""
    k_mass, k_inertia, k_gain, k_tau = jax.random.split(key, 4)

    def scale(k: jax.Array, spread: float, shape: tuple[int, ...] = ()) -> jax.Array:
        return jax.random.uniform(k, shape, minval=1 - spread, maxval=1 + spread)

    return sim.Body(
        mass=sim.NOMINAL.mass * scale(k_mass, 0.2),
        inertia=sim.NOMINAL.inertia * scale(k_inertia, 0.2, (3,)),
        motor_gain=scale(k_gain, 0.1, (4,)),
        motor_tau=jax.random.uniform(k_tau, minval=0.02, maxval=0.05),
    )


def reset(key: jax.Array) -> tuple[EnvState, jax.Array]:
    """Random body, start and target in the box. The start is mid-flight: up to 2 m/s
    in any direction, tilted up to 30 degrees, spinning."""
    k_body, k_pos, k_vel, k_axis, k_angle, k_omega, k_target = jax.random.split(key, 7)
    axis = jax.random.normal(k_axis, (3,))
    axis = axis / jnp.linalg.norm(axis)
    angle = jax.random.uniform(k_angle, maxval=jnp.radians(30))
    drone = sim.State(
        pos=jax.random.uniform(k_pos, (3,), minval=BOX_LOW, maxval=BOX_HIGH),
        vel=jax.random.uniform(k_vel, (3,), minval=-2, maxval=2),
        quat=jnp.concatenate([jnp.cos(angle / 2)[None], jnp.sin(angle / 2) * axis]),
        omega=jax.random.uniform(k_omega, (3,), minval=-1, maxval=1),
        thrust=jnp.full(4, sim.HOVER_THRUST),
    )
    target = jax.random.uniform(k_target, (3,), minval=BOX_LOW, maxval=BOX_HIGH)
    s = EnvState(drone, random_body(k_body), target, jnp.int32(0))
    return s, observe(s)


def step(
    s: EnvState, action: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Same action and reward as hover."""
    cmd = sim.HOVER_THRUST * (1 + action)
    drone = jax.lax.fori_loop(
        0, SUBSTEPS, lambda _, d: sim.step(d, cmd, s.body), s.drone
    )
    s = s._replace(drone=drone, t=s.t + 1)
    dist = jnp.linalg.norm(s.target - drone.pos)
    reward = jnp.exp(-dist - 0.1 * jnp.linalg.norm(drone.omega))
    out = jnp.any(drone.pos < BOX_LOW - MARGIN) | jnp.any(drone.pos > BOX_HIGH + MARGIN)
    return s, observe(s), reward, out, s.t >= MAX_STEPS
