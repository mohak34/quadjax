"""Motor failure task: hold a point while, at a random moment, one motor weakens or dies.

Half the failures kill the motor outright; the rest leave it at 0-50% of its thrust.
The policy is told each motor's health (1 = fine, 0 = dead), as if a separate fault detector
had spotted the failure. Without that it would have to work it out from how the drone moves.

With one motor dead, the drone cannot hold its heading: the three left over cannot cancel
each other's twist. What still works is to give up on heading and spin fast about the
vertical axis while holding position. So the reward only counts roll and pitch spin,
never yaw spin, otherwise the one way to survive would be punished.

The policy runs at HZ = 250 instead of 50: spinning at ~20 rad/s, the drone would turn
almost a quarter turn between two decisions at 50 Hz. gamma goes up to match, so the
policy still looks about as far ahead in seconds.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from quadjax import ppo, sim
from quadjax.waypoint import random_body

# m, higher than hover: the failure costs some height
TARGET = jnp.array([0.0, 0.0, 1.5])
HZ = 250  # policy decisions per second
SUBSTEPS = round(1 / (HZ * sim.DT))  # physics steps per decision
MAX_STEPS = 6 * HZ  # 6 s
FAIL_AT = (HZ // 2, 2 * HZ)  # policy steps: the failure comes between 0.5 and 2 s in
OBS_SIZE = 22  # hover's 18, then the four motor healths
ACT_SIZE = 4
CONFIG = ppo.Config(total_steps=1_000_000_000, gamma=0.998)
CHECKS = [(f"after {sec} s", sec * HZ - 1) for sec in (2, 4, 6)]


class EnvState(NamedTuple):
    drone: sim.State
    body: sim.Body
    motor: jax.Array  # which motor fails
    left: jax.Array  # its thrust after the failure, as a share of normal
    fail_at: jax.Array  # policy step it fails at
    t: jax.Array


def health(s: EnvState) -> jax.Array:
    """(4,) share of normal thrust each motor can make right now."""
    hit = (jnp.arange(4) == s.motor) & (s.t >= s.fail_at)
    return jnp.where(hit, s.left, 1.0)


def observe(s: EnvState) -> jax.Array:
    d = s.drone
    return jnp.concatenate(
        [TARGET - d.pos, d.vel, sim.rotation(d.quat).ravel(), d.omega, health(s)]
    )


def distance(s: EnvState) -> jax.Array:
    """True distance to the target, for tests."""
    return jnp.linalg.norm(TARGET - s.drone.pos)


def reset(key: jax.Array) -> tuple[EnvState, jax.Array]:
    """Random body, near the target and nearly still. Picks the failure in advance."""
    k_body, k_pos, k_vel, k_motor, k_dead, k_left, k_at = jax.random.split(key, 7)
    drone = sim.hover_state(
        TARGET + jax.random.uniform(k_pos, (3,), minval=-0.3, maxval=0.3)
    )._replace(vel=jax.random.uniform(k_vel, (3,), minval=-0.3, maxval=0.3))
    left = jnp.where(
        jax.random.bernoulli(k_dead),
        0.0,
        jax.random.uniform(k_left, maxval=0.5),
    )
    s = EnvState(
        drone,
        random_body(k_body),
        jax.random.randint(k_motor, (), 0, 4),
        left,
        jax.random.randint(k_at, (), *FAIL_AT),
        jnp.int32(0),
    )
    return s, observe(s)


def step(
    s: EnvState, action: jax.Array
) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
    cmd = sim.HOVER_THRUST * (1 + action)
    body = s.body._replace(motor_gain=s.body.motor_gain * health(s))
    drone = jax.lax.fori_loop(0, SUBSTEPS, lambda _, d: sim.step(d, cmd, body), s.drone)
    s = s._replace(drone=drone, t=s.t + 1)
    dist = jnp.linalg.norm(TARGET - drone.pos)
    reward = jnp.exp(-dist - 0.1 * jnp.linalg.norm(drone.omega[:2]))
    crashed = (drone.pos[2] < 0) | (dist > 3)
    return s, observe(s), reward, crashed, s.t >= MAX_STEPS
