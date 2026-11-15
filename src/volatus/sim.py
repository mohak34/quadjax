"""Quadrotor rigid-body physics in JAX.

`step` moves one drone forward by DT. Wrap it in jax.vmap to step thousands at once.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Crazyflie 2. Mass and inertia match Menagerie cf2.xml; the rest is from Bitcraze specs.
MASS = 0.027  # kg
GRAVITY = 9.81  # m/s^2
INERTIA = jnp.array([2.3951e-5, 2.3951e-5, 3.2347e-5])  # kg m^2, inertia diagonal
ARM = 0.0325  # m, each motor sits this far from the center along x and along y
YAW_COEF = 0.006  # m, yaw torque per newton of motor thrust (propeller drag)
MOTOR_TAU = 0.03  # s, time for a motor to cover 63% of a change in command
MAX_THRUST = 0.15  # N per motor
DT = 0.002  # s, 500 Hz physics

HOVER_THRUST = MASS * GRAVITY / 4  # N per motor to exactly cancel gravity

# Motor order: front-right, back-right, back-left, front-left. Body x points forward, y left, z up.
MOTOR_X = jnp.array([ARM, -ARM, -ARM, ARM])
MOTOR_Y = jnp.array([-ARM, -ARM, ARM, ARM])
# +1 spins counterclockwise seen from above. Diagonal pairs spin the same way so yaw torques cancel.
MOTOR_SPIN = jnp.array([1.0, -1.0, 1.0, -1.0])


class Body(NamedTuple):
    """Physical numbers that differ from drone to drone. Training varies them so the policy
    cannot rely on one exact drone."""

    mass: jax.Array  # kg
    inertia: jax.Array  # (3,) kg m^2
    motor_gain: jax.Array  # (4,) thrust each motor really makes per newton commanded
    motor_tau: jax.Array  # s


NOMINAL = Body(jnp.float32(MASS), INERTIA, jnp.ones(4), jnp.float32(MOTOR_TAU))


class State(NamedTuple):
    pos: jax.Array  # (3,) world frame, m
    vel: jax.Array  # (3,) world frame, m/s
    quat: jax.Array  # (4,) w, x, y, z; rotates body-frame vectors into the world frame
    omega: jax.Array  # (3,) angular velocity in the body frame, rad/s
    thrust: jax.Array  # (4,) what each motor is producing right now, N


def hover_state(pos: jax.Array) -> State:
    """Level, still, motors already spun up to hover thrust."""
    return State(
        pos=pos,
        vel=jnp.zeros(3),
        quat=jnp.array([1.0, 0.0, 0.0, 0.0]),
        omega=jnp.zeros(3),
        thrust=jnp.full(4, HOVER_THRUST),
    )


def mix(thrust: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Four motor thrusts to total thrust (N) and body torque (N m).

    Force f pushing up at (x, y) gives torque (y f, -x f, 0). Each spinning
    propeller also twists the body the opposite way to its spin, which is yaw.
    """
    total = thrust.sum()
    torque = jnp.stack(
        [
            (MOTOR_Y * thrust).sum(),
            (-MOTOR_X * thrust).sum(),
            (-MOTOR_SPIN * YAW_COEF * thrust).sum(),
        ]
    )
    return total, torque


def quat_mul(a: jax.Array, b: jax.Array) -> jax.Array:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def rotation(q: jax.Array) -> jax.Array:
    """3x3 rotation matrix. Its columns are the drone's forward, left and up axes in world coordinates."""
    w, x, y, z = q
    return jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def step(s: State, cmd: jax.Array, body: Body = NOMINAL) -> State:
    """Advance one drone by DT. cmd is the (4,) thrust each motor is asked for, N."""
    # Motors lag: actual thrust closes this fraction of the gap to the command each step.
    alpha = 1.0 - jnp.exp(-DT / body.motor_tau)
    target = jnp.clip(cmd, 0.0, MAX_THRUST) * body.motor_gain
    thrust = s.thrust + (target - s.thrust) * alpha
    total, torque = mix(thrust)

    # Newton: thrust pushes along the body's up axis, gravity pulls straight down.
    acc = rotation(s.quat)[:, 2] * total / body.mass - jnp.array([0.0, 0.0, GRAVITY])
    vel = s.vel + acc * DT
    pos = s.pos + vel * DT

    # Euler's rotation equation: J dw/dt = torque - w x (J w).
    # The cross term is the gyroscopic effect of an already-spinning body.
    omega_dot = (torque - jnp.cross(s.omega, body.inertia * s.omega)) / body.inertia
    omega = s.omega + omega_dot * DT

    # Orientation changes at dq/dt = 0.5 q * (0, w). Renormalize so rounding never stretches q.
    quat = s.quat + 0.5 * DT * quat_mul(s.quat, jnp.concatenate([jnp.zeros(1), omega]))
    quat = quat / jnp.linalg.norm(quat)

    return State(pos, vel, quat, omega, thrust)


@jax.jit
def rollout(s: State, cmds: jax.Array) -> tuple[State, State]:
    """Apply cmds of shape (steps, 4) in order. Returns the final state and every state along the way."""

    def body(s: State, cmd: jax.Array) -> tuple[State, State]:
        s = step(s, cmd)
        return s, s

    return jax.lax.scan(body, s, cmds)
