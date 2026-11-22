"""Compare the trained policies with a classical controller on the same 1000 starts that
train.py tests on.

The controller is a geometric controller (Lee, Leok and McClamroch, 2010) behind a cascaded
position loop, as in PX4: the position error sets a velocity to fly at, capped at MAX_SPEED
so long moves do not overshoot, and the velocity error plus an integral of the position
error set the acceleration to ask for. That sets the total thrust and which way the drone's
up axis must point. An attitude loop turns the body there
and holds the heading at zero. The four thrusts come from inverting the motor mixer.

It gets the same 18 numbers the policy sees (target offset, velocity, orientation, spin), at
the same decision rate, and only knows the nominal drone. The integral term makes up for a
drone that is heavier or lighter than nominal. It has no idea a motor can fail.

Usage: baseline.py [hover|waypoint|failure ...], default all three.
"""

import pickle
import sys
from typing import Any

import jax
import jax.numpy as jnp

from quadjax import failure, hover, ppo, sim, waypoint

N = 1000
# position error to velocity, velocity error to acceleration, integral: 1/s, 1/s, 1/s^3
KP, KV, KI = 3.0, 9.0, 27.0
KR, KW = 200.0, 25.0  # attitude loop: 1/s^2, 1/s
MAX_SPEED = 1.5  # m/s
MAX_TILT = jnp.radians(35.0)
MAX_INTEGRAL = 3.0 / KI  # m s, so the integral asks for at most 3 m/s^2
G = jnp.array([0.0, 0.0, sim.GRAVITY])

# Rows: total thrust, then roll, pitch and yaw torque, per motor thrust (same as sim.mix).
MIXER = jnp.stack(
    [jnp.ones(4), sim.MOTOR_Y, -sim.MOTOR_X, -sim.MOTOR_SPIN * sim.YAW_COEF]
)
UNMIX = jnp.linalg.inv(MIXER)

TASKS: dict[str, tuple[Any, int]] = {
    "hover": (hover, 50),
    "waypoint": (waypoint, 50),
    "failure": (failure, failure.HZ),
}


def control(
    obs: jax.Array, integral: jax.Array, hz: int
) -> tuple[jax.Array, jax.Array]:
    """One decision: (action in the policy's units, new integral)."""
    error, vel = obs[:3], obs[3:6]
    rot, omega = obs[6:15].reshape(3, 3), obs[15:18]
    integral = jnp.clip(integral + error / hz, -MAX_INTEGRAL, MAX_INTEGRAL)

    want_vel = KP * error
    want_vel = want_vel * jnp.minimum(
        1.0, MAX_SPEED / (jnp.linalg.norm(want_vel) + 1e-9)
    )
    acc = KV * (want_vel - vel) + KI * integral + G
    # Never ask for more tilt than MAX_TILT: shrink the sideways part of the request.
    up = jnp.maximum(acc[2], 0.3 * sim.GRAVITY)
    side = acc[:2] * jnp.minimum(
        1.0, jnp.tan(MAX_TILT) * up / (jnp.linalg.norm(acc[:2]) + 1e-9)
    )
    acc = jnp.concatenate([side, up[None]])

    thrust = sim.MASS * acc @ rot[:, 2]
    z = acc / jnp.linalg.norm(acc)
    y = jnp.cross(z, jnp.array([1.0, 0.0, 0.0]))
    y = y / jnp.linalg.norm(y)
    want = jnp.stack([jnp.cross(y, z), y, z], axis=1)

    # Rotation error: how far, and about which body axis, the body is from where it should be.
    e = 0.5 * (want.T @ rot - rot.T @ want)
    e_rot = jnp.array([e[2, 1], e[0, 2], e[1, 0]])
    torque = sim.INERTIA * (-KR * e_rot - KW * omega) + jnp.cross(
        omega, sim.INERTIA * omega
    )
    motors = UNMIX @ jnp.concatenate([thrust[None], torque])
    return motors / sim.HOVER_THRUST - 1, integral


def run(name: str) -> None:
    env, hz = TASKS[name]
    with open(f"out/{name}_params.pkl", "rb") as f:
        actor = pickle.load(f).actor
    start, obs = jax.vmap(env.reset)(jax.random.split(jax.random.key(1), N))

    def policy(
        c: tuple[Any, jax.Array], _step: None
    ) -> tuple[tuple[Any, jax.Array], jax.Array]:
        s, obs = c
        s, obs, _, crashed, _ = jax.vmap(env.step)(s, ppo.mlp(actor, obs))
        return (s, obs), crashed

    def classical(
        c: tuple[Any, jax.Array, jax.Array], _step: None
    ) -> tuple[tuple[Any, jax.Array, jax.Array], jax.Array]:
        s, obs, integral = c
        action, integral = jax.vmap(control, in_axes=(0, 0, None))(obs, integral, hz)
        s, obs, _, crashed, _ = jax.vmap(env.step)(s, action)
        return (s, obs, integral), crashed

    (end, _), crashed = jax.lax.scan(policy, (start, obs), None, env.MAX_STEPS)
    report(name, "policy", crashed.any(axis=0), jax.vmap(env.distance)(end))
    (end, _, _), crashed = jax.lax.scan(
        classical, (start, obs, jnp.zeros((N, 3))), None, env.MAX_STEPS
    )
    report(name, "classical", crashed.any(axis=0), jax.vmap(env.distance)(end))


def report(name: str, who: str, crashed: jax.Array, dist: jax.Array) -> None:
    ok = dist[~crashed]
    median = f"{float(jnp.median(ok)) * 100:5.1f} cm" if ok.size else "    -"
    print(
        f"{name:9s} {who:10s} crashed {float(crashed.mean()) * 100:5.1f}%  "
        f"median distance at the end {median}",
        flush=True,
    )


for name in sys.argv[1:] or list(TASKS):
    run(name)
