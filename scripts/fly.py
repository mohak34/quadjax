"""Fly one drone open-loop under three fixed motor commands and print what happens."""

import jax
import jax.numpy as jnp

from volatus import sim


def roll_pitch_deg(q: jax.Array) -> tuple[float, float]:
    w, x, y, z = q
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1, 1))
    return float(jnp.degrees(roll)), float(jnp.degrees(pitch))


def fly(name: str, cmd: jax.Array, seconds: float) -> None:
    steps = round(seconds / sim.DT)
    start = sim.hover_state(jnp.array([0.0, 0.0, 1.0]))
    _, traj = sim.rollout(start, jnp.broadcast_to(cmd, (steps, 4)))

    print(f"\n{name}  motors = {[round(float(c), 4) for c in cmd]} N")
    print(f"{'t (s)':>6} {'x':>8} {'y':>8} {'z':>8} {'roll':>7} {'pitch':>7}")
    every = steps // 6
    for i in range(every - 1, steps, every):
        x, y, z = traj.pos[i]
        roll, pitch = roll_pitch_deg(traj.quat[i])
        print(
            f"{(i + 1) * sim.DT:6.2f} {x:8.4f} {y:8.4f} {z:8.4f} {roll:7.1f} {pitch:7.1f}"
        )


h = sim.HOVER_THRUST
fly("hover: all four at hover thrust", jnp.full(4, h), 1.0)
fly("climb: all four +10%", jnp.full(4, h * 1.1), 1.0)
fly("tip: front-right +5%", jnp.array([h * 1.05, h, h, h]), 0.3)
