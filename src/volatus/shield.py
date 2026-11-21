"""Safety filter for the forest policy: before using an action, check that the drone could
still stop safely after taking it. If not, brake instead.

The check: fly the policy's action for one step, then hand over to a backup that stops at
the point reached and holds it, and simulate that for HORIZON steps. If that path touches
anything (with MARGIN to spare), the action is rejected and the drone brakes now, toward
the point it is at. That point stays locked until the policy's action passes again, so the
brake being flown is always one the check already passed a step earlier. The backup is the
waypoint policy with the locked point as its target.

This only guarantees anything as far as the check's model is right. By default it uses the
nominal drone (not the randomized one really flying), the true position and the true tree
map. A real drone would build that map from its own depth frames.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from volatus import forest, ppo, sim

HORIZON = 75  # policy steps of braking checked ahead, 1.5 s: stop, swing back, settle
MARGIN = 0.1  # m added to every trunk, the ground, the walls and the ceiling


class Mode(NamedTuple):
    braking: jax.Array  # bool: the last action was the brake
    hold: jax.Array  # (3,) where the brake is stopping


def idle() -> Mode:
    return Mode(jnp.bool_(False), jnp.zeros(3))


def brake(backup: ppo.Layers, d: sim.State, hold: jax.Array) -> jax.Array:
    obs = jnp.concatenate([hold - d.pos, d.vel, sim.rotation(d.quat).ravel(), d.omega])
    return ppo.mlp(backup, obs)


def too_close(pos: jax.Array, trees: jax.Array) -> jax.Array:
    gap = jnp.linalg.norm(pos[:2] - trees[:, :2], axis=1) - trees[:, 2]
    return (
        jnp.any(gap < forest.DRONE_RADIUS + MARGIN)
        | (pos[2] < MARGIN)
        | (jnp.abs(pos[1]) > forest.WALL_Y - MARGIN)
        | (pos[2] > forest.CEILING - MARGIN)
    )


def fly(
    d: sim.State, action: jax.Array, trees: jax.Array, body: sim.Body
) -> tuple[sim.State, jax.Array]:
    """One policy step, checked every physics step."""
    cmd = sim.HOVER_THRUST * (1 + action)

    def physics(
        _: int, carry: tuple[sim.State, jax.Array]
    ) -> tuple[sim.State, jax.Array]:
        d, hit = carry
        d = sim.step(d, cmd, body)
        return d, hit | too_close(d.pos, trees)

    return jax.lax.fori_loop(0, forest.SUBSTEPS, physics, (d, jnp.bool_(False)))


def would_crash(
    backup: ppo.Layers,
    d: sim.State,
    action: jax.Array,
    trees: jax.Array,
    body: sim.Body,
) -> jax.Array:
    d, hit = fly(d, action, trees, body)
    hold = d.pos

    def braking(
        c: tuple[sim.State, jax.Array], _step: None
    ) -> tuple[tuple[sim.State, jax.Array], None]:
        d, hit = c
        d, h = fly(d, brake(backup, d, hold), trees, body)
        return (d, hit | h), None

    (_, hit), _ = jax.lax.scan(braking, (d, hit), None, HORIZON)
    return hit


def filter_action(
    backup: ppo.Layers,
    s: forest.EnvState,
    action: jax.Array,
    mode: Mode,
    body: sim.Body = sim.NOMINAL,
) -> tuple[jax.Array, Mode]:
    """The action to fly and the new mode. body is the drone the check simulates."""
    bad = would_crash(backup, s.drone, action, s.trees, body)
    hold = jnp.where(mode.braking, mode.hold, s.drone.pos)
    return jnp.where(bad, brake(backup, s.drone, hold), action), Mode(bad, hold)
