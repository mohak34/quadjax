"""Fly each trained policy in MuJoCo without retraining, and compare with our sim on the same
1000 starts that train.py tests on.

The MuJoCo world differs from ours on purpose, on top of being a different engine:
- each motor is its own actuator pushing up at the end of its arm, so roll, pitch and yaw
  torques come from MuJoCo's geometry, not from our mixer
- MuJoCo's own motor lag, a first-order filter on each command
- RK4 integration instead of semi-implicit Euler
- air drag on the whole body (MuJoCo's inertia-box fluid model in air), which our sim lacks
The yaw drag our sim assumes is applied by hand, since it is part of the world we picked.
The forest depth image still comes from our raycaster, on MuJoCo's pose; check_depth.py
shows it matches MuJoCo's raycaster to well under 1 mm.

Usage: transfer.py [hover|waypoint|forest|failure ...], default all four.

mujoco is compiled C++ with no type stubs, so ty cannot see its members; those lines carry ty ignores.
"""

import pickle
import sys
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from quadjax import failure, forest, hover, ppo, sim, waypoint

N = 1000
DRONE = 'type="general" dyntype="filter" ctrllimited="true"'


class Task(NamedTuple):
    module: Any
    substeps: int
    # true when the drone crashed: checked after every physics step, then after the policy step
    bad_substep: Callable[[Any, jax.Array], jax.Array]
    bad_step: Callable[[Any], jax.Array]


def never(_s: Any, _pos: jax.Array) -> jax.Array:
    return jnp.bool_(False)


def far(target: jax.Array) -> Callable[[Any], jax.Array]:
    return lambda s: (s.drone.pos[2] < 0) | (jnp.linalg.norm(target - s.drone.pos) > 3)


def out_of_box(s: waypoint.EnvState) -> jax.Array:
    pos = s.drone.pos
    return jnp.any(pos < waypoint.BOX_LOW - waypoint.MARGIN) | jnp.any(
        pos > waypoint.BOX_HIGH + waypoint.MARGIN
    )


def in_tree(s: forest.EnvState, pos: jax.Array) -> jax.Array:
    return forest.hits_tree(pos, s.trees) | (pos[2] < 0)


def past_walls(s: forest.EnvState) -> jax.Array:
    pos = s.drone.pos
    return (jnp.abs(pos[1]) > forest.WALL_Y) | (pos[2] > forest.CEILING)


TASKS = {
    "hover": Task(hover, hover.SUBSTEPS, never, far(hover.TARGET)),
    "waypoint": Task(waypoint, waypoint.SUBSTEPS, never, out_of_box),
    "forest": Task(forest, forest.SUBSTEPS, in_tree, past_walls),
    "failure": Task(failure, failure.SUBSTEPS, never, far(failure.TARGET)),
}


def body_of(s: Any) -> sim.Body:
    """Each episode's drone. Hover flies the nominal one."""
    if hasattr(s, "body"):
        return s.body
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (N, *jnp.shape(x))), sim.NOMINAL)


def gains_of(s: Any) -> jax.Array:
    """(N, 4) thrust per newton commanded right now, failures included."""
    gain = body_of(s).motor_gain
    if isinstance(s, failure.EnvState):
        gain = gain * jax.vmap(failure.health)(s)
    return gain


def mujoco_drone(mass: float, inertia: np.ndarray, gain: np.ndarray, tau: float) -> Any:
    sites = "".join(
        f'<site name="m{i}" pos="{float(x)} {float(y)} 0"/>'
        for i, (x, y) in enumerate(zip(sim.MOTOR_X, sim.MOTOR_Y))
    )
    # gear: push 1 N up along the motor's z and twist -spin * YAW_COEF N m about it
    motors = "".join(
        f'<general site="m{i}" gear="0 0 1 0 0 {-float(spin) * sim.YAW_COEF}" '
        f'dyntype="filter" dynprm="{tau}" gainprm="{g}" '
        f'ctrllimited="true" ctrlrange="0 {sim.MAX_THRUST}"/>'
        for i, (spin, g) in enumerate(zip(sim.MOTOR_SPIN, gain))
    )
    xml = f"""
<mujoco>
  <option timestep="{sim.DT}" integrator="RK4" density="1.2" viscosity="1.8e-5"/>
  <worldbody>
    <body>
      <freejoint/>
      <inertial pos="0 0 0" mass="{mass}" diaginertia="{" ".join(str(i) for i in inertia)}"/>
      {sites}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)  # ty: ignore[missing-attribute]


def run(name: str, task: Task) -> None:
    env = task.module
    with open(f"out/{name}_params.pkl", "rb") as f:
        actor = pickle.load(f).actor
    policy = jax.jit(lambda obs: ppo.mlp(actor, obs))
    start, obs0 = jax.vmap(env.reset)(jax.random.split(jax.random.key(1), N))

    # Our sim, exactly as train.py tests it.
    def ours(
        c: tuple[Any, jax.Array], _step: None
    ) -> tuple[tuple[Any, jax.Array], jax.Array]:
        s, obs = c
        s, obs, _, crashed, _ = jax.vmap(env.step)(s, policy(obs))
        return (s, obs), crashed

    (end, _), crashed = jax.lax.scan(ours, (start, obs0), None, env.MAX_STEPS)
    report(name, "our sim", crashed.any(axis=0), jax.vmap(env.distance)(end))

    # MuJoCo. Each policy step, our task code still does the bookkeeping (time, depth frames,
    # past commands) and then the drone is overwritten with where MuJoCo flew it.
    body = jax.device_get(body_of(start))
    models = [
        mujoco_drone(
            float(body.mass[i]),
            body.inertia[i],
            body.motor_gain[i],
            float(body.motor_tau[i]),
        )
        for i in range(N)
    ]
    datas = [mujoco.MjData(m) for m in models]  # ty: ignore[missing-attribute]
    d0 = jax.device_get(start.drone)
    for i, data in enumerate(datas):
        data.qpos[:] = np.concatenate([d0.pos[i], d0.quat[i]])
        data.qvel[:] = np.concatenate([d0.vel[i], d0.omega[i]])
        data.act[:] = d0.thrust[i] / body.motor_gain[i]

    step = jax.jit(jax.vmap(env.step))
    observe = jax.jit(jax.vmap(env.observe))
    bad_substep = jax.jit(jax.vmap(task.bad_substep))
    bad_step = jax.jit(jax.vmap(task.bad_step))
    gains = jax.jit(gains_of)
    look = jax.jit(jax.vmap(forest.look))

    s, obs = start, obs0
    dead = np.zeros(N, bool)
    for _ in range(env.MAX_STEPS):
        action = policy(obs)
        cmd = np.asarray(sim.HOVER_THRUST * (1 + action), np.float64)
        gain = np.asarray(gains(s), np.float64)
        for i, (model, data) in enumerate(zip(models, datas)):
            model.actuator_gainprm[:, 0] = gain[i]
            data.ctrl[:] = cmd[i]
        for _ in range(task.substeps):
            for model, data in zip(models, datas):
                w = data.qvel[5]
                data.qfrc_applied[5] = -sim.YAW_DRAG * w * abs(w)
                mujoco.mj_step(model, data)  # ty: ignore[missing-attribute]
            pos = jnp.asarray(np.stack([data.qpos[:3] for data in datas]), jnp.float32)
            dead |= np.asarray(bad_substep(s, pos))
        s, _, _, _, _ = step(s, action)
        q = np.stack([data.qpos for data in datas])
        v = np.stack([data.qvel for data in datas])
        act = np.stack([data.act for data in datas])
        drone = sim.State(
            pos=jnp.asarray(q[:, :3], jnp.float32),
            vel=jnp.asarray(v[:, :3], jnp.float32),
            quat=jnp.asarray(q[:, 3:], jnp.float32),
            omega=jnp.asarray(v[:, 3:], jnp.float32),
            thrust=jnp.asarray(act * gain, jnp.float32),
        )
        s = s._replace(drone=drone)
        if isinstance(s, forest.EnvState):
            s = s._replace(frames=s.frames.at[:, 0].set(look(s)))
        dead |= np.asarray(bad_step(s))
        obs = observe(s)
    report(name, "MuJoCo", jnp.asarray(dead), jax.vmap(env.distance)(s))


def report(name: str, where: str, crashed: jax.Array, dist: jax.Array) -> None:
    ok = dist[~crashed]
    median = f"{float(jnp.median(ok)) * 100:5.1f} cm" if ok.size else "    -"
    print(
        f"{name:9s} {where:8s} crashed {float(crashed.mean()) * 100:5.1f}%  "
        f"median distance at the end {median}",
        flush=True,
    )


for name in sys.argv[1:] or list(TASKS):
    run(name, TASKS[name])
