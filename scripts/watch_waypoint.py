"""Three drones with different bodies fly the same waypoint course with the trained
waypoint policy. Each moves on to the next waypoint once within REACHED of the current one.

Prints how long each drone took per waypoint and saves out/waypoint.gif.
"""

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from quadjax import ppo, render, sim, waypoint

OUT = Path("out")
REACHED = 0.15  # m
MAX_STEPS = 750  # 15 s
COURSE = jnp.array(
    [
        [1.2, 1.2, 1.0],
        [-1.2, 1.2, 1.8],
        [-1.2, -1.2, 0.8],
        [1.2, -1.2, 1.8],
        [0.0, 0.0, 1.2],
    ]
)
N = 3
NOMINAL = sim.NOMINAL
BODIES = [
    ("light (mass -20%)", NOMINAL._replace(mass=NOMINAL.mass * 0.8)),
    ("normal", NOMINAL),
    (
        "heavy and lopsided (mass +20%, one motor -10%)",
        NOMINAL._replace(mass=NOMINAL.mass * 1.2, motor_gain=jnp.array([0.9, 1, 1, 1])),
    ),
]

with open(OUT / "waypoint_params.pkl", "rb") as f:
    params: ppo.Params = pickle.load(f)

body = jax.tree.map(lambda *xs: jnp.stack(xs), *(b for _, b in BODIES))
drone = jax.vmap(sim.hover_state)(jnp.tile(jnp.array([0.0, 0.0, 0.6]), (N, 1)))
state = waypoint.EnvState(
    drone, body, jnp.tile(COURSE[0], (N, 1)), jnp.zeros(N, jnp.int32)
)


@jax.jit
def policy_step(s: waypoint.EnvState) -> waypoint.EnvState:
    obs = jax.vmap(waypoint.observe)(s)
    s, *_ = jax.vmap(waypoint.step)(s, ppo.mlp(params.actor, obs))
    return s


scene = render.Scene(
    N, np.asarray(COURSE), lookat=np.array([0.0, 0.0, 1.0]), distance=5.5
)
next_wp = np.zeros(N, int)
arrivals = [[] for _ in range(N)]
frames = []
for t in range(MAX_STEPS):
    if t % 2 == 0:  # policy runs at 50 Hz, the GIF at 25 fps
        frames.append(
            scene.frame(np.asarray(state.drone.pos), np.asarray(state.drone.quat))
        )
    state = policy_step(state)
    dist = np.linalg.norm(np.asarray(state.target - state.drone.pos), axis=1)
    for i in range(N):
        if next_wp[i] < len(COURSE) and dist[i] < REACHED:
            arrivals[i].append((t + 1) / 50)
            next_wp[i] += 1
    if (next_wp == len(COURSE)).all() and t >= max(a[-1] for a in arrivals) * 50 + 75:
        break  # everyone finished; keep 1.5 s of holding at the end
    state = state._replace(target=COURSE[np.minimum(next_wp, len(COURSE) - 1)])

for (name, _), times in zip(BODIES, arrivals):
    shown = ", ".join(f"{x:.1f}" for x in times)
    print(f"{name}: reached waypoints at {shown} s ({len(times)}/{len(COURSE)})")
render.save_gif(frames, OUT / "waypoint.gif")
print(f"saved {OUT}/waypoint.gif ({len(frames)} frames)")
