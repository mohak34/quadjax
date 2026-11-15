"""PPO for continuous actions, written out in plain JAX.

The whole update (collect experience, score it, learn from it) is one jitted function,
so thousands of environments and the networks all stay on the GPU.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

type Layers = list[tuple[jax.Array, jax.Array]]


class Env[S](NamedTuple):
    reset: Callable[[jax.Array], tuple[S, jax.Array]]
    # (state, action) -> (state, obs, reward, crashed, timed_out)
    step: Callable[[S, jax.Array], tuple[S, jax.Array, jax.Array, jax.Array, jax.Array]]
    obs_size: int
    act_size: int


class Config(NamedTuple):
    num_envs: int = 2048
    num_steps: int = 64  # steps each env takes per update
    total_steps: int = 30_000_000
    epochs: int = 4  # passes over each batch of experience
    minibatches: int = 32
    lr: float = 3e-4
    gamma: float = 0.99  # how much a reward one step later is worth
    gae_lambda: float = 0.95
    clip: float = 0.2  # how far one update may move the policy
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64
    init_log_std: float = -1.0  # starting exploration noise, std = e^-1 = 0.37

    @property
    def num_updates(self) -> int:
        return self.total_steps // (self.num_envs * self.num_steps)


class Params(NamedTuple):
    actor: Layers  # obs -> mean action
    log_std: jax.Array  # (act_size,) log of the exploration noise
    critic: Layers  # obs -> guess of future reward


class Adam(NamedTuple):
    m: Params  # running average of gradients
    v: Params  # running average of squared gradients
    t: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    finished_return: jax.Array  # episode's total reward if it ended this step, else 0


class Runner[S](NamedTuple):
    params: Params
    opt: Adam
    env_state: S
    obs: jax.Array
    ep_return: jax.Array  # running total reward of each env's current episode
    key: jax.Array


def init_mlp(key: jax.Array, sizes: list[int], last_scale: float) -> Layers:
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0)
        w = jax.nn.initializers.orthogonal(scale)(k, (n_in, n_out))
        layers.append((w, jnp.zeros(n_out)))
    return layers


def mlp(layers: Layers, x: jax.Array) -> jax.Array:
    for w, b in layers[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = layers[-1]
    return x @ w + b


def log_prob(action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log probability of action under the policy's Gaussian, summed over action dimensions."""
    z = (action - mean) / jnp.exp(log_std)
    return (-0.5 * z**2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def init_params(key: jax.Array, env: Env, cfg: Config) -> Params:
    k_actor, k_critic = jax.random.split(key)
    h = cfg.hidden
    return Params(
        # Tiny last layer: the untrained policy outputs ~0, which hover reads as "hover thrust".
        actor=init_mlp(k_actor, [env.obs_size, h, h, env.act_size], 0.01),
        log_std=jnp.full(env.act_size, cfg.init_log_std),
        critic=init_mlp(k_critic, [env.obs_size, h, h, 1], 1.0),
    )


def adam_step(
    params: Params, grads: Params, opt: Adam, lr: jax.Array
) -> tuple[Params, Adam]:
    b1, b2, eps = 0.9, 0.999, 1e-5
    t = opt.t + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt.m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g**2, opt.v, grads)
    params = jax.tree.map(
        lambda p, m, v: p - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps),
        params,
        m,
        v,
    )
    return params, Adam(m, v, t)


def clip_by_global_norm(grads: Params, max_norm: float) -> Params:
    norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    return jax.tree.map(lambda g: g * jnp.minimum(1.0, max_norm / (norm + 1e-8)), grads)


def make_train[S](env: Env[S], cfg: Config) -> tuple[Callable, Callable]:
    """Returns (init, update). init(key) builds a Runner; update(runner) does one PPO update."""
    batch = cfg.num_envs * cfg.num_steps
    mb_size = batch // cfg.minibatches

    def init(key: jax.Array) -> Runner[S]:
        k_params, k_env, key = jax.random.split(key, 3)
        params = init_params(k_params, env, cfg)
        zeros = jax.tree.map(jnp.zeros_like, params)
        env_state, obs = jax.vmap(env.reset)(jax.random.split(k_env, cfg.num_envs))
        return Runner(
            params,
            Adam(zeros, zeros, jnp.int32(0)),
            env_state,
            obs,
            jnp.zeros(cfg.num_envs),
            key,
        )

    def collect(r: Runner[S], _: None) -> tuple[Runner[S], Transition]:
        """Every env takes one step with the current policy plus noise."""
        key, k_act, k_env = jax.random.split(r.key, 3)
        mean = mlp(r.params.actor, r.obs)
        action = mean + jnp.exp(r.params.log_std) * jax.random.normal(k_act, mean.shape)
        logp = log_prob(action, mean, r.params.log_std)
        value = mlp(r.params.critic, r.obs)[:, 0]
        env_state, obs, reward, crashed, timed_out = jax.vmap(env.step)(
            r.env_state, action
        )
        # A timeout is not a real end: the drone would have kept earning reward.
        # Credit the critic's guess of that future reward, so timeouts don't look like crashes.
        timeout_only = timed_out & ~crashed
        learn_reward = (
            reward + cfg.gamma * mlp(r.params.critic, obs)[:, 0] * timeout_only
        )
        done = crashed | timed_out
        ep_return = r.ep_return + reward

        fresh_state, fresh_obs = jax.vmap(env.reset)(
            jax.random.split(k_env, cfg.num_envs)
        )
        env_state = jax.tree.map(
            lambda a, b: jnp.where(done.reshape(-1, *[1] * (a.ndim - 1)), a, b),
            fresh_state,
            env_state,
        )
        obs = jnp.where(done[:, None], fresh_obs, obs)

        t = Transition(
            r.obs,
            action,
            logp,
            value,
            learn_reward,
            done,
            jnp.where(done, ep_return, 0.0),
        )
        r = Runner(
            r.params, r.opt, env_state, obs, jnp.where(done, 0.0, ep_return), key
        )
        return r, t

    def advantages(
        traj: Transition, last_value: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """GAE: walk backward in time asking "how much better than the critic expected was this?"."""

        def back(carry: tuple[jax.Array, jax.Array], t: Transition):
            gae, next_value = carry
            alive = 1.0 - t.done
            delta = t.reward + cfg.gamma * next_value * alive - t.value
            gae = delta + cfg.gamma * cfg.gae_lambda * alive * gae
            return (gae, t.value), gae

        _, adv = jax.lax.scan(
            back, (jnp.zeros_like(last_value), last_value), traj, reverse=True
        )
        return adv, adv + traj.value  # advantage, and the return target for the critic

    def loss(
        params: Params, t: Transition, adv: jax.Array, target: jax.Array
    ) -> jax.Array:
        mean = mlp(params.actor, t.obs)
        ratio = jnp.exp(log_prob(t.action, mean, params.log_std) - t.logp)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # The PPO clip: once the new policy differs from the old by more than `clip`,
        # pushing further earns nothing, so one batch can't drag it too far.
        policy_loss = -jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
        ).mean()
        value_loss = 0.5 * ((mlp(params.critic, t.obs)[:, 0] - target) ** 2).mean()
        return policy_loss + cfg.vf_coef * value_loss

    def learn(r: Runner[S], data: tuple[Transition, jax.Array, jax.Array]) -> Runner[S]:
        """Several passes over the batch in shuffled minibatches."""

        def epoch(r: Runner[S], _: None) -> tuple[Runner[S], None]:
            key, k_perm = jax.random.split(r.key)
            perm = jax.random.permutation(k_perm, batch)
            mbs = jax.tree.map(
                lambda x: x[perm].reshape(cfg.minibatches, mb_size, *x.shape[1:]), data
            )

            def minibatch(r: Runner[S], mb: tuple[Transition, jax.Array, jax.Array]):
                grads = jax.grad(loss)(r.params, *mb)
                grads = clip_by_global_norm(grads, cfg.max_grad_norm)
                # Learning rate shrinks linearly to 0 so the policy settles at the end.
                done_frac = r.opt.t / (cfg.num_updates * cfg.epochs * cfg.minibatches)
                params, opt = adam_step(
                    r.params, grads, r.opt, cfg.lr * (1 - done_frac)
                )
                return r._replace(params=params, opt=opt), None

            r, _ = jax.lax.scan(minibatch, r._replace(key=key), mbs)
            return r, None

        r, _ = jax.lax.scan(epoch, r, None, cfg.epochs)
        return r

    @jax.jit
    def update(r: Runner[S]) -> tuple[Runner[S], dict[str, jax.Array]]:
        r, traj = jax.lax.scan(collect, r, None, cfg.num_steps)
        adv, target = advantages(traj, mlp(r.params.critic, r.obs)[:, 0])
        flat = jax.tree.map(
            lambda x: x.reshape(batch, *x.shape[2:]), (traj, adv, target)
        )
        r = learn(r, flat)
        episodes = jnp.maximum(traj.done.sum(), 1)
        metrics = {
            "episode_return": traj.finished_return.sum() / episodes,
            "episodes": traj.done.sum(),
            "noise_std": jnp.exp(r.params.log_std).mean(),
        }
        return r, metrics

    return jax.jit(init), update
