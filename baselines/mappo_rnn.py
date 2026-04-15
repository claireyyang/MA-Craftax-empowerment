"""
Code is adapted from the MAPPO RNN implementation of JaxMARL (https://github.com/FLAIROx/JaxMARL/tree/main) 
Credit goes to the original authors: Rutherford et al.
"""

# ===========================
# Imports and Configuration
# ===========================
import os
import sys
import sqlite3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import functools
import yaml
from functools import partial
from typing import Sequence, NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import optax
import distrax

import wandb

from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.wrappers.baselines import JaxMARLWrapper

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_coop.world_gen.empowerment import estimate_craftax_empowerment_monte_carlo


# ===========================
# Environment Wrappers
# ===========================
class WorldStateWrapper(JaxMARLWrapper):
    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """ 
        For each agent: [agent obs, all other agent obs]
        """
        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])

# ===========================
# Model Definitions
# ===========================
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi

class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)

# ===========================
# Data Structures and Utilities
# ===========================
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def unbatchify_actions(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs))
    return {a: x[i] for i, a in enumerate(agent_list)}

# ===========================
# Training Function
# ===========================
def make_train(config, env):
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = WorldStateWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify_actions(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                
                if config.get("RANDOM_ASSISTANT", False):
                    rng, _rng_rand = jax.random.split(rng)
                    num_acts = env.action_space("agent_1").n
                    env_act["agent_1"] = jax.random.randint(_rng_rand, (config["NUM_ENVS"],), 0, num_acts)

                # VALUE
                world_state = last_obs["world_state"].swapaxes(0,1)
                world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                # EMPOWERMENT HYBRID REWARD
                def _calculate_empowerment(state):
                    # env._env._env accesses the unwrapped CraftaxCoopSymbolicEnv
                    # state is LogEnvState, its .env_state is the inner EnvState
                    return estimate_craftax_empowerment_monte_carlo(
                        env._env._env, state.env_state, n_trajectories=5, horizon=3
                    )
                empowerment_by_agent = jax.vmap(_calculate_empowerment)(env_state)
                # empowerment_by_agent is a list [agent0_emp, agent1_emp, agent2_emp]
                # each tensor is of shape [NUM_ENVS]
                alpha = 1.0
                reward["agent_1"] = reward["agent_1"] + alpha * empowerment_by_agent[2]

                # Sharing frequencies (ignoring the exact boundary step for resets)
                food_diff = env_state.env_state.food_given_matrix - runner_state[1].env_state.food_given_matrix
                drink_diff = env_state.env_state.drink_given_matrix - runner_state[1].env_state.drink_given_matrix
                step_food_given = jnp.maximum(0, food_diff)
                step_drink_given = jnp.maximum(0, drink_diff)
                
                info["custom_step_food_h_to_u"] = jnp.repeat(step_food_given[:, 1, 2], 3)
                info["custom_step_food_h_to_b"] = jnp.repeat(step_food_given[:, 1, 0], 3)
                info["custom_step_drink_h_to_u"] = jnp.repeat(step_drink_given[:, 1, 2], 3)
                info["custom_step_drink_h_to_b"] = jnp.repeat(step_drink_given[:, 1, 0], 3)

                info["custom_emp_agent_0"] = empowerment_by_agent[0]
                info["custom_emp_agent_1"] = empowerment_by_agent[1]
                info["custom_emp_agent_2"] = empowerment_by_agent[2]
                
                info["custom_val_agent_1"] = value.reshape((3, config["NUM_ENVS"]))[1]
                info["custom_rew_agent_0"] = reward["agent_0"]
                info["custom_rew_agent_1"] = reward["agent_1"]
                info["custom_rew_agent_2"] = reward["agent_2"]

                # Extract health and alive status
                info["custom_health_agent_0"] = env_state.env_state.player_health[:, 0]
                info["custom_health_agent_1"] = env_state.env_state.player_health[:, 1]
                info["custom_health_agent_2"] = env_state.env_state.player_health[:, 2]
                info["custom_alive_agent_0"] = env_state.env_state.player_alive[:, 0]
                info["custom_alive_agent_1"] = env_state.env_state.player_alive[:, 1]
                info["custom_alive_agent_2"] = env_state.env_state.player_alive[:, 2]

                agent_mask = jnp.ones((len(env.agents), config["NUM_ENVS"]))
                if config.get("RANDOM_ASSISTANT", False):
                    agent_idx = env.agents.index("agent_1")
                    agent_mask = agent_mask.at[agent_idx].set(0.0)
                info["agent_mask"] = agent_mask.reshape((config["NUM_ACTORS"],))

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        
                        mask = traj_batch.info["agent_mask"]
                        loss_actor = (loss_actor * mask).sum() / (mask.sum() + 1e-8)
                        entropy = (pi.entropy() * mask).sum() / (mask.sum() + 1e-8)

                        # debug
                        approx_kl = (((ratio - 1) - logratio) * mask).sum() / (mask.sum() + 1e-8)
                        clip_frac = (jnp.array(jnp.abs(ratio - 1) > config["CLIP_EPS"], dtype=jnp.float32) * mask).sum() / (mask.sum() + 1e-8)

                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state,  traj_batch.done)) 

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped)
                        )
                        mask = traj_batch.info["agent_mask"]
                        value_loss = (value_loss * mask).sum() / (mask.sum() + 1e-8)
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree.map(lambda x: jnp.reshape(
                    x, (1, config["NUM_ACTORS"], -1)
                ), init_hstates)

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            metric["update_steps"] = update_steps
            rng = update_state[-1]

            def callback(metric):
                env_step = (
                    metric["update_steps"]
                    * config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                )
                to_log = {
                    "env_step": env_step,
                    **metric["loss"],
                }

                # Approximate per-episode quantities via rollout sums scaled by typical frequencies
                to_log["agent_0_mean_episode_returns"] = float(metric["custom_rew_agent_2"].sum(axis=0).mean())
                to_log["agent_1_mean_episode_returns"] = float(metric["custom_rew_agent_0"].sum(axis=0).mean())
                to_log["helper_mean_episode_returns"] = float(metric["custom_rew_agent_1"].sum(axis=0).mean())

                to_log["agent_0_mean_episode_empowerment"] = float(metric["custom_emp_agent_2"].mean())
                to_log["agent_1_mean_episode_empowerment"] = float(metric["custom_emp_agent_0"].mean())
                to_log["helper_mean_episode_critic_predictions"] = float(metric["custom_val_agent_1"].mean())
                to_log["freeze_frequency_mean"] = 0.0
                
                # Format to match index assignments from returns (agent_0->2, agent_1->0, helper->1)
                to_log["agent_0_mean_health"] = float(metric["custom_health_agent_2"].mean())
                to_log["agent_1_mean_health"] = float(metric["custom_health_agent_0"].mean())
                to_log["helper_mean_health"] = float(metric["custom_health_agent_1"].mean())
                to_log["agent_0_mean_alive"] = float(metric["custom_alive_agent_2"].mean())
                to_log["agent_1_mean_alive"] = float(metric["custom_alive_agent_0"].mean())
                to_log["helper_mean_alive"] = float(metric["custom_alive_agent_1"].mean())
                
                to_log["helper_gave_food_to_user"] = float(metric["custom_step_food_h_to_u"].sum(axis=0).mean())
                to_log["helper_gave_food_to_bystander"] = float(metric["custom_step_food_h_to_b"].sum(axis=0).mean())
                to_log["helper_gave_water_to_user"] = float(metric["custom_step_drink_h_to_u"].sum(axis=0).mean())
                to_log["helper_gave_water_to_bystander"] = float(metric["custom_step_drink_h_to_b"].sum(axis=0).mean())

                if metric["returned_episode"].any():
                    to_log.update(jax.tree.map(
                        lambda x: float(x[metric["returned_episode"]].mean()),
                        metric["user_info"]
                    ))
                    to_log["episode_lengths"] = float(metric["returned_episode_lengths"][metric["returned_episode"]].mean())
                    to_log["episode_returns"] = float(metric["returned_episode_returns"][metric["returned_episode"]].mean())
                print({k: v for k, v in to_log.items() if "loss" not in k})
                
                if wandb.run is not None:
                    wandb.log(to_log)
                    wandb_id = wandb.run.id
                else:
                    wandb_id = "offline"

                db_path = "train_data/experiment_data.db"
                os.makedirs("train_data", exist_ok=True)
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS phase2_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        wandbid TEXT,
                        helper_objective TEXT,
                        epoch INTEGER,
                        helper_mean_episode_returns REAL,
                        helper_mean_episode_critic_predictions REAL,
                        agent_0_mean_episode_returns REAL,
                        agent_1_mean_episode_returns REAL,
                        agent_0_mean_episode_empowerment REAL,
                        agent_1_mean_episode_empowerment REAL,
                        freeze_frequency_mean REAL,
                        agent_0_mean_health REAL,
                        agent_1_mean_health REAL,
                        helper_mean_health REAL,
                        agent_0_mean_alive REAL,
                        agent_1_mean_alive REAL,
                        helper_mean_alive REAL,
                        helper_gave_food_to_user REAL,
                        helper_gave_food_to_bystander REAL,
                        helper_gave_water_to_user REAL,
                        helper_gave_water_to_bystander REAL
                    )
                ''')
                
                # Add columns if they do not exist (for backward compatibility with existing db)
                try:
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN agent_0_mean_health REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN agent_1_mean_health REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN helper_mean_health REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN agent_0_mean_alive REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN agent_1_mean_alive REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN helper_mean_alive REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN helper_gave_food_to_user REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN helper_gave_food_to_bystander REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN helper_gave_water_to_user REAL')
                    cursor.execute('ALTER TABLE phase2_metrics ADD COLUMN helper_gave_water_to_bystander REAL')
                except sqlite3.OperationalError:
                    pass # Columns already exist

                cursor.execute('CREATE INDEX IF NOT EXISTS idx_wandbid ON phase2_metrics (wandbid)')
                
                cursor.execute('''
                    INSERT INTO phase2_metrics (
                        wandbid, helper_objective, epoch, helper_mean_episode_returns,
                        helper_mean_episode_critic_predictions, agent_0_mean_episode_returns,
                        agent_1_mean_episode_returns, agent_0_mean_episode_empowerment,
                        agent_1_mean_episode_empowerment, freeze_frequency_mean,
                        agent_0_mean_health, agent_1_mean_health, helper_mean_health,
                        agent_0_mean_alive, agent_1_mean_alive, helper_mean_alive,
                        helper_gave_food_to_user, helper_gave_food_to_bystander,
                        helper_gave_water_to_user, helper_gave_water_to_bystander
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    wandb_id,
                    "user_empowerment_hybrid",
                    int(metric["update_steps"]),
                    to_log["helper_mean_episode_returns"],
                    to_log["helper_mean_episode_critic_predictions"],
                    to_log["agent_0_mean_episode_returns"],
                    to_log["agent_1_mean_episode_returns"],
                    to_log["agent_0_mean_episode_empowerment"],
                    to_log["agent_1_mean_episode_empowerment"],
                    to_log["freeze_frequency_mean"],
                    to_log["agent_0_mean_health"],
                    to_log["agent_1_mean_health"],
                    to_log["helper_mean_health"],
                    to_log["agent_0_mean_alive"],
                    to_log["agent_1_mean_alive"],
                    to_log["helper_mean_alive"],
                    to_log["helper_gave_food_to_user"],
                    to_log["helper_gave_food_to_bystander"],
                    to_log["helper_gave_water_to_user"],
                    to_log["helper_gave_water_to_bystander"]
                ))
                conn.commit()
                conn.close()

            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# ===========================
# Main Run Function
# ===========================
def single_run(config):
    alg_name = config.get("ALG_NAME", "mappo-rnn")
    env_name = config.get("ENV_NAME", "Craftax-Coop-Symbolic")
    env = make_craftax_env_from_name(env_name)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=config["RUN_NAME"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="Name of the config YAML file (in baselines/config/)")
    parser.add_argument("--random", action="store_true", help="Force agent_1 to act randomly for baseline disempowerment.")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "config", args.config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["RANDOM_ASSISTANT"] = config.get("RANDOM_ASSISTANT", False) or args.random
    single_run(config)


if __name__ == "__main__":
    main()
