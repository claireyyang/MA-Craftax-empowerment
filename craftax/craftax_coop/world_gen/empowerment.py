import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
def estimate_craftax_empowerment_monte_carlo(
    env,
    initial_state,
    n_trajectories: int = 10,
    horizon: int = 3,
    map_width: int = 48,  # Default Craftax-Coop map size
    map_height: int = 48,
):
    """
    Estimates empowerment for Craftax-Coop agents via Monte Carlo rollouts.
    
    IMPORTANT DIFFERENCES FROM GRIDWORLD:
    1. PyTree Batching: Craftax's state is massive. We use jax.tree_util.tree_map to batched 
       the entire dataclass automatically instead of jnp.repeating 50 individual fields.
    2. Joint State Definition: We compute Mutual Information over a joint distribution of 
       (Position x Health x Alive Status) instead of just Position.
    3. Memory Optimized: We use jax.lax.scan over agents and actions. 
       This prevents OOM by forcing sequential execution instead of massive parallel materializations (vmap)
       while maintaining incredibly fast compilation times since the XLA graph is completely collapsed.
    """
    key = jax.random.PRNGKey(0)
    num_agents = len(env.agents)
    
    try:
        num_actions = len(env.action_set)
    except AttributeError:
        num_actions = env.action_spaces[env.agents[0]].n

    # Create batch of initial states [n_trajectories, ...]
    batch_state = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.asarray(x)[None, ...], n_trajectories, axis=0),
        initial_state
    )

    def compute_for_agent_action(agent_idx, action, key):
        key, step_key = jax.random.split(key)
        
        # 1. First step actions
        first_step_actions = {}
        for i, aid in enumerate(env.agents):
            key, subkey = jax.random.split(key)
            rand_acts = jax.random.randint(subkey, (n_trajectories,), 0, num_actions)
            replaced_acts = jnp.where(agent_idx == i, jnp.full((n_trajectories,), action), rand_acts)
            first_step_actions[aid] = replaced_acts

        _, next_state, _, _, _ = jax.vmap(
            lambda s, a_dict: env.step_env(step_key, s, a_dict)
        )(batch_state, first_step_actions)

        # 2. Remaining steps via scan
        def step_trajectory(carry, _):
            current_state, loop_key = carry
            
            random_actions = {}
            key_splits = jax.random.split(loop_key, num_agents + 1)
            next_loop_key = key_splits[0]

            for i, aid in enumerate(env.agents):
                random_actions[aid] = jax.random.randint(
                    key_splits[i + 1], (n_trajectories,), 0, num_actions
                )

            _, n_state, _, _, _ = jax.vmap(
                lambda s, a_dict: env.step_env(key_splits[-1], s, a_dict)
            )(current_state, random_actions)

            return (n_state, next_loop_key), n_state

        initial_carry = (next_state, key)
        (final_state, _), _ = jax.lax.scan(
            step_trajectory, initial_carry, None, length=horizon - 1
        )

        # Extract features specifically for the target agent
        agent_pos = jnp.take(final_state.player_position, agent_idx, axis=1) # shape: [n_traj, 2]
        agent_alive = jnp.take(final_state.player_alive, agent_idx, axis=1)  # shape: [n_traj]
        agent_health = jnp.take(final_state.player_health, agent_idx, axis=1) # shape: [n_traj]

        health_bins = jnp.where(agent_health > 4, 2, jnp.where(agent_health > 0, 1, 0))
        alive_int = agent_alive.astype(jnp.int32)
        
        w_x_h = map_width * map_height
        state_indices = (
            (alive_int * 3 * w_x_h) +
            (health_bins * w_x_h) +
            (agent_pos[:, 1] * map_width) + # Y
            (agent_pos[:, 0])               # X
        )
        return state_indices

    # Scan sequentially over actions to prevent huge memory allocations
    def compute_for_agent(carry_unused, iter_data):
        agent_idx, key = iter_data
        
        actions = jnp.arange(num_actions)
        keys = jax.random.split(key, num_actions)
        
        def scan_over_actions(c, act_data):
            action, a_key = act_data
            res = compute_for_agent_action(agent_idx, action, a_key)
            return c, res

        _, all_final_states_indices = jax.lax.scan(
            scan_over_actions, None, (actions, keys)
        )
        # shape of all_final_states_indices: [num_actions, n_trajectories]

        num_categories = 2 * 3 * map_width * map_height
        state_one_hot = jax.nn.one_hot(all_final_states_indices, num_categories)

        # Compute marginal and conditionals
        p_s_plus_given_s_a = jnp.mean(state_one_hot, axis=1) # [num_actions, num_categories]
        p_s_plus_given_s = jnp.mean(p_s_plus_given_s_a, axis=0) # [num_categories]

        mi_terms = jnp.where(
            (p_s_plus_given_s_a > 0) & (p_s_plus_given_s > 0)[None, ...],
            p_s_plus_given_s_a * jnp.log2(p_s_plus_given_s_a / p_s_plus_given_s[None, ...]),
            0.0,
        )

        p_a_given_s = 1.0 / num_actions
        mi = p_a_given_s * jnp.sum(mi_terms)
        return carry_unused, mi

    # Scan sequentially over agents to prevent huge memory allocations
    agent_indices = jnp.arange(num_agents)
    keys = jax.random.split(key, num_agents)
    
    _, empowerment_array = jax.lax.scan(
        compute_for_agent, None, (agent_indices, keys)
    )
    # empowerment_array shape: [num_agents]
    
    # Return as list of arrays to match original calling signature
    return [empowerment_array[i] for i in range(num_agents)]
