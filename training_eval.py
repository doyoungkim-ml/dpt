"""
Evaluation helper module for training scripts.
Provides functions to run evaluations and log plots to wandb during training.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from collections import defaultdict

import wandb
from evals import eval_bandit, eval_linear_bandit
from utils import (
    build_bandit_data_filename,
    build_linear_bandit_data_filename,
    build_darkroom_data_filename,
    build_miniworld_data_filename,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# Global cache for non-learnable baseline results
_baseline_cache = {}


def _deploy_online_vec_with_state_tracking_darkroom(vec_env, controller, ctx_rollouts, H, horizon):
    """
    Wrapper around deploy_online_vec that also tracks visited states for darkroom.
    Returns (cumulative_means, state_visits) where state_visits is a dict mapping
    (env_idx, episode_idx) -> list of visited states (2D coordinates).
    
    For darkroom evaluation: runs ctx_rollouts separate episodes for each environment.
    Each rollout starts with empty context, accumulates context during the rollout (up to H steps),
    then resets for the next rollout.
    """
    from evals.eval_darkroom import deploy_online_vec
    from utils import convert_to_tensor
    
    num_envs = vec_env.num_envs
    state_visits = defaultdict(list)  # (env_idx, episode) -> list of states
    
    # Store step-level rewards across all rollouts
    # Shape: (num_envs, ctx_rollouts, horizon) - rewards at each step for each rollout
    step_rewards = []
    # Store states and actions for visualization
    # Shape: (num_envs, ctx_rollouts, horizon, 2) for states, (num_envs, ctx_rollouts, horizon, 5) for actions
    rollout_states = []
    rollout_actions = []
    
    # Run ctx_rollouts separate episodes - each rollout is independent with its own context
    for episode_idx in range(ctx_rollouts):
        # Each rollout starts with empty context
        # The context will accumulate during the rollout via deploy_online_vec
        # Use deploy_online_vec which handles context accumulation within a single rollout
        # We need to manually step through horizon steps to accumulate context
        
        # Start with empty context
        batch = {
            'context_states': torch.zeros((num_envs, 1, vec_env.state_dim)).float().to(device),
            'context_actions': torch.zeros((num_envs, 1, vec_env.action_dim)).float().to(device),
            'context_next_states': torch.zeros((num_envs, 1, vec_env.state_dim)).float().to(device),
            'context_rewards': torch.zeros((num_envs, 1, 1)).float().to(device),
        }
        controller.set_batch(batch)
        
        # Deploy for this rollout - context accumulates during the rollout
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(controller)
        
        # Track states for this episode - include both obs and next_obs to get all visited states
        states_lnr_np = states_lnr if isinstance(states_lnr, np.ndarray) else states_lnr.cpu().numpy()
        next_states_lnr_np = next_states_lnr if isinstance(next_states_lnr, np.ndarray) else next_states_lnr.cpu().numpy()
        actions_lnr_np = actions_lnr if isinstance(actions_lnr, np.ndarray) else actions_lnr.cpu().numpy()
        
        for env_idx in range(num_envs):
            # Combine obs and next_obs to get all visited states
            all_states = states_lnr_np[env_idx].tolist()
            # Add the final next_obs state if it's different from the last obs
            if len(next_states_lnr_np[env_idx]) > 0:
                final_next_state = next_states_lnr_np[env_idx][-1].tolist()
                if len(all_states) == 0 or all_states[-1] != final_next_state:
                    all_states.append(final_next_state)
            state_visits[(env_idx, episode_idx)].extend(all_states)
        
        # Store step-level rewards for this rollout
        # rewards_lnr has shape (num_envs, horizon) - rewards at each step
        step_rewards.append(rewards_lnr)
        # Store states and actions for visualization
        rollout_states.append(states_lnr_np)  # Shape: (num_envs, horizon, 2)
        rollout_actions.append(actions_lnr_np)  # Shape: (num_envs, horizon, 5)
    
    # Stack to get shape (num_envs, ctx_rollouts, horizon)
    step_rewards = np.stack(step_rewards, axis=1)
    rollout_states = np.stack(rollout_states, axis=1)  # Shape: (num_envs, ctx_rollouts, horizon, 2)
    rollout_actions = np.stack(rollout_actions, axis=1)  # Shape: (num_envs, ctx_rollouts, horizon, 5)
    
    # Average across rollouts and environments to get step-level performance
    # Shape: (horizon,) - average reward at each step across all environments and rollouts
    step_rewards_mean = np.mean(step_rewards, axis=(0, 1))  # Average over envs and rollouts
    step_rewards_sem = scipy.stats.sem(step_rewards.reshape(-1, horizon), axis=0)  # SEM across all (env, rollout) pairs
    
    # Also return cumulative means for backward compatibility (average across rollouts)
    cum_means = np.sum(step_rewards, axis=-1)  # Shape: (num_envs, ctx_rollouts)
    
    # Average step-level rewards across rollouts for each environment (for individual trajectories)
    # Shape: (num_envs, horizon) - average step reward for each environment across all rollouts
    step_rewards_per_env = np.mean(step_rewards, axis=1)
    
    return (np.stack(cum_means, axis=1), state_visits, step_rewards_mean, step_rewards_sem, step_rewards_per_env, 
            step_rewards, rollout_states, rollout_actions)


def _deploy_online_vec_with_state_tracking_miniworld(vec_env, controller, Heps, H, horizon, learner=False):
    """
    Wrapper around deploy_online_vec that also tracks visited positions for miniworld.
    Returns (cumulative_means, state_visits) where state_visits is a dict mapping
    (env_idx, episode_idx) -> list of visited positions (2D coordinates from agent.pos).
    """
    from evals.eval_miniworld import deploy_online_vec
    from utils import convert_to_tensor
    
    assert H % horizon == 0
    ctx_rollouts = H // horizon
    
    num_envs = vec_env.num_envs
    state_visits = defaultdict(list)  # (env_idx, episode) -> list of positions
    
    obs_dim = (3, 25, 25)
    state_dim = 2
    action_dim = 4
    context_images = torch.zeros(
        (num_envs, ctx_rollouts, horizon, *obs_dim)).float().to(device)
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, action_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)
    
    cum_means = []
    episode_idx = 0
    
    for i in range(ctx_rollouts):
        batch = {
            'context_images': context_images[:, :i].reshape(num_envs, -1, *obs_dim),
            'context_states': context_states[:, :i].reshape(num_envs, -1, state_dim),
            'context_actions': context_actions[:, :i].reshape(num_envs, -1, action_dim),
            'context_rewards': context_rewards[:, :i].reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        
        images_lnr, states_lnr, actions_lnr, _, rewards_lnr = vec_env.deploy_eval(controller)
        
        # Track positions: extract agent positions from environments
        for env_idx in range(num_envs):
            # Get positions by checking env state
            # We need to track positions during the episode, so we'll do it after each step
            # For now, let's collect positions from the environment directly
            env = vec_env.envs[env_idx]
            # Store initial position
            pos = env.agent.pos[[0, -1]]  # Get x and z coordinates
            state_visits[(env_idx, episode_idx)].append(pos.tolist())
        
        # Also track positions during the rollout by checking env states
        # We'll need to manually track during deployment
        # For simplicity, let's do a separate tracking pass
        if learner:
            context_images[:, i] = images_lnr
            context_states[:, i] = torch.tensor(states_lnr)
            context_actions[:, i] = torch.tensor(actions_lnr)
            context_rewards[:, i] = torch.tensor(rewards_lnr[:, :, None])
        
        cum_means.append(np.sum(rewards_lnr, axis=-1))
        episode_idx += 1
    
    # Continue tracking for remaining episodes
    # We'll need to manually track positions during each step
    # For now, let's use a simpler approach: track positions by manually stepping through
    # Actually, let's create a simpler version that tracks during a single deployment
    
    # For remaining episodes, we can't easily track without modifying the core deploy
    # Let's use a different approach: create a custom deployment that tracks positions
    
    return np.stack(cum_means, axis=1), state_visits


def _track_positions_during_deployment_miniworld(vec_env, controller, Heps, H, horizon):
    """
    Manually deploy and track agent positions for miniworld.
    Returns (cumulative_means, state_visits).
    This is a simplified version that tracks positions by manually stepping through episodes.
    """
    from utils import convert_to_tensor
    from skimage.transform import resize
    
    assert H % horizon == 0
    ctx_rollouts = H // horizon
    
    num_envs = vec_env.num_envs
    state_visits = defaultdict(list)  # (env_idx, episode) -> list of positions
    
    obs_dim = (3, 25, 25)
    state_dim = 2
    action_dim = 4
    context_images = torch.zeros(
        (num_envs, ctx_rollouts, horizon, *obs_dim)).float().to(device)
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, action_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)
    
    cum_means = []
    episode_idx = 0
    
    for i in range(ctx_rollouts):
        if i == 0:
            batch = {
                'context_images': torch.zeros((num_envs, 1, *obs_dim)).float().to(device),
                'context_states': torch.zeros((num_envs, 1, state_dim)).float().to(device),
                'context_actions': torch.zeros((num_envs, 1, action_dim)).float().to(device),
                'context_rewards': torch.zeros((num_envs, 1, 1)).float().to(device),
            }
        else:
            batch = {
                'context_images': context_images[:, :i].reshape(num_envs, -1, *obs_dim),
                'context_states': context_states[:, :i].reshape(num_envs, -1, state_dim),
                'context_actions': context_actions[:, :i].reshape(num_envs, -1, action_dim),
                'context_rewards': context_rewards[:, :i].reshape(num_envs, -1, 1),
            }
        controller.set_batch(batch)
        
        # Manual deployment to track positions
        images = vec_env.reset()
        pose = [env.agent.pos[[0, -1]] for env in vec_env.envs]
        angle = [env.agent.dir_vec[[0, -1]] for env in vec_env.envs]
        
        episode_positions = [[] for _ in range(num_envs)]
        episode_images = []
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        for step in range(horizon):
            # Record current positions
            for env_idx in range(num_envs):
                pos = vec_env.envs[env_idx].agent.pos[[0, -1]]
                episode_positions[env_idx].append(pos.tolist())
            
            action = controller.act(images, pose, angle)
            
            images_resized = [resize(img, (25, 25, 3), anti_aliasing=True) for img in images]
            image_tensor = torch.stack([controller.transform(img) for img in images_resized])
            
            episode_images.append(image_tensor)
            episode_states.append(np.array(angle))
            episode_actions.append(action)
            
            images, rew, done, _, _ = vec_env.step(np.argmax(action, axis=-1))
            pose = [env.agent.pos[[0, -1]] for env in vec_env.envs]
            angle = [env.agent.dir_vec[[0, -1]] for env in vec_env.envs]
            episode_rewards.append(rew)
            
            if all(done):
                break
        
        # Store positions for this episode
        for env_idx in range(num_envs):
            state_visits[(env_idx, episode_idx)].extend(episode_positions[env_idx])
        
        # Store context - stack along time dimension
        if episode_images:
            context_images[:, i, :len(episode_images)] = torch.stack(episode_images, dim=1)
            context_states[:, i, :len(episode_states)] = torch.tensor(np.stack(episode_states, axis=1))
            context_actions[:, i, :len(episode_actions)] = torch.tensor(np.stack(episode_actions, axis=1))
            context_rewards[:, i, :len(episode_rewards), 0] = torch.tensor(np.array(episode_rewards))
        
        cum_means.append(np.sum(episode_rewards, axis=0) if episode_rewards else np.zeros(num_envs))
        episode_idx += 1
    
    # Continue for remaining episodes
    for _ in range(ctx_rollouts, Heps):
        batch = {
            'context_images': context_images.reshape(num_envs, -1, *obs_dim),
            'context_states': context_states.reshape(num_envs, -1, state_dim),
            'context_actions': context_actions.reshape(num_envs, -1, action_dim),
            'context_rewards': context_rewards.reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        
        # Manual deployment
        images = vec_env.reset()
        pose = [env.agent.pos[[0, -1]] for env in vec_env.envs]
        angle = [env.agent.dir_vec[[0, -1]] for env in vec_env.envs]
        
        episode_positions = [[] for _ in range(num_envs)]
        episode_rewards = []
        
        for step in range(horizon):
            for env_idx in range(num_envs):
                pos = vec_env.envs[env_idx].agent.pos[[0, -1]]
                episode_positions[env_idx].append(pos.tolist())
            
            action = controller.act(images, pose, angle)
            images, rew, done, _, _ = vec_env.step(np.argmax(action, axis=-1))
            pose = [env.agent.pos[[0, -1]] for env in vec_env.envs]
            angle = [env.agent.dir_vec[[0, -1]] for env in vec_env.envs]
            episode_rewards.append(rew)
            
            if all(done):
                break
        
        for env_idx in range(num_envs):
            state_visits[(env_idx, episode_idx)].extend(episode_positions[env_idx])
        
        cum_means.append(np.sum(episode_rewards, axis=0) if episode_rewards else np.zeros(num_envs))
        episode_idx += 1
    
    return np.stack(cum_means, axis=1), state_visits


def _visualize_state_exploration(state_visits, grid_dim, n_eval, Heps, title_prefix="State Exploration", simple=False):
    """
    Create visualization of state space exploration over episodes.
    
    Args:
        state_visits: dict mapping (env_idx, episode_idx) -> list of 2D positions
        grid_dim: dimensions of the grid (dim, dim) for darkroom, or (max_x, max_z) for miniworld
        n_eval: number of environments
        Heps: number of episodes
        title_prefix: prefix for plot titles
        simple: if True, show only one figure with all visits aggregated
    
    Returns:
        matplotlib figure
    """
    # Create a grid to count visits
    if isinstance(grid_dim, (int, float)):
        grid_size = (int(grid_dim), int(grid_dim))
    else:
        grid_size = (int(grid_dim[0]), int(grid_dim[1]))
    
    # Aggregate visits across all environments and episodes
    visit_counts = np.zeros(grid_size)
    
    for (env_idx, episode_idx), positions in state_visits.items():
        for pos in positions:
            # Handle both integer and float positions
            x, y = pos[0], pos[1]
            # Round to nearest integer for grid indexing
            x = int(round(x))
            y = int(round(y))
            # Clip to grid bounds
            x = max(0, min(x, grid_size[0] - 1))
            y = max(0, min(y, grid_size[1] - 1))
            visit_counts[x, y] += 1
    
    if simple:
        # Simple single figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        im = ax.imshow(visit_counts.T, origin='lower', cmap='viridis', aspect='auto')
        ax.set_title(f'{title_prefix}')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        plt.colorbar(im, ax=ax, label='Visit count')
        plt.tight_layout()
        return fig
    else:
        # Original three-panel figure
        visit_counts_early = np.zeros(grid_size)  # First half of episodes
        visit_counts_late = np.zeros(grid_size)   # Second half of episodes
        
        mid_episode = Heps // 2
        
        for (env_idx, episode_idx), positions in state_visits.items():
            for pos in positions:
                x, y = pos[0], pos[1]
                x = int(round(x))
                y = int(round(y))
                x = max(0, min(x, grid_size[0] - 1))
                y = max(0, min(y, grid_size[1] - 1))
                if episode_idx < mid_episode:
                    visit_counts_early[x, y] += 1
                else:
                    visit_counts_late[x, y] += 1
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Overall exploration
        im1 = axes[0].imshow(visit_counts.T, origin='lower', cmap='viridis', aspect='auto')
        axes[0].set_title(f'{title_prefix}: All Episodes')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')
        plt.colorbar(im1, ax=axes[0], label='Visit count')
        
        # Early episodes
        im2 = axes[1].imshow(visit_counts_early.T, origin='lower', cmap='viridis', aspect='auto')
        axes[1].set_title(f'{title_prefix}: Early Episodes (0-{mid_episode-1})')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')
        plt.colorbar(im2, ax=axes[1], label='Visit count')
        
        # Late episodes
        im3 = axes[2].imshow(visit_counts_late.T, origin='lower', cmap='viridis', aspect='auto')
        axes[2].set_title(f'{title_prefix}: Late Episodes ({mid_episode}-{Heps-1})')
        axes[2].set_xlabel('X coordinate')
        axes[2].set_ylabel('Y coordinate')
        plt.colorbar(im3, ax=axes[2], label='Visit count')
        
        plt.tight_layout()
        return fig


def _visualize_rollout_path_single_ax(ax, states, actions, goal, dim, title="Rollout Path"):
    """
    Visualize a single rollout path through the 2D grid on a given axis.
    
    Args:
        ax: matplotlib axis to plot on
        states: Array of shape (horizon, 2) - states visited during rollout
        actions: Array of shape (horizon, 5) - actions taken during rollout
        goal: Tuple (gx, gy) - goal position
        dim: Grid dimension
        title: Plot title
    """
    # Create grid - all cells are light grey (value 0.5) for background
    grid = np.ones((dim, dim)) * 0.5  # Light grey background for all cells
    
    # Mark goal position with darker grey
    # Note: grid is transposed with .T when displayed, so we use (goal_y, goal_x) to match scatter coordinates
    goal_x, goal_y = int(goal[0]), int(goal[1])
    grid[goal_y, goal_x] = 0.8  # Goal has darker grey value (using transposed coordinates)
    
    # Plot the path
    states_array = np.array(states)
    if len(states_array.shape) == 2 and states_array.shape[1] == 2:
        # Plot path
        path_x = states_array[:, 0]
        path_y = states_array[:, 1]
        
        # Plot path with arrows showing movement direction
        for i in range(len(path_x) - 1):
            x1, y1 = int(path_x[i]), int(path_y[i])
            x2, y2 = int(path_x[i+1]), int(path_y[i+1])
            
            # Only draw arrow if there's actual movement
            if x1 != x2 or y1 != y2:
                # Draw arrow
                ax.arrow(y1, x1, y2-y1, x2-x1, head_width=0.2, head_length=0.2, 
                        fc='blue', ec='blue', alpha=0.6, length_includes_head=True)
        
        # Mark start position
        start_x, start_y = int(path_x[0]), int(path_y[0])
        ax.scatter([start_y], [start_x], color='green', s=200, marker='o', 
                  label='Start', zorder=5)
        
        # Mark goal position
        ax.scatter([goal_y], [goal_x], color='red', s=200, marker='*', 
                  label='Goal', zorder=5)
        
        # Mark visited states
        for i in range(len(path_x)):
            x, y = int(path_x[i]), int(path_y[i])
            if i == 0:
                continue  # Already marked as start
            elif x == goal_x and y == goal_y:
                continue  # Already marked as goal
            else:
                ax.scatter([y], [x], color='blue', s=30, alpha=0.5, zorder=3)
    
    # Show grid - light grey background for all cells, darker grey for goal
    ax.imshow(grid.T, origin='lower', cmap='Greys', alpha=0.3)
    ax.set_xlim(-0.5, dim - 0.5)
    ax.set_ylim(-0.5, dim - 0.5)
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Y coordinate')
    ax.set_ylabel('X coordinate')
    ax.set_title(title, fontsize=10)
    if ax == ax.get_figure().axes[0]:  # Only add legend to first subplot
        ax.legend(fontsize=8)


def _visualize_rollout_path(states, actions, goal, dim, title="Rollout Path"):
    """
    Visualize a single rollout path through the 2D grid.
    
    Args:
        states: Array of shape (horizon, 2) - states visited during rollout
        actions: Array of shape (horizon, 5) - actions taken during rollout
        goal: Tuple (gx, gy) - goal position
        dim: Grid dimension
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create grid
    grid = np.zeros((dim, dim))
    
    # Mark goal position
    goal_x, goal_y = int(goal[0]), int(goal[1])
    grid[goal_x, goal_y] = 2  # Goal has special value
    
    # Plot the path
    states_array = np.array(states)
    if len(states_array.shape) == 2 and states_array.shape[1] == 2:
        # Plot path
        path_x = states_array[:, 0]
        path_y = states_array[:, 1]
        
        # Plot path with arrows showing movement direction
        for i in range(len(path_x) - 1):
            x1, y1 = int(path_x[i]), int(path_y[i])
            x2, y2 = int(path_x[i+1]), int(path_y[i+1])
            
            # Draw arrow
            ax.arrow(y1, x1, y2-y1, x2-x1, head_width=0.2, head_length=0.2, 
                    fc='blue', ec='blue', alpha=0.6, length_includes_head=True)
        
        # Mark start position
        start_x, start_y = int(path_x[0]), int(path_y[0])
        ax.scatter([start_y], [start_x], color='green', s=200, marker='o', 
                  label='Start', zorder=5)
        
        # Mark goal position
        ax.scatter([goal_y], [goal_x], color='red', s=200, marker='*', 
                  label='Goal', zorder=5)
        
        # Mark visited states
        for i in range(len(path_x)):
            x, y = int(path_x[i]), int(path_y[i])
            if i == 0:
                continue  # Already marked as start
            elif x == goal_x and y == goal_y:
                continue  # Already marked as goal
            else:
                ax.scatter([y], [x], color='blue', s=50, alpha=0.5, zorder=3)
    
    # Show grid
    ax.imshow(grid.T, origin='lower', cmap='Greys', alpha=0.3)
    ax.set_xlim(-0.5, dim - 0.5)
    ax.set_ylim(-0.5, dim - 0.5)
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Y coordinate')
    ax.set_ylabel('X coordinate')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    return fig


def evaluate_bandit_model(model, config, args, eval_trajs, n_eval, horizon, var, bandit_type='uniform'):
    """
    Evaluate a bandit model and return both plot figures and raw data for wandb logging.
    Uses caching for non-learnable baselines.
    
    Returns:
        dict: Dictionary with keys 'online_fig', 'entropy_fig', 'crossentropy_fig'
              and their corresponding data
    """
    model.eval()
    
    results = {}
    
    # Create cache key for baselines
    cache_key = f'bandit_{n_eval}_{horizon}_{var}_{bandit_type}'
    
    # 1. Online evaluation: suboptimality and cumulative regret
    from ctrls.ctrl_bandit import (
        BanditTransformerController, OptPolicy, EmpMeanPolicy, 
        ThompsonSamplingPolicy, UCBPolicy
    )
    from envs.bandit_env import BanditEnvVec, BanditEnv
    from evals.eval_bandit import deploy_online_vec
    
    # Create environments
    envs = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj['means']
        env = BanditEnv(means, horizon, var=var)
        envs.append(env)
    
    vec_env = BanditEnvVec(envs)
    
    # Check cache for baselines or compute them
    if cache_key in _baseline_cache:
        print(f"Using cached baselines for {cache_key}")
        baseline_data = _baseline_cache[cache_key]
        all_means = baseline_data.copy()
    else:
        print(f"Computing baselines for {cache_key}")
        all_means = {}
        opt_controller = OptPolicy(envs, batch_size=len(envs))
        opt_cum_means = deploy_online_vec(vec_env, opt_controller, horizon).T
        all_means['opt'] = opt_cum_means
        
        emp_controller = EmpMeanPolicy(envs[0], online=True, batch_size=len(envs))
        emp_cum_means = deploy_online_vec(vec_env, emp_controller, horizon).T
        all_means['emp'] = emp_cum_means
        
        ucb_controller = UCBPolicy(envs[0], const=1.0, batch_size=len(envs))
        ucb_cum_means = deploy_online_vec(vec_env, ucb_controller, horizon).T
        all_means['ucb'] = ucb_cum_means
        
        thmp_controller = ThompsonSamplingPolicy(
            envs[0], std=var, sample=True, prior_mean=0.5, prior_var=1/12.0,
            warm_start=False, batch_size=len(envs)
        )
        thmp_cum_means = deploy_online_vec(vec_env, thmp_controller, horizon).T
        all_means['thmp'] = thmp_cum_means
        
        _baseline_cache[cache_key] = all_means
    
    # Evaluate learner (always fresh)
    lnr_controller = BanditTransformerController(model, sample=True, batch_size=len(envs))
    lnr_cum_means = deploy_online_vec(vec_env, lnr_controller, horizon).T
    all_means['lnr'] = lnr_cum_means
    
    # Calculate regrets
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}
    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    
    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}
    
    # Store raw data for wandb
    episode_indices = np.arange(horizon)
    results['online_data'] = {
        'suboptimality': {k: {'episode': episode_indices, 'mean': means[k], 'sem': sems[k]} 
                          for k in means.keys()},
        'cumulative_regret': {k: {'episode': episode_indices, 'mean': regret_means[k], 'sem': regret_sems[k]} 
                              for k in regret_means.keys() if k != 'opt'}
    }
    
    # Create online plots
    fig_online, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    baseline_colors = {
        'emp': 'tab:blue',
        'ucb': 'tab:orange',
        'thmp': 'black',
        'lcb': 'tab:red',
        'lnr': 'tab:red',
    }
    
    for key in means.keys():
        if key == 'opt':
            ax1.plot(means[key], label=key, linestyle='--', color='black', linewidth=2)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], 
                           means[key] + sems[key], alpha=0.2, color='black')
        elif key == 'lnr':
            ax1.plot(means[key], label=key, linewidth=2, color=baseline_colors[key])
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], 
                           means[key] + sems[key], alpha=0.2, color=baseline_colors[key])
        else:
            ax1.plot(means[key], label=key, linestyle='--', linewidth=2, 
                    color=baseline_colors.get(key, 'gray'))
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], 
                           means[key] + sems[key], alpha=0.2, color=baseline_colors.get(key, 'gray'))
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation: Suboptimality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key, linewidth=2, color=baseline_colors.get(key, 'gray'))
            ax2.fill_between(np.arange(horizon), regret_means[key] - regret_sems[key], 
                           regret_means[key] + regret_sems[key], alpha=0.2, color=baseline_colors.get(key, 'gray'))
    
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Online Evaluation: Cumulative Regret')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results['online_fig'] = fig_online
    
    # 4. Entropy and cross-entropy plots
    eval_config = {
        'horizon': horizon,
        'var': var,
        'n_eval': n_eval,
        'bandit_type': bandit_type,
    }
    
    # Use cache for baseline entropy computations
    entropy_cache_key = f'entropy_{cache_key}'
    if entropy_cache_key in _baseline_cache:
        print(f"Using cached entropy baselines for {entropy_cache_key}")
        entropy_results = _baseline_cache[entropy_cache_key].copy()
        # Recompute learner entropy
        lnr_entropy = eval_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=False)
        entropy_results['lnr'] = lnr_entropy['lnr']
    else:
        print(f"Computing entropy baselines for {entropy_cache_key}")
        entropy_results = eval_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=True)
        _baseline_cache[entropy_cache_key] = entropy_results.copy()
    
    # Store raw data for entropy plots
    episode_range = np.arange(len(entropy_results['lnr']['entropy_mean']))
    results['entropy_data'] = {
        'entropy': {k: {'episode': episode_range, 'mean': v['entropy_mean'], 'sem': v['entropy_sem']} 
                    for k, v in entropy_results.items()},
        'cross_entropy': {k: {'episode': episode_range, 'mean': v['cross_entropy_mean'], 'sem': v['cross_entropy_sem']} 
                          for k, v in entropy_results.items()}
    }
    
    fig_entropy, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    baseline_colors_ent = {
        'lnr': 'tab:red',
        'emp': 'tab:blue',
        'ucb': 'tab:orange',
        'thmp': 'black',
    }
    
    for method_name, method_data in entropy_results.items():
        color = baseline_colors_ent.get(method_name, 'gray')
        linestyle = '-' if method_name == 'lnr' else '--'
        ax1.plot(episode_range, method_data['entropy_mean'], label=method_name,
                linewidth=2, color=color, linestyle=linestyle)
        ax1.fill_between(episode_range, 
                        method_data['entropy_mean'] - method_data['entropy_sem'],
                        method_data['entropy_mean'] + method_data['entropy_sem'],
                        alpha=0.2, color=color)
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Entropy (nats)')
    ax1.set_title('Policy Entropy During Episodes')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    for method_name, method_data in entropy_results.items():
        color = baseline_colors_ent.get(method_name, 'gray')
        linestyle = '-' if method_name == 'lnr' else '--'
        ax2.plot(episode_range, method_data['cross_entropy_mean'], label=method_name,
                linewidth=2, color=color, linestyle=linestyle)
        ax2.fill_between(episode_range,
                        method_data['cross_entropy_mean'] - method_data['cross_entropy_sem'],
                        method_data['cross_entropy_mean'] + method_data['cross_entropy_sem'],
                        alpha=0.2, color=color)
    
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cross-Entropy (nats)')
    ax2.set_title('Cross-Entropy with Optimal Policy')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results['entropy_fig'] = fig_entropy
    results['crossentropy_fig'] = fig_entropy
    
    model.train()
    return results


def evaluate_linear_bandit_model(model, config, args, eval_trajs, n_eval, horizon, var):
    """Evaluate a linear bandit model and return plot figures with caching."""
    model.eval()
    
    results = {}
    
    # Create cache key for baselines
    cache_key = f'linear_bandit_{n_eval}_{horizon}_{var}'
    
    # Similar to bandit but for linear bandit
    from ctrls.ctrl_bandit import BanditTransformerController, OptPolicy, ThompsonSamplingPolicy, LinUCBPolicy
    from envs.bandit_env import BanditEnvVec, LinearBanditEnv
    from evals.eval_linear_bandit import deploy_online_vec
    
    envs = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
        envs.append(env)
    
    vec_env = BanditEnvVec(envs)
    
    # Check cache for baselines or compute them
    if cache_key in _baseline_cache:
        print(f"Using cached baselines for {cache_key}")
        baseline_data = _baseline_cache[cache_key]
        all_means = baseline_data.copy()
    else:
        print(f"Computing baselines for {cache_key}")
        all_means = {}
        opt_controller = OptPolicy(envs, batch_size=len(envs))
        opt_cum_means = deploy_online_vec(vec_env, opt_controller, horizon).T
        all_means['opt'] = opt_cum_means
        
        thmp_controller = ThompsonSamplingPolicy(
            envs[0], std=var, sample=True, prior_mean=0.0, prior_var=1.0,
            warm_start=False, batch_size=len(envs)
        )
        thmp_cum_means = deploy_online_vec(vec_env, thmp_controller, horizon).T
        all_means['thmp'] = thmp_cum_means
        
        linucb_controller = LinUCBPolicy(envs[0], const=1.0, batch_size=len(envs))
        linucb_cum_means = deploy_online_vec(vec_env, linucb_controller, horizon).T
        all_means['linucb'] = linucb_cum_means
        
        _baseline_cache[cache_key] = all_means
    
    # Evaluate learner (always fresh)
    lnr_controller = BanditTransformerController(model, sample=True, batch_size=len(envs))
    lnr_cum_means = deploy_online_vec(vec_env, lnr_controller, horizon).T
    all_means['lnr'] = lnr_cum_means
    
    all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}
    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    
    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}
    
    # Store raw data for wandb
    episode_indices = np.arange(horizon)
    results['online_data'] = {
        'suboptimality': {k: {'episode': episode_indices, 'mean': means[k], 'sem': sems[k]} 
                          for k in means.keys()},
        'cumulative_regret': {k: {'episode': episode_indices, 'mean': regret_means[k], 'sem': regret_sems[k]} 
                              for k in regret_means.keys() if k != 'opt'}
    }
    
    fig_online, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    baseline_colors = {
        'thmp': 'black',
        'linucb': 'tab:orange',
        'lnr': 'tab:red',
    }
    
    for key in means.keys():
        if key == 'opt':
            ax1.plot(means[key], label=key, linestyle='--', color='black', linewidth=2)
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], 
                           means[key] + sems[key], alpha=0.2, color='black')
        elif key == 'lnr':
            ax1.plot(means[key], label=key, linewidth=2, color=baseline_colors[key])
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], 
                           means[key] + sems[key], alpha=0.2, color=baseline_colors[key])
        else:
            ax1.plot(means[key], label=key, linestyle='--', linewidth=2, 
                    color=baseline_colors.get(key, 'gray'))
            ax1.fill_between(np.arange(horizon), means[key] - sems[key], 
                           means[key] + sems[key], alpha=0.2, color=baseline_colors.get(key, 'gray'))
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Suboptimality')
    ax1.set_title('Online Evaluation: Suboptimality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for key in regret_means.keys():
        if key != 'opt':
            ax2.plot(regret_means[key], label=key, linewidth=2, color=baseline_colors.get(key, 'gray'))
            ax2.fill_between(np.arange(horizon), regret_means[key] - regret_sems[key], 
                           regret_means[key] + regret_sems[key], alpha=0.2, color=baseline_colors.get(key, 'gray'))
    
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Online Evaluation: Cumulative Regret')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results['online_fig'] = fig_online
    
    # Entropy and cross-entropy plots
    eval_config = {
        'horizon': horizon,
        'var': var,
        'n_eval': n_eval,
    }
    
    # Use cache for baseline entropy computations
    entropy_cache_key = f'entropy_{cache_key}'
    if entropy_cache_key in _baseline_cache:
        print(f"Using cached entropy baselines for {entropy_cache_key}")
        entropy_results = _baseline_cache[entropy_cache_key].copy()
        # Recompute learner entropy
        lnr_entropy = eval_linear_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=False)
        entropy_results['lnr'] = lnr_entropy['lnr']
    else:
        print(f"Computing entropy baselines for {entropy_cache_key}")
        entropy_results = eval_linear_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=True)
        _baseline_cache[entropy_cache_key] = entropy_results.copy()
    
    # Store raw data for entropy plots
    episode_range = np.arange(len(entropy_results['lnr']['entropy_mean']))
    results['entropy_data'] = {
        'entropy': {k: {'episode': episode_range, 'mean': v['entropy_mean'], 'sem': v['entropy_sem']} 
                    for k, v in entropy_results.items()},
        'cross_entropy': {k: {'episode': episode_range, 'mean': v['cross_entropy_mean'], 'sem': v['cross_entropy_sem']} 
                          for k, v in entropy_results.items()}
    }
    
    fig_entropy, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    baseline_colors_ent = {
        'lnr': 'tab:red',
        'thmp': 'black',
        'linucb': 'tab:orange',
    }
    
    for method_name, method_data in entropy_results.items():
        color = baseline_colors_ent.get(method_name, 'gray')
        linestyle = '-' if method_name == 'lnr' else '--'
        ax1.plot(episode_range, method_data['entropy_mean'], label=method_name,
                linewidth=2, color=color, linestyle=linestyle)
        ax1.fill_between(episode_range, 
                        method_data['entropy_mean'] - method_data['entropy_sem'],
                        method_data['entropy_mean'] + method_data['entropy_sem'],
                        alpha=0.2, color=color)
    
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Entropy (nats)')
    ax1.set_title('Policy Entropy During Episodes')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    for method_name, method_data in entropy_results.items():
        color = baseline_colors_ent.get(method_name, 'gray')
        linestyle = '-' if method_name == 'lnr' else '--'
        ax2.plot(episode_range, method_data['cross_entropy_mean'], label=method_name,
                linewidth=2, color=color, linestyle=linestyle)
        ax2.fill_between(episode_range,
                        method_data['cross_entropy_mean'] - method_data['cross_entropy_sem'],
                        method_data['cross_entropy_mean'] + method_data['cross_entropy_sem'],
                        alpha=0.2, color=color)
    
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cross-Entropy (nats)')
    ax2.set_title('Cross-Entropy with Optimal Policy')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results['entropy_fig'] = fig_entropy
    results['crossentropy_fig'] = fig_entropy
    
    model.train()
    return results


def evaluate_darkroom_model(model, config, args, n_eval, horizon, dim, permuted=False):
    """Evaluate a darkroom model and return plot figures with data (online only)."""
    model.eval()
    
    results = {}
    
    # Darkroom evaluation parameters
    H = args.get('H', 100)  # Context window size (for reference, but not used in independent rollouts)
    horizon = args.get('H', 100)  # Fixed episode length (same as training)
    ctx_rollouts = args.get('ctx_rollouts', 40)  # Number of separate independent rollouts per environment
    # Note: Each rollout is independent - context starts empty for each rollout and accumulates during that rollout
    
    # Create cache key for baselines
    cache_key = f'darkroom_{n_eval}_{ctx_rollouts}_{H}_{horizon}_{dim}_{permuted}'
    
    # Create environments dynamically (like bandits)
    from ctrls.ctrl_darkroom import DarkroomOptPolicy, DarkroomTransformerController
    from envs.darkroom_env import DarkroomEnv, DarkroomEnvPermuted, DarkroomEnvVec
    from evals.eval_darkroom import deploy_online_vec
    
    envs = []
    for i_eval in range(n_eval):
        if permuted:
            env = DarkroomEnvPermuted(dim, i_eval, horizon)
        else:
            # Generate goal for evaluation (similar to eval_online.py)
            goals = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
            np.random.RandomState(seed=0).shuffle(goals)
            train_test_split = int(.8 * len(goals))
            eval_goals = goals[train_test_split:]  # Use held-out goals
            goal = eval_goals[i_eval % len(eval_goals)]
            env = DarkroomEnv(dim, goal, horizon)
        envs.append(env)
    
    vec_env = DarkroomEnvVec(envs)
    
    # Check cache for baselines or compute them
    if cache_key in _baseline_cache:
        print(f"Using cached baselines for {cache_key}")
        baseline_data = _baseline_cache[cache_key]
        lnr_means = baseline_data['lnr'].copy()
        state_visits = baseline_data.get('state_visits', {})
        step_rewards_mean = baseline_data.get('step_rewards_mean', None)
        step_rewards_sem = baseline_data.get('step_rewards_sem', None)
        if not state_visits or step_rewards_mean is None:
            # Need to recompute with state tracking
            print("Recomputing with state tracking...")
            lnr_controller = DarkroomTransformerController(model, batch_size=n_eval, sample=True)
            lnr_means, state_visits, step_rewards_mean, step_rewards_sem, step_rewards_per_env, step_rewards, rollout_states, rollout_actions = _deploy_online_vec_with_state_tracking_darkroom(
                vec_env, lnr_controller, ctx_rollouts, H, horizon)
            baseline_data['state_visits'] = state_visits
            baseline_data['step_rewards_mean'] = step_rewards_mean
            baseline_data['step_rewards_sem'] = step_rewards_sem
            baseline_data['step_rewards_per_env'] = step_rewards_per_env
            baseline_data['step_rewards'] = step_rewards
            baseline_data['rollout_states'] = rollout_states
            baseline_data['rollout_actions'] = rollout_actions
        else:
            step_rewards_per_env = baseline_data.get('step_rewards_per_env')
            step_rewards = baseline_data.get('step_rewards')
            rollout_states = baseline_data.get('rollout_states')
            rollout_actions = baseline_data.get('rollout_actions')
            if step_rewards_per_env is None or step_rewards is None:
                # Need to recompute
                print("Recomputing step rewards per env...")
                lnr_controller = DarkroomTransformerController(model, batch_size=n_eval, sample=True)
                _, _, _, _, step_rewards_per_env, step_rewards, rollout_states, rollout_actions = _deploy_online_vec_with_state_tracking_darkroom(
                    vec_env, lnr_controller, ctx_rollouts, H, horizon)
                baseline_data['step_rewards_per_env'] = step_rewards_per_env
                baseline_data['step_rewards'] = step_rewards
                baseline_data['rollout_states'] = rollout_states
                baseline_data['rollout_actions'] = rollout_actions
    else:
        print(f"Computing baselines for {cache_key}")
        # Get learner results with state tracking
        lnr_controller = DarkroomTransformerController(model, batch_size=n_eval, sample=True)
        lnr_means, state_visits, step_rewards_mean, step_rewards_sem, step_rewards_per_env, step_rewards, rollout_states, rollout_actions = _deploy_online_vec_with_state_tracking_darkroom(
            vec_env, lnr_controller, ctx_rollouts, H, horizon)
        _baseline_cache[cache_key] = {
            'lnr': lnr_means.copy(), 
            'state_visits': state_visits,
            'step_rewards_mean': step_rewards_mean,
            'step_rewards_sem': step_rewards_sem,
            'step_rewards_per_env': step_rewards_per_env,
            'step_rewards': step_rewards,
            'rollout_states': rollout_states,
            'rollout_actions': rollout_actions
        }
    
    # Calculate means (lnr_means has shape (n_eval, ctx_rollouts))
    num_episodes = ctx_rollouts  # ctx_rollouts separate episodes
    lnr_means_mean = np.mean(lnr_means, axis=0)
    lnr_means_sem = scipy.stats.sem(lnr_means, axis=0)
    
    # Store raw data for wandb
    episode_indices = np.arange(num_episodes)
    step_indices = np.arange(horizon)
    results['online_data'] = {
        'learner': {'episode': episode_indices, 'mean': lnr_means_mean, 'sem': lnr_means_sem},
        'learner_step': {'step': step_indices, 'mean': step_rewards_mean, 'sem': step_rewards_sem}
    }
    
    # Create online plot - step-level performance showing context accumulation
    fig_online, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot individual trajectories (first 10 environments, averaged across rollouts)
    # step_rewards_per_env has shape (n_eval, horizon) - step-level rewards for each environment
    for i in range(min(n_eval, 10)):
        ax.plot(step_rewards_per_env[i], color='blue', alpha=0.1)
    
    # Plot 5 sample individual rollouts (cumulative returns)
    # step_rewards has shape (n_eval, ctx_rollouts, horizon)
    # Sample 5 random (env, rollout) pairs
    np.random.seed(42)  # For reproducibility
    sample_indices = []
    for _ in range(min(5, n_eval * ctx_rollouts)):
        env_idx = np.random.randint(0, n_eval)
        rollout_idx = np.random.randint(0, ctx_rollouts)
        sample_indices.append((env_idx, rollout_idx))
    
    # Different colors for each rollout
    rollout_colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
    
    # Store actions for sample rollouts for wandb logging
    sample_rollout_actions = []
    
    # Plot cumulative returns for sample rollouts
    # Add small offset to avoid log(0) issues
    log_offset = 0.1
    
    for i, (env_idx, rollout_idx) in enumerate(sample_indices):
        # Calculate cumulative return for this rollout
        rollout_rewards = step_rewards[env_idx, rollout_idx, :]  # Shape: (horizon,)
        cumulative_returns = np.cumsum(rollout_rewards)
        color = rollout_colors[i % len(rollout_colors)]
        
        # Add offset for log scale
        cumulative_returns_log = cumulative_returns + log_offset
        ax.plot(cumulative_returns_log, color=color, alpha=0.6, linewidth=1.5, 
               linestyle='--', label=f'Rollout {i+1}' if i < 5 else '')
        
        # Store actions for this rollout
        if rollout_actions is not None:
            actions = rollout_actions[env_idx, rollout_idx, :, :]  # Shape: (horizon, 5)
            # Convert one-hot actions to action indices
            action_indices = np.argmax(actions, axis=-1)  # Shape: (horizon,)
            sample_rollout_actions.append({
                'env_idx': int(env_idx),
                'rollout_idx': int(rollout_idx),
                'actions': action_indices.tolist(),
                'cumulative_return': float(np.sum(rollout_rewards))
            })
    
    # Plot means with error bars - step-level performance (cumulative)
    cumulative_returns_mean = np.cumsum(step_rewards_mean)
    # Add offset for log scale
    cumulative_returns_mean_log = cumulative_returns_mean + log_offset
    
    ax.plot(cumulative_returns_mean_log, label='Learner (mean cumulative)', color='blue', linewidth=2)
    
    # Compute cumulative SEM (simplified - actual would need proper propagation)
    cumulative_sem = np.cumsum(step_rewards_sem)
    ax.fill_between(step_indices, 
                    cumulative_returns_mean_log - cumulative_sem, 
                    cumulative_returns_mean_log + cumulative_sem, 
                    alpha=0.2, color='blue')
    
    ax.set_xlabel('Step (Context Accumulation)')
    ax.set_ylabel('Cumulative Return (log scale, offset=0.1)')
    ax.set_yscale('log')  # Use logarithmic scale
    ax.set_title(f'Step-Level Performance During Context Accumulation (n={n_eval} envs, {ctx_rollouts} rollouts)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results['online_fig'] = fig_online
    
    # Create path visualizations for sample rollouts
    if rollout_states is not None and rollout_actions is not None and len(sample_indices) > 0:
        # Create a figure with 5 subplots showing sample rollout paths
        num_samples = min(5, len(sample_indices))
        fig_paths, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        
        for idx, (env_idx, rollout_idx) in enumerate(sample_indices[:num_samples]):
            ax = axes[idx]
            # Get states and goal for this rollout
            states = rollout_states[env_idx, rollout_idx, :, :]  # Shape: (horizon, 2)
            actions = rollout_actions[env_idx, rollout_idx, :, :]  # Shape: (horizon, 5)
            
            # Get goal for this environment
            if permuted:
                goal = np.array([dim - 1, dim - 1])
            else:
                goal = envs[env_idx].goal
            
            # Visualize path
            _visualize_rollout_path_single_ax(ax, states, actions, goal, dim, 
                                             f'Rollout {idx+1} (Env {env_idx}, Rollout {rollout_idx})')
        
        plt.tight_layout()
        results['path_fig'] = fig_paths
    
    # Store sample rollout actions for wandb logging
    if sample_rollout_actions:
        results['sample_rollout_actions'] = sample_rollout_actions
    
    # Create state exploration visualization - simple single figure showing exploration
    if state_visits:
        fig_exploration = _visualize_state_exploration(
            state_visits, dim, n_eval, num_episodes, 
            title_prefix="Darkroom State Exploration", simple=True)
        results['exploration_fig'] = fig_exploration
    
    model.train()
    return results


def evaluate_miniworld_model(model, config, args, n_eval, horizon):
    """Evaluate a miniworld model and return plot figures with data (online only)."""
    model.eval()
    
    results = {}
    
    # Miniworld evaluation parameters
    Heps = 40
    H = args.get('H', 100)
    
    # Create environments
    from evals import eval_miniworld
    import gymnasium as gym
    import miniworld
    from ctrls.ctrl_miniworld import MiniworldOptPolicy, MiniworldTransformerController, MiniworldRandPolicy
    from envs.miniworld_env import MiniworldEnvVec
    from evals.eval_miniworld import deploy_online_vec
    from utils import convert_to_tensor
    
    envs = []
    for i_eval in range(n_eval):
        env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
        env.set_task(env_id=8000 + i_eval)
        envs.append(env)
    
    vec_env = MiniworldEnvVec(envs)
    
    # Learner evaluation with state tracking
    lnr_controller = MiniworldTransformerController(model, batch_size=n_eval, sample=True)
    lnr_means, state_visits = _track_positions_during_deployment_miniworld(
        vec_env, lnr_controller, Heps, H, horizon)
    
    # Optimal policy (without tracking for now, just for comparison)
    opt_controller = MiniworldOptPolicy(vec_env, batch_size=n_eval)
    from evals.eval_miniworld import deploy_online_vec
    opt_means = deploy_online_vec(vec_env, opt_controller, 1, H, horizon)
    opt_means = np.repeat(opt_means, Heps, axis=-1)
    
    # Random policy (without tracking for now)
    rand_controller = MiniworldRandPolicy(vec_env, batch_size=n_eval)
    rand_means = deploy_online_vec(vec_env, rand_controller, Heps, H, horizon)
    
    lnr_means_mean = np.mean(lnr_means, axis=0)
    lnr_means_sem = scipy.stats.sem(lnr_means, axis=0)
    opt_means_mean = np.mean(opt_means, axis=0)
    opt_means_sem = scipy.stats.sem(opt_means, axis=0)
    rand_means_mean = np.mean(rand_means, axis=0)
    rand_means_sem = scipy.stats.sem(rand_means, axis=0)
    
    # Store raw data for wandb
    episode_indices = np.arange(Heps)
    results['online_data'] = {
        'learner': {'episode': episode_indices, 'mean': lnr_means_mean, 'sem': lnr_means_sem},
        'optimal': {'episode': episode_indices, 'mean': opt_means_mean, 'sem': opt_means_sem},
        'random': {'episode': episode_indices, 'mean': rand_means_mean, 'sem': rand_means_sem}
    }
    
    # Create online plot
    fig_online, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot individual trajectories
    for i in range(min(n_eval, 10)):
        ax.plot(lnr_means[i], color='blue', alpha=0.1)
        ax.plot(opt_means[i], color='green', alpha=0.1)
        ax.plot(rand_means[i], color='orange', alpha=0.1)
    
    # Plot means with error bars
    ax.plot(lnr_means_mean, label='LNR', color='blue', linewidth=2)
    ax.fill_between(episode_indices, lnr_means_mean - lnr_means_sem, 
                    lnr_means_mean + lnr_means_sem, alpha=0.2, color='blue')
    
    ax.plot(opt_means_mean, label='Optimal', color='green', linewidth=2)
    ax.fill_between(episode_indices, opt_means_mean - opt_means_sem,
                    opt_means_mean + opt_means_sem, alpha=0.2, color='green')
    
    ax.plot(rand_means_mean, label='Random', color='orange', linewidth=2)
    ax.fill_between(episode_indices, rand_means_mean - rand_means_sem,
                    rand_means_mean + rand_means_sem, alpha=0.2, color='orange')
    
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Reward')
    ax.set_title(f'Online Evaluation on {n_eval} Envs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    results['online_fig'] = fig_online
    
    # Create state exploration visualization
    if state_visits:
        # Determine grid bounds from visited positions
        all_positions = []
        for positions in state_visits.values():
            all_positions.extend(positions)
        if all_positions:
            all_positions = np.array(all_positions)
            max_x = int(np.ceil(all_positions[:, 0].max())) + 1
            max_z = int(np.ceil(all_positions[:, 1].max())) + 1
            min_x = int(np.floor(all_positions[:, 0].min()))
            min_z = int(np.floor(all_positions[:, 1].min()))
            # Use grid dimensions with some padding
            grid_dim = (max_x - min_x + 2, max_z - min_z + 2)
            # Adjust positions to be relative to grid
            adjusted_state_visits = defaultdict(list)
            for (env_idx, ep_idx), positions in state_visits.items():
                adjusted_positions = [[int(round(pos[0])) - min_x + 1, int(round(pos[1])) - min_z + 1] 
                                     for pos in positions]
                adjusted_state_visits[(env_idx, ep_idx)] = adjusted_positions
            fig_exploration = _visualize_state_exploration(
                adjusted_state_visits, grid_dim, n_eval, Heps, 
                title_prefix="Miniworld State Exploration")
            results['exploration_fig'] = fig_exploration
    
    model.train()
    return results


def load_eval_data(args, n_eval, dataset_config, envname, var=None, cov=None):
    """
    Load evaluation trajectories.
    
    Args:
        args: Arguments dict
        n_eval: Number of evaluation environments
        dataset_config: Dataset configuration dict
        envname: Environment name
        var: Override variance (default: from args)
        cov: Override covariance (default: from args)
    """
    # Use provided values or fall back to args
    eval_var = var if var is not None else args.get('var', 0.0)
    eval_cov = cov if cov is not None else args.get('cov', 0.0)
    
    if envname in ['bandit', 'bandit_bernoulli']:
        dataset_config.update({'var': eval_var, 'cov': eval_cov, 'type': 'uniform'})
        eval_filepath = build_bandit_data_filename(envname, n_eval, dataset_config, mode=2)
    elif envname == 'linear_bandit':
        dataset_config.update({'lin_d': args.get('lin_d', 2), 'var': eval_var, 'cov': eval_cov})
        eval_filepath = build_linear_bandit_data_filename(envname, n_eval, dataset_config, mode=2)
    elif envname.startswith('darkroom'):
        dataset_config.update({'rollin_type': 'uniform'})
        eval_filepath = build_darkroom_data_filename(envname, n_eval, dataset_config, mode=2)
    elif envname == 'miniworld':
        dataset_config.update({'rollin_type': 'uniform'})
        eval_filepath = build_miniworld_data_filename(envname, 0, n_eval, dataset_config, mode=2)
    else:
        raise ValueError(f'Environment {envname} not supported')
    
    with open(eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)
    
    return eval_trajs


def _collect_rollout_data(model, envs, eval_trajs, envname, horizon, var):
    """
    Collect rollout data for logging inference trajectories.
    Returns a list of dicts with rollout information for each environment.
    """
    from ctrls.ctrl_bandit import OptPolicy, BanditTransformerController
    from envs.bandit_env import BanditEnvVec
    
    # Import the correct deploy function based on environment
    if envname == 'linear_bandit':
        from evals.eval_linear_bandit import deploy_online_vec
    else:
        from evals.eval_bandit import deploy_online_vec
    
    vec_env = BanditEnvVec(envs)
    
    # Get optimal rollout for reference
    opt_controller = OptPolicy(envs, batch_size=len(envs))
    opt_means = deploy_online_vec(vec_env, opt_controller, horizon).T
    
    # Get learner rollout
    lnr_controller = BanditTransformerController(model, sample=True, batch_size=len(envs))
    lnr_means = deploy_online_vec(vec_env, lnr_controller, horizon).T
    
    rollouts = []
    for env_idx in range(min(len(envs), 5)):  # Log first 5 environments
        traj = eval_trajs[env_idx]
        rollouts.append({
            'env_id': env_idx,
            'optimal_reward': float(np.mean(opt_means[env_idx])),
            'learner_reward': float(np.mean(lnr_means[env_idx])),
            'true_means': traj['means'].tolist() if envname in ['bandit', 'bandit_bernoulli'] else None,
        })
    
    return rollouts


def log_evaluation_plots_to_wandb(model, config, args, envname, epoch):
    """
    Main function to run evaluation and log plots to wandb.
    
    Simplified version that only logs online evaluation metrics and entropy plots.
    
    Args:
        model: The trained model to evaluate
        config: Model configuration dict
        args: Training arguments dict
        envname: Environment name (e.g., 'bandit', 'linear_bandit')
        epoch: Current epoch number
    """
    print(f"Running evaluation for epoch {epoch}...")
    
    # Get evaluation parameters
    n_eval = args.get('n_eval', 100)
    horizon = args.get('hor', args.get('H', 100))
    var = args.get('var', 0.0)
    cov = args.get('cov', 0.0)
    
    print(f"  Eval params: n_eval={n_eval}, horizon={horizon}, var={var}, cov={cov}")
    
    dataset_config = {'horizon': horizon, 'dim': args.get('dim', 10)}
    
    try:
        if envname in ['bandit', 'bandit_bernoulli', 'linear_bandit']:
            # Load evaluation trajectories for bandits (they need trajectory data)
            eval_trajs = load_eval_data(args, n_eval, dataset_config, envname, var=var, cov=cov)
            n_eval = min(n_eval, len(eval_trajs))
            
            if envname in ['bandit', 'bandit_bernoulli']:
                results = evaluate_bandit_model(model, config, args, eval_trajs, n_eval, horizon, var, 
                                              bandit_type='uniform' if envname == 'bandit' else 'bernoulli')
            elif envname == 'linear_bandit':
                results = evaluate_linear_bandit_model(model, config, args, eval_trajs, n_eval, horizon, var)
        elif envname.startswith('darkroom'):
            # Generate environments dynamically (no need to load eval data)
            dim = args.get('dim', 10)
            permuted = envname == 'darkroom_permuted'
            results = evaluate_darkroom_model(model, config, args, n_eval, horizon, dim, permuted=permuted)
        elif envname == 'miniworld':
            # Generate environments dynamically (no need to load eval data)
            results = evaluate_miniworld_model(model, config, args, n_eval, horizon)
        else:
            print(f"Evaluation not implemented for {envname}")
            return
        
        # Log plots to wandb
        # 1. Online image with suboptimality and cumulative regret
        if 'online_fig' in results:
            wandb.log({f"eval/online_image": wandb.Image(results['online_fig'])}, commit=False)
            plt.close(results['online_fig'])
        
        # 2. Entropy plots
        if 'entropy_fig' in results:
            wandb.log({f"eval/entropy_crossentropy_image": wandb.Image(results['entropy_fig'])}, commit=False)
            plt.close(results['entropy_fig'])
        
        # 3. State exploration visualization (for darkroom and miniworld)
        if 'exploration_fig' in results:
            wandb.log({f"eval/state_exploration": wandb.Image(results['exploration_fig'])}, commit=False)
            plt.close(results['exploration_fig'])
        
        # 4. Path visualization (for darkroom)
        if 'path_fig' in results:
            wandb.log({f"eval/path_visualization": wandb.Image(results['path_fig'])}, commit=False)
            plt.close(results['path_fig'])
        
        # 5. Log sample rollout actions (for darkroom)
        if envname.startswith('darkroom') and 'sample_rollout_actions' in results:
            sample_actions = results['sample_rollout_actions']
            if sample_actions:
                # Create a table with actions for each sample rollout
                columns = ['env_idx', 'rollout_idx', 'cumulative_return', 'actions']
                table_data = []
                for action_data in sample_actions:
                    # Convert actions list to string for display
                    actions_str = ', '.join(map(str, action_data['actions']))
                    table_data.append([
                        action_data['env_idx'],
                        action_data['rollout_idx'],
                        f"{action_data['cumulative_return']:.2f}",
                        actions_str
                    ])
                wandb.log({f"eval/sample_rollout_actions": wandb.Table(columns=columns, data=table_data)}, commit=False)
        
        # 6. Log rollout inference data for a few environments (only for bandits)
        if envname in ['bandit', 'bandit_bernoulli', 'linear_bandit']:
            # Create environments for rollout collection
            from envs.bandit_env import BanditEnv, LinearBanditEnv
            envs = []
            for i_eval in range(n_eval):
                traj = eval_trajs[i_eval]
                if envname == 'linear_bandit':
                    env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
                else:
                    env = BanditEnv(traj['means'], horizon, var=var)
                envs.append(env)
            
            rollouts = _collect_rollout_data(model, envs, eval_trajs, envname, horizon, var)
            if rollouts:
                # Convert rollouts to table format
                columns = list(rollouts[0].keys())
                table_data = [[row.get(col) for col in columns] for row in rollouts]
                wandb.log({f"eval/rollouts": wandb.Table(columns=columns, data=table_data)}, commit=False)
        
        print("Evaluation plots logged to wandb")
        
    except Exception as e:
        print(f"Warning: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
