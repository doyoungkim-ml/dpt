import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

import os
# Fix OpenBLAS deadlock by setting environment variables before any other imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import time
import yaml
import pickle
import tempfile
from pathlib import Path

import wandb
import torch
from torchvision.transforms import transforms

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.stats

import common_args
from dataset import Dataset, ImageDataset
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
    worker_init_fn,
)

from envs import bandit_env, darkroom_env
from ctrls.ctrl_bandit import BanditTransformerController, ThompsonSamplingPolicy
from ctrls.ctrl_darkroom import DarkroomTransformerController
from evals import eval_bandit

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

from training_eval import log_evaluation_plots_to_wandb
EVAL_ENABLED = True


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
    
    goal_x, goal_y = int(goal[0]), int(goal[1])
    
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
                ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.2, head_length=0.2, 
                        fc='blue', ec='blue', alpha=0.6, length_includes_head=True)
        
        # Mark start position
        start_x, start_y = int(path_x[0]), int(path_y[0])
        ax.scatter([start_x], [start_y], color='green', s=200, marker='o', 
                  label='Start', zorder=5)
        
        # Mark goal position
        ax.scatter([goal_x], [goal_y], color='red', s=200, marker='*', 
                  label='Goal', zorder=5)
        
        # Mark visited states
        for i in range(len(path_x)):
            x, y = int(path_x[i]), int(path_y[i])
            if i == 0:
                continue  # Already marked as start
            elif x == goal_x and y == goal_y:
                continue  # Already marked as goal
            else:
                ax.scatter([x], [y], color='blue', s=30, alpha=0.5, zorder=3)
    
    # Show grid - light grey background for all cells
    ax.imshow(grid, origin='lower', cmap='Greys', alpha=0.3)
    ax.set_xlim(-0.5, dim - 0.5)
    ax.set_ylim(-0.5, dim - 0.5)
    ax.set_xticks(np.arange(dim))
    ax.set_yticks(np.arange(dim))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title, fontsize=10)
    if ax == ax.get_figure().axes[0]:  # Only add legend to first subplot
        ax.legend(fontsize=8)


def collect_rollout_data_bandit(model, envs, horizon, n_hists, n_samples, sample_action=True):
    """
    Collect rollout data using the current model for bandit environments.
    Uses the same vectorized rollout approach as evaluation code.
    
    Args:
        model: The current model (always required, even if randomly initialized)
        envs: List of bandit environments
        horizon: Context horizon
        n_hists: Number of histories per environment
        n_samples: Number of samples per history
        
    Returns:
        List of trajectories in the format expected by Dataset
    """
    trajs = []
    
    # Set model to eval mode for inference
    model.eval()
    
    # Create vectorized environment
    vec_env = bandit_env.BanditEnvVec(envs)
    num_envs = len(envs)
    
    # Create controller with batch_size matching number of environments
    controller = BanditTransformerController(model, sample=sample_action, batch_size=num_envs)
    
    # Note: deploy_online_vec will print its own progress messages
    # Collect n_hists histories for all environments at once
    for j in range(n_hists):
        if n_hists > 1:
            print(f"  History {j+1}/{n_hists}...")
            import sys
            sys.stdout.flush()
        
        # Use deploy_online_vec to collect rollouts for all environments at once
        _, meta = eval_bandit.deploy_online_vec(vec_env, controller, horizon, include_meta=True)
        
        context_states = meta['context_states']  # Shape: (num_envs, horizon, dx)
        context_actions = meta['context_actions']  # Shape: (num_envs, horizon, du)
        context_rewards = meta['context_rewards'][:, :, 0]  # Shape: (num_envs, horizon)
        
        # Create trajectories for each environment
        for env_idx, env in enumerate(envs):
            # Create n_samples trajectories with the same context
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a
                
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states[env_idx],  # Shape: (horizon, dx)
                    'context_actions': context_actions[env_idx],  # Shape: (horizon, du)
                    'context_rewards': context_rewards[env_idx],  # Shape: (horizon,)
                    'means': env.means,
                }
                trajs.append(traj)
    
    print(f"\n  Collected {len(trajs)} trajectories total")
    import sys
    sys.stdout.flush()
    return trajs


def collect_rollout_data_linear_bandit(model, envs, horizon, n_hists, n_samples, arms, sample_action=True):
    """
    Collect rollout data using the current model for linear bandit environments.
    Uses the same vectorized rollout approach as evaluation code.
    
    Args:
        model: The current model (always required, even if randomly initialized)
        envs: List of linear bandit environments
        horizon: Context horizon
        n_hists: Number of histories per environment
        n_samples: Number of samples per history
        arms: Fixed arm features for linear bandits
        sample_action: Whether to sample actions from the model
        
    Returns:
        List of trajectories in the format expected by Dataset
    """
    trajs = []
    
    # Set model to eval mode for inference
    model.eval()
    
    # Create vectorized environment
    vec_env = bandit_env.BanditEnvVec(envs)
    num_envs = len(envs)
    
    # Create controller with batch_size matching number of environments
    controller = BanditTransformerController(model, sample=sample_action, batch_size=num_envs)
    
    # Collect n_hists histories for all environments at once
    for j in range(n_hists):
        if n_hists > 1:
            print(f"  History {j+1}/{n_hists}...")
            import sys
            sys.stdout.flush()
        
        # Use deploy_online_vec to collect rollouts for all environments at once
        # Note: deploy_online_vec will print its own progress messages
        _, meta = eval_bandit.deploy_online_vec(vec_env, controller, horizon, include_meta=True)
        
        context_states = meta['context_states']  # Shape: (num_envs, horizon, dx)
        context_actions = meta['context_actions']  # Shape: (num_envs, horizon, du)
        context_rewards = meta['context_rewards'][:, :, 0]  # Shape: (num_envs, horizon)
        
        # Create trajectories for each environment
        for env_idx, env in enumerate(envs):
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a
                
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states[env_idx],  # Shape: (horizon, dx)
                    'context_actions': context_actions[env_idx],  # Shape: (horizon, du)
                    'context_rewards': context_rewards[env_idx],  # Shape: (horizon,)
                    'means': env.means,
                    'arms': arms,
                    'theta': env.theta,
                    'var': env.var,
                }
                trajs.append(traj)
    
    print(f"\n  Collected {len(trajs)} trajectories total")
    import sys
    sys.stdout.flush()
    return trajs


def collect_rollout_data_darkroom(model, envs, horizon, n_hists, n_samples, sample_action=True):
    """
    Collect rollout data using the current model for darkroom environments.
    Uses vectorized rollout approach similar to evaluation code.
    
    Args:
        model: The current model (always required, even if randomly initialized)
        envs: List of darkroom environments
        horizon: Context horizon
        n_hists: Number of histories per environment
        n_samples: Number of samples per history
        sample_action: Whether to sample actions from the model
        
    Returns:
        List of trajectories in the format expected by Dataset
        Also returns rollout_data dict with rewards and states for visualization (if n_hists == 1)
    """
    from envs.darkroom_env import DarkroomEnvVec
    from utils import convert_to_tensor
    
    trajs = []
    
    # Set model to eval mode for inference
    model.eval()
    
    # Create vectorized environment
    vec_env = DarkroomEnvVec(envs)
    num_envs = len(envs)
    
    # Create controller with batch_size matching number of environments
    controller = DarkroomTransformerController(model, batch_size=num_envs, sample=sample_action)
    
    # Store rollout data for visualization (only for first history if n_hists == 1)
    rollout_data = None
    if n_hists == 1:
        # Store step-level rewards and states for visualization
        # Shape: (num_envs, horizon) for rewards, (num_envs, horizon, 2) for states
        rollout_rewards = []
        rollout_states = []
        rollout_actions = []
    
    # Collect n_hists histories for all environments at once
    for j in range(n_hists):
        if n_hists > 1:
            print(f"  History {j+1}/{n_hists}...")
            import sys
            sys.stdout.flush()
        
        # Collect rollout data using vectorized approach
        # Use torch tensors to avoid repeated conversions
        context_states_t = torch.zeros((num_envs, horizon, vec_env.state_dim)).float().to(device)
        context_actions_t = torch.zeros((num_envs, horizon, vec_env.action_dim)).float().to(device)
        context_next_states_t = torch.zeros((num_envs, horizon, vec_env.state_dim)).float().to(device)
        context_rewards_t = torch.zeros((num_envs, horizon, 1)).float().to(device)
        
        # Reset all environments
        states = vec_env.reset()
        states_t = torch.tensor(np.array(states)).float().to(device)  # (num_envs, state_dim)
        
        with torch.no_grad():  # No gradients needed during rollout
            step_times = []
            for h in range(horizon):
                step_start = time.time()
                if h % max(1, horizon // 10) == 0 or h == horizon - 1:
                    avg_time = np.mean(step_times) if step_times else 0
                    print(f"    Step {h+1}/{horizon} (avg: {avg_time*1000:.1f}ms/step)...", end='\r')
                    import sys
                    sys.stdout.flush()
                
                # Prepare batch with accumulated context
                if h == 0:
                    # Empty context - create dummy context
                    batch = {
                        'context_states': torch.zeros((num_envs, 1, vec_env.state_dim)).float().to(device),
                        'context_actions': torch.zeros((num_envs, 1, vec_env.action_dim)).float().to(device),
                        'context_next_states': torch.zeros((num_envs, 1, vec_env.state_dim)).float().to(device),
                        'context_rewards': torch.zeros((num_envs, 1, 1)).float().to(device),
                    }
                else:
                    batch = {
                        'context_states': context_states_t[:, :h, :],
                        'context_actions': context_actions_t[:, :h, :],
                        'context_next_states': context_next_states_t[:, :h, :],
                        'context_rewards': context_rewards_t[:, :h, :],
                    }
                controller.set_batch(batch)
                
                # Get actions from controller
                # Controller.act() expects a list, but we can pass numpy array directly
                # Convert tensor to numpy once
                states_np = states_t.cpu().numpy()
                # Controller will convert to tensor internally, but this avoids list conversion overhead
                states_list = [states_np[i] for i in range(num_envs)]
                actions = controller.act(states_list)  # Returns (num_envs, action_dim) array
                
                # Convert to list format for step() - actions is already numpy array
                actions_list = [actions[i] for i in range(num_envs)]
                
                # Execute in environments
                next_states_list, rewards_list, dones, _ = vec_env.step(actions_list)
                
                # Store transitions (convert to tensor once, batch conversion)
                next_states_t = torch.tensor(np.array(next_states_list), dtype=torch.float32).to(device)
                rewards_t = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(device).unsqueeze(1)
                actions_t = torch.tensor(actions, dtype=torch.float32).to(device)
                
                context_states_t[:, h, :] = states_t
                context_actions_t[:, h, :] = actions_t
                context_next_states_t[:, h, :] = next_states_t
                context_rewards_t[:, h, :] = rewards_t
                
                states_t = next_states_t
                states = next_states_list
                
                step_times.append(time.time() - step_start)
        
        # Convert to numpy at the end
        context_states = context_states_t.cpu().numpy()
        context_actions = context_actions_t.cpu().numpy()
        context_next_states = context_next_states_t.cpu().numpy()
        context_rewards = context_rewards_t.cpu().numpy()[:, :, 0]  # Remove last dimension
        
        # Store rollout data for visualization (only for first history)
        if n_hists == 1 and j == 0:
            rollout_rewards.append(context_rewards)  # Shape: (num_envs, horizon)
            rollout_states.append(context_states)  # Shape: (num_envs, horizon, 2)
            rollout_actions.append(context_actions)  # Shape: (num_envs, horizon, 5)
        
        # Create trajectories for each environment
        for env_idx, env in enumerate(envs):
            for k in range(n_samples):
                # For MDPs, compute optimal action for each state in the context sequence
                # This is state-dependent: each state gets its own optimal action
                context_states_env = context_states[env_idx]  # Shape: (horizon, state_dim)
                optimal_actions = np.array([
                    env.opt_action(context_states_env[h]) 
                    for h in range(horizon)
                ])  # Shape: (horizon, action_dim)
                
                # Also keep query_state for backward compatibility (though not used in training)
                query_state = env.sample_state()
                query_optimal_action = env.opt_action(query_state)
                
                # Debug: Log first few trajectories to verify state-dependence
                if len(trajs) < 5:
                    # Map action indices to names for darkroom
                    action_names = {0: "right", 1: "left", 2: "up", 3: "down", 4: "stay"}
                    
                    # Format context states and actions
                    context_strs = []
                    context_actions_env = context_actions[env_idx]  # Shape: (horizon, action_dim)
                    for h in range(min(horizon, 10)):  # Show first 10 context states
                        state = context_states_env[h]
                        policy_action_idx = np.argmax(context_actions_env[h])  # Action policy took
                        optimal_action_idx = np.argmax(optimal_actions[h])  # Optimal action for this state
                        policy_action_name = action_names.get(policy_action_idx, f"action_{policy_action_idx}")
                        optimal_action_name = action_names.get(optimal_action_idx, f"action_{optimal_action_idx}")
                        context_strs.append(f"{tuple(state)}: {policy_action_name} (opt: {optimal_action_name})")
                    
                    if horizon > 10:
                        context_strs.append("...")
                    
                    goal_str = f"Goal state: {tuple(env.goal)}"
                    context_str = ", ".join(context_strs)
                    print(f"  Debug traj {len(trajs)}: {goal_str}, context: {context_str}")
                
                traj = {
                    'query_state': query_state,  # Kept for backward compatibility
                    'optimal_action': query_optimal_action,  # Kept for backward compatibility
                    'optimal_actions': optimal_actions,  # NEW: optimal actions for each context state
                    'context_states': context_states[env_idx],  # Shape: (horizon, state_dim)
                    'context_actions': context_actions[env_idx],  # Shape: (horizon, action_dim)
                    'context_next_states': context_next_states[env_idx],  # Shape: (horizon, state_dim)
                    'context_rewards': context_rewards[env_idx],  # Shape: (horizon,)
                    'goal': env.goal,
                }
                
                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index
                
                trajs.append(traj)
    
    print(f"\n  Collected {len(trajs)} trajectories total")
    import sys
    sys.stdout.flush()
    
    # Prepare rollout_data for visualization
    if n_hists == 1 and rollout_rewards:
        rollout_data = {
            'rewards': rollout_rewards[0],  # Shape: (num_envs, horizon)
            'states': rollout_states[0],  # Shape: (num_envs, horizon, 2)
            'actions': rollout_actions[0],  # Shape: (num_envs, horizon, 5)
        }
    else:
        rollout_data = None
    
    return trajs, rollout_data


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    if not os.path.exists('datasets'):
        os.makedirs('datasets', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory containing checkpoints to resume from')

    cli_args = vars(parser.parse_args())

    # Load config from YAML file
    with open(cli_args['config'], 'r') as f:
        config_args = yaml.safe_load(f)

    # Use config values as args and merge CLI-only overrides
    args = config_args
    if cli_args.get('checkpoint_dir') is not None:
        args['checkpoint_dir'] = cli_args['checkpoint_dir']

    print("Args: ", args)

    # Determine experiment identifier (job ID or timestamp)
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        experiment_id = job_id
    else:
        experiment_id = str(int(time.time()))

    # Get config base name (without .yaml extension)
    config_basename = Path(cli_args['config']).stem
    
    print(f"Experiment ID: {experiment_id}")
    print(f"Config: {config_basename}")

    env = args['env']
    
    print(f"EVAL_ENABLED: {EVAL_ENABLED}, env: {env}")
    
    # Initialize wandb
    wandb.init(
        project="dpt-training",
        name=f"online_rollout_{config_basename}",
        config=args,
        id=experiment_id,
        resume="allow",
    )
    
    n_envs = args.get('envs', 100000)
    n_hists = args.get('hists', 1)
    n_samples = args.get('samples', 1)
    horizon = args.get('H', 100)
    dim = args.get('dim', 10)
    state_dim = dim
    action_dim = dim
    n_embd = args.get('embd', 32)
    n_head = args.get('head', 1)
    n_layer = args.get('layer', 3)
    lr = args.get('lr', 1e-3)
    shuffle = args.get('shuffle', False)
    dropout = args.get('dropout', 0)
    var = args.get('var', 0.0)
    cov = args.get('cov', 0.0)
    num_epochs = args.get('chutney', 1000)
    seed = args.get('seed', 0)
    lin_d = args.get('lin_d', 2)
    sample_action = args.get('sample_action', True)  # Whether to sample actions during rollout
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tmp_seed)
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    if shuffle and env == 'linear_bandit':
        raise Exception("Are you sure you want to shuffle on the linear bandit? Data collected from an adaptive algorithm in a stochastic setting can bias the learner if shuffled.")

    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
    }
    
    if env == 'bandit':
        state_dim = 1
        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'linear_bandit':
        state_dim = 1
        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(env, model_config)

    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5
        filename = build_darkroom_model_filename(env, model_config)

    elif env == 'miniworld':
        state_dim = 2
        action_dim = 4
        filename = build_miniworld_model_filename(env, model_config)

    else:
        raise NotImplementedError

    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
    }
    if env == 'miniworld':
        config.update({'image_size': 25, 'store_gpu': False})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)

    # Watch model for gradient and parameter tracking
    wandb.watch(model, log='all', log_freq=100)

    params = {
        'batch_size': 64,
        'shuffle': True,
    }

    # Create or reuse experiment directory structure
    checkpoint_dir = args.get('checkpoint_dir')
    if checkpoint_dir is not None and os.path.isdir(checkpoint_dir):
        experiment_dir = checkpoint_dir
    else:
        experiment_dir = f'models/{env}/{config_basename}'
        os.makedirs(experiment_dir, exist_ok=True)

    # Save/Update config file in experiment directory
    config_path = f'{experiment_dir}/config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    log_filename = f'{experiment_dir}/logs.txt'
    if checkpoint_dir is None:
        with open(log_filename, 'w') as f:
            pass
    else:
        os.makedirs(experiment_dir, exist_ok=True)
        if not os.path.exists(log_filename):
            with open(log_filename, 'w') as f:
                pass
    
    def printw(string):
        """A drop-in replacement for print that also writes to a log file."""
        print(string)
        with open(log_filename, 'a') as f:
            print(string, file=f)

    # Resume model weights if checkpoint_dir provided
    def _find_latest_checkpoint(directory):
        epoch_ckpts = []
        try:
            for fn in os.listdir(directory):
                if fn.startswith('epoch') and fn.endswith('.pt'):
                    num_str = fn[len('epoch'):-3]
                    if num_str.isdigit():
                        epoch_ckpts.append((int(num_str), os.path.join(directory, fn)))
        except FileNotFoundError:
            return None
        if epoch_ckpts:
            epoch_ckpts.sort(key=lambda x: x[0])
            return epoch_ckpts[-1][1]
        final_path = os.path.join(directory, 'final_model.pt')
        return final_path if os.path.exists(final_path) else None

    def _get_existing_max_epoch(directory):
        max_epoch = 0
        try:
            for fn in os.listdir(directory):
                if fn.startswith('epoch') and fn.endswith('.pt'):
                    num_str = fn[len('epoch'):-3]
                    if num_str.isdigit():
                        max_epoch = max(max_epoch, int(num_str))
        except FileNotFoundError:
            return 0
        return max_epoch

    resume_epoch_offset = 0
    if checkpoint_dir is not None:
        latest_ckpt = _find_latest_checkpoint(experiment_dir)
        if latest_ckpt is not None:
            state = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(state)
            printw(f"Resumed model weights from checkpoint: {latest_ckpt}")
            resume_epoch_offset = _get_existing_max_epoch(experiment_dir)
        else:
            printw(f"No checkpoint found in {experiment_dir}; starting fresh.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # test_loss = []  # Commented out: test logic disabled
    train_loss = []

    # Metrics persistence to preserve curves across resumes
    metrics_path = os.path.join(experiment_dir, 'metrics.npz')
    if checkpoint_dir is not None and os.path.exists(metrics_path):
        try:
            loaded = np.load(metrics_path, allow_pickle=True)
            train_loss = list(loaded.get('train_loss', np.array([])).tolist())
            # test_loss = list(loaded.get('test_loss', np.array([])).tolist())  # Commented out: test logic disabled
            printw(f"Loaded previous metrics from {metrics_path} (epochs={len(train_loss)})")
        except Exception as e:
            printw(f"Warning: Failed to load metrics from {metrics_path}: {e}")

    # Limit to remaining epochs if resuming
    remaining_epochs = max(0, num_epochs - resume_epoch_offset)

    # Prepare environment generation parameters
    n_train_envs = int(.8 * n_envs)
    # n_test_envs = n_envs - n_train_envs  # Commented out: test logic disabled
    
    # For linear bandit, generate arms once
    arms = None
    if env == 'linear_bandit':
        rng = np.random.RandomState(seed=1234)
        arms = rng.normal(size=(dim, lin_d)) / np.sqrt(lin_d)

    printw("Starting online rollout training...")
    printw(f"Num epochs: {remaining_epochs}")
    
    for epoch in range(remaining_epochs):
        current_epoch = resume_epoch_offset + epoch + 1
        printw(f"\n{'='*60}")
        printw(f"Epoch: {current_epoch}")
        printw(f"{'='*60}")
        
        # STEP 1: Collect rollout data using current model
        printw(f"Collecting rollout data for epoch {current_epoch}...")
        printw(f"  n_train_envs={n_train_envs}, n_hists={n_hists}, n_samples={n_samples}, horizon={horizon}")
        rollout_start_time = time.time()
        
        # Always use the current model (even if randomly initialized for first epoch)
        # The model will sample actions based on its current parameters
        rollout_model = model
        
        rollout_data = None  # Initialize for non-darkroom environments
        if env == 'bandit':
            # Create training environments
            train_envs = [bandit_env.sample(dim, horizon, var) for _ in range(n_train_envs)]
            # test_envs = [bandit_env.sample(dim, horizon, var) for _ in range(n_test_envs)]  # Commented out: test logic disabled
            
            # Collect rollout data using vectorized rollout (same as evaluation)
            train_trajs = collect_rollout_data_bandit(
                rollout_model, train_envs, horizon, n_hists, n_samples, sample_action
            )
            # test_trajs = collect_rollout_data_bandit(
            #     rollout_model, test_envs, horizon, n_hists, n_samples, sample_action
            # )  # Commented out: test logic disabled
            
        elif env == 'linear_bandit':
            # Create training environments
            train_envs = [bandit_env.sample_linear(arms, horizon, var) for _ in range(n_train_envs)]
            # test_envs = [bandit_env.sample_linear(arms, horizon, var) for _ in range(n_test_envs)]  # Commented out: test logic disabled
            
            # Collect rollout data
            train_trajs = collect_rollout_data_linear_bandit(
                rollout_model, train_envs, horizon, n_hists, n_samples, arms, sample_action
            )
            # test_trajs = collect_rollout_data_linear_bandit(
            #     rollout_model, test_envs, horizon, n_hists, n_samples, arms, sample_action
            # )  # Commented out: test logic disabled
            
        elif env.startswith('darkroom'):
            # Create training environments
            goals = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
            np.random.RandomState(seed=0).shuffle(goals)
            train_test_split = int(.8 * len(goals))
            train_goals = goals[:train_test_split]
            # test_goals = goals[train_test_split:]  # Commented out: test logic disabled
            
            train_goals = np.repeat(train_goals, max(1, n_train_envs // len(train_goals)), axis=0)[:n_train_envs]
            # test_goals = np.repeat(test_goals, max(1, n_test_envs // len(test_goals)), axis=0)[:n_test_envs]  # Commented out: test logic disabled
            
            if env == 'darkroom_heldout':
                train_envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in train_goals]
                # test_envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in test_goals]  # Commented out: test logic disabled
            else:
                train_envs = [darkroom_env.DarkroomEnvPermuted(dim, i, horizon) for i in range(n_train_envs)]
                # test_envs = [darkroom_env.DarkroomEnvPermuted(dim, i + n_train_envs, horizon) for i in range(n_test_envs)]  # Commented out: test logic disabled
            
            # Collect rollout data
            train_trajs, rollout_data = collect_rollout_data_darkroom(
                rollout_model, train_envs, horizon, n_hists, n_samples, sample_action
            )
            # test_trajs = collect_rollout_data_darkroom(
            #     rollout_model, test_envs, horizon, n_hists, n_samples, sample_action
            # )  # Commented out: test logic disabled
            
        else:
            raise NotImplementedError(f"Environment {env} not yet supported for rollout training")
        
        # Save rollout data to temporary files
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir='datasets') as f_train:
            train_path = f_train.name
            pickle.dump(train_trajs, f_train)
        
        # Store rollout_data for visualization (only for darkroom)
        train_rollout_data = rollout_data if env.startswith('darkroom') else None
        # Store train_envs for visualization (only for darkroom)
        train_envs_for_viz = train_envs if env.startswith('darkroom') else None
        
        # with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir='datasets') as f_test:
        #     test_path = f_test.name
        #     pickle.dump(test_trajs, f_test)
        # Commented out: test logic disabled
        
        rollout_time = time.time() - rollout_start_time
        printw(f"Collected {len(train_trajs)} training trajectories")  # Removed test trajectory count
        printw(f"Rollout time: {rollout_time:.2f}s")
        
        # STEP 2: Create datasets and dataloaders
        printw("Creating datasets and dataloaders...")
        dataset_start_time = time.time()
        if env == 'miniworld':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            params.update({
                'num_workers': 16,
                'prefetch_factor': 2,
                'persistent_workers': True,
                'pin_memory': True,
                'batch_size': 64,
                'worker_init_fn': worker_init_fn,
            })
            train_dataset = ImageDataset([train_path], config, transform)
            # test_dataset = ImageDataset([test_path], config, transform)  # Commented out: test logic disabled
        else:
            train_dataset = Dataset(train_path, config)
            # test_dataset = Dataset(test_path, config)  # Commented out: test logic disabled
        
        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        # test_loader = torch.utils.data.DataLoader(test_dataset, **params)  # Commented out: test logic disabled
        
        dataset_time = time.time() - dataset_start_time
        printw(f"Num train batches: {len(train_loader)}")
        printw(f"Dataset creation time: {dataset_time:.2f}s")
        # printw(f"Num test batches: {len(test_loader)}")  # Commented out: test logic disabled
        
        # STEP 3: Evaluate on test set
        # Commented out: test logic disabled
        # printw("Evaluating on test set...")
        # start_time = time.time()
        # with torch.no_grad():
        #     epoch_test_loss = 0.0
        #     epoch_test_entropy = 0.0
        #     for i, batch in enumerate(test_loader):
        #         print(f"Test batch {i} of {len(test_loader)}", end='\r')
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         true_actions = batch['optimal_actions']
        #         pred_actions, _ = model(batch)
        #         true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
        #         true_actions = true_actions.reshape(-1, action_dim)
        #         pred_actions = pred_actions.reshape(-1, action_dim)
        #
        #         loss = loss_fn(pred_actions, true_actions)
        #         epoch_test_loss += loss.item() / horizon
        #         
        #         # Compute entropy
        #         pred_probs = torch.softmax(pred_actions, dim=-1)
        #         entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1).sum()
        #         epoch_test_entropy += entropy.item() / horizon
        #
        # test_loss.append(epoch_test_loss / len(test_dataset))
        # test_entropy_val = (epoch_test_entropy / len(test_dataset))
        # end_time = time.time()
        # printw(f"\tTest loss: {test_loss[-1]}")
        # printw(f"\tTest entropy: {test_entropy_val:.4f}")
        # printw(f"\tEval time: {end_time - start_time:.2f}s")

        # STEP 4: Train for one epoch
        printw("Training...")
        epoch_train_loss = 0.0
        epoch_train_entropy = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Train batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            pred_actions, _ = model(batch)
            
            # For MDPs, use optimal_actions_per_state if available (one optimal action per context state)
            # For bandits, fall back to repeating single optimal_action
            if 'optimal_actions_per_state' in batch and batch['optimal_actions_per_state'] is not None:
                # Use per-state optimal actions (shape: [batch, horizon, action_dim])
                true_actions = batch['optimal_actions_per_state']  # Already has shape [batch, horizon, action_dim]
            else:
                # Fall back to single optimal_action (for bandit environments)
                true_actions = batch['optimal_actions']
                true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
            
            # Reshape for loss computation
            true_actions = true_actions.reshape(-1, action_dim)
            pred_actions = pred_actions.reshape(-1, action_dim)

            optimizer.zero_grad()
            loss = loss_fn(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() / horizon
            
            # Compute entropy
            pred_probs = torch.softmax(pred_actions, dim=-1)
            entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1).sum()
            epoch_train_entropy += entropy.item() / horizon

        train_loss.append(epoch_train_loss / len(train_dataset))
        train_entropy_val = (epoch_train_entropy / len(train_dataset))
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]}")
        printw(f"\tTrain entropy: {train_entropy_val:.4f}")
        printw(f"\tTrain time: {end_time - start_time:.2f}s")
        
        # Log training metrics to wandb
        wandb.log({
            "epoch": current_epoch,
            "train_ce_loss": train_loss[-1],
            "train_entropy": train_entropy_val,
            "train_alpha": 1.0,
            "train_beta": 0.0,
            "total_loss": train_loss[-1],
            # "test_loss": test_loss[-1],  # Commented out: test logic disabled
            # "test_ce_loss": test_loss[-1],  # Commented out: test logic disabled
            # "test_entropy": test_entropy_val,  # Commented out: test logic disabled
            "rollout_time": rollout_time,
            "train_time": end_time - start_time,  # Changed from eval_time to train_time
        })
        
        # Plot training rollout rewards and paths (for darkroom)
        if train_rollout_data is not None and env.startswith('darkroom') and train_envs_for_viz is not None:
            rollout_rewards = train_rollout_data['rewards']  # Shape: (num_envs, horizon)
            rollout_states = train_rollout_data['states']  # Shape: (num_envs, horizon, 2)
            rollout_actions = train_rollout_data['actions']  # Shape: (num_envs, horizon, 5)
            
            # Compute step-level statistics
            step_rewards_mean = np.mean(rollout_rewards, axis=0)  # Shape: (horizon,)
            step_rewards_sem = scipy.stats.sem(rollout_rewards, axis=0)  # Shape: (horizon,)
            step_indices = np.arange(horizon)
            
            # Sample 5 random environments for visualization (same as eval)
            np.random.seed(42 + current_epoch)  # For reproducibility with epoch variation
            num_samples = min(5, n_train_envs)
            sample_env_indices = np.random.choice(n_train_envs, size=num_samples, replace=False)
            
            # Different colors for each rollout (same as eval)
            rollout_colors = ['orange', 'purple', 'brown', 'pink', 'cyan']
            
            # Create online plot - step-level performance showing context accumulation
            fig_online, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Plot individual trajectories (first 10 environments) in light blue
            for i in range(min(n_train_envs, 10)):
                cumulative_returns = np.cumsum(rollout_rewards[i])
                ax.plot(cumulative_returns, color='blue', alpha=0.1, linewidth=1)
            
            # Plot sample rollouts with explicit labels and colors (like eval)
            log_offset = 0.1
            for i, env_idx in enumerate(sample_env_indices):
                rollout_rewards_env = rollout_rewards[env_idx]  # Shape: (horizon,)
                cumulative_returns = np.cumsum(rollout_rewards_env)
                color = rollout_colors[i % len(rollout_colors)]
                cumulative_returns_log = cumulative_returns + log_offset
                ax.plot(cumulative_returns_log, color=color, alpha=0.6, linewidth=1.5,
                       linestyle='--', label=f'Rollout {i+1} (Env {env_idx})')
            
            # Plot mean cumulative returns
            cumulative_returns_mean = np.cumsum(step_rewards_mean)
            cumulative_returns_mean_log = cumulative_returns_mean + log_offset
            cumulative_sem = np.cumsum(step_rewards_sem)
            
            ax.plot(cumulative_returns_mean_log, label='Train (mean cumulative)', color='blue', linewidth=2)
            ax.fill_between(step_indices,
                            cumulative_returns_mean_log - cumulative_sem,
                            cumulative_returns_mean_log + cumulative_sem,
                            alpha=0.2, color='blue')
            
            ax.set_xlabel('Step (Context Accumulation)')
            ax.set_ylabel('Cumulative Return (log scale, offset=0.1)')
            ax.set_yscale('log')
            ax.set_title(f'Train Rollout: Step-Level Performance (n={n_train_envs} envs)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            wandb.log({f"train/online_image": wandb.Image(fig_online)}, commit=False)
            plt.close(fig_online)
            
            # Create path visualizations for sample rollouts (5 random rollouts)
            fig_paths, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
            if num_samples == 1:
                axes = [axes]
            
            for idx, env_idx in enumerate(sample_env_indices):
                ax = axes[idx]
                states = rollout_states[env_idx]  # Shape: (horizon, 2)
                actions = rollout_actions[env_idx]  # Shape: (horizon, 5)
                
                # Get goal for this environment using stored train_envs_for_viz
                if env == 'darkroom_heldout':
                    goal = train_envs_for_viz[env_idx].goal
                else:
                    # For DarkroomEnvPermuted, goal is always at bottom right
                    goal = np.array([dim - 1, dim - 1])
                
                _visualize_rollout_path_single_ax(ax, states, actions, goal, dim,
                                                 f'Train Rollout {idx+1} (Env {env_idx})')
            
            plt.tight_layout()
            wandb.log({f"train/path_visualization": wandb.Image(fig_paths)}, commit=False)
            plt.close(fig_paths)

        # Save checkpoint
        if (epoch + 1) % args.get('eval_every', 10) == 0:
            torch.save(model.state_dict(), f'{experiment_dir}/epoch{current_epoch}.pt')
            
            # Run evaluation and log plots to wandb
            if EVAL_ENABLED and (env in ['bandit', 'bandit_bernoulli', 'linear_bandit'] or env.startswith('darkroom') or env == 'miniworld'):
                printw(f"Running evaluation for epoch {current_epoch}, env={env}")
                try:
                    log_evaluation_plots_to_wandb(model, config, args, env, current_epoch)
                except Exception as e:
                    printw(f"Warning: Evaluation failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                printw(f"Evaluation skipped: EVAL_ENABLED={EVAL_ENABLED}, env={env}")

        # Clean up temporary files
        try:
            os.unlink(train_path)
            # os.unlink(test_path)  # Commented out: test logic disabled
        except Exception as e:
            printw(f"Warning: Failed to delete temporary files: {e}")

        # Persist metrics
        if (epoch + 1) % 10 == 0:
            try:
                np.savez_compressed(
                    metrics_path,
                    train_loss=np.array(train_loss),
                    # test_loss=np.array(test_loss),  # Commented out: test logic disabled
                )
            except Exception as e:
                printw(f"Warning: Failed to save metrics to {metrics_path}: {e}")

    torch.save(model.state_dict(), f'{experiment_dir}/final_model.pt')
    wandb.finish()
    printw("Done.")
