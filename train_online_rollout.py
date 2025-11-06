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


def collect_rollout_data_bandit(model, envs, horizon, n_hists, n_samples, sample_action=True):
    """
    Collect rollout data using the current model for bandit environments.
    
    Args:
        model: The current model (or None for random rollout)
        envs: List of bandit environments
        horizon: Context horizon
        n_hists: Number of histories per environment
        n_samples: Number of samples per history
        
    Returns:
        List of trajectories in the format expected by Dataset
    """
    trajs = []
    
    # Create controller (will be reused for all environments)
    controller = None
    if model is not None:
        controller = BanditTransformerController(model, sample=sample_action, batch_size=1)
    
    for env_idx, env in enumerate(envs):
        for j in range(n_hists):
            # Roll out one history
            context_states, context_actions, _, context_rewards = _rollout_bandit_single(
                env, controller, horizon
            )
            
            # Create n_samples trajectories with the same context
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a
                
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards,
                    'means': env.means,
                }
                trajs.append(traj)
    
    return trajs


def _rollout_bandit_single(env, controller, horizon):
    """
    Roll out a single history for a bandit environment.
    """
    xs, us, xps, rs = [], [], [], []
    
    # If no controller, use random actions
    if controller is None:
        for h in range(horizon):
            x = np.array([1])
            u = np.zeros(env.dim)
            i = np.random.choice(np.arange(env.dim))
            u[i] = 1.0
            xp, r = env.transit(x, u)
            
            xs.append(x)
            us.append(u)
            xps.append(xp)
            rs.append(r)
    else:
        # Use model controller
        # Create context as we go
        context_states_np = np.zeros((1, horizon, env.dx))
        context_actions_np = np.zeros((1, horizon, env.du))
        context_rewards_np = np.zeros((1, horizon, 1))
        
        for h in range(horizon):
            # Prepare batch with accumulated context
            batch = {
                'context_states': context_states_np[:, :h, :],
                'context_actions': context_actions_np[:, :h, :],
                'context_rewards': context_rewards_np[:, :h, :],
            }
            controller.set_batch_numpy_vec(batch)
            
            # Get action from controller
            x = np.array([1])
            # act_numpy_vec expects batch input, returns batch output
            u_batch = controller.act_numpy_vec(x[None, :])  # Shape: (1, action_dim)
            u = u_batch[0]  # Get single action
            
            # Execute in environment
            xp, r = env.transit(x, u)
            
            # Store transition
            xs.append(x)
            us.append(u)
            xps.append(xp)
            rs.append(r)
            
            # Update context for next step
            context_states_np[0, h, :] = x
            context_actions_np[0, h, :] = u
            context_rewards_np[0, h, 0] = r
    
    xs, us, xps, rs = np.array(xs), np.array(us), np.array(xps), np.array(rs)
    return xs, us, xps, rs


def collect_rollout_data_linear_bandit(model, envs, horizon, n_hists, n_samples, arms, sample_action=True):
    """
    Collect rollout data using the current model for linear bandit environments.
    """
    trajs = []
    
    # Create controller (will be reused for all environments)
    controller = None
    if model is not None:
        controller = BanditTransformerController(model, sample=sample_action, batch_size=1)
    
    for env_idx, env in enumerate(envs):
        for j in range(n_hists):
            context_states, context_actions, _, context_rewards = _rollout_bandit_single(
                env, controller, horizon
            )
            
            for k in range(n_samples):
                query_state = np.array([1])
                optimal_action = env.opt_a
                
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_rewards': context_rewards,
                    'means': env.means,
                    'arms': arms,
                    'theta': env.theta,
                    'var': env.var,
                }
                trajs.append(traj)
    
    return trajs


def collect_rollout_data_darkroom(model, envs, horizon, n_hists, n_samples, sample_action=True):
    """
    Collect rollout data using the current model for darkroom environments.
    """
    trajs = []
    
    # Create controller
    if model is None:
        controller = None
    else:
        controller = DarkroomTransformerController(model, batch_size=1, sample=sample_action)
    
    for env in envs:
        for j in range(n_hists):
            # Roll out one history
            context_states, context_actions, context_next_states, context_rewards = _rollout_darkroom_single(
                env, controller, horizon
            )
            
            for k in range(n_samples):
                query_state = env.sample_state()
                optimal_action = env.opt_action(query_state)
                
                traj = {
                    'query_state': query_state,
                    'optimal_action': optimal_action,
                    'context_states': context_states,
                    'context_actions': context_actions,
                    'context_next_states': context_next_states,
                    'context_rewards': context_rewards,
                    'goal': env.goal,
                }
                
                # Add perm_index for DarkroomEnvPermuted
                if hasattr(env, 'perm_index'):
                    traj['perm_index'] = env.perm_index
                
                trajs.append(traj)
    
    return trajs


def _rollout_darkroom_single(env, controller, horizon):
    """
    Roll out a single history for a darkroom environment.
    """
    states = []
    actions = []
    next_states = []
    rewards = []
    
    state = env.reset()
    
    for _ in range(horizon):
        if controller is None:
            # Random action
            action = env.sample_action()
        else:
            # Use model to get action
            # Build context from accumulated history
            if len(states) == 0:
                # Empty context
                batch = {
                    'context_states': torch.zeros((1, 1, env.state_dim)).float().to(device),
                    'context_actions': torch.zeros((1, 1, env.action_dim)).float().to(device),
                    'context_next_states': torch.zeros((1, 1, env.state_dim)).float().to(device),
                    'context_rewards': torch.zeros((1, 1, 1)).float().to(device),
                }
            else:
                batch = {
                    'context_states': torch.tensor(np.array(states))[None, :, :].float().to(device),
                    'context_actions': torch.tensor(np.array(actions))[None, :, :].float().to(device),
                    'context_next_states': torch.tensor(np.array(next_states))[None, :, :].float().to(device),
                    'context_rewards': torch.tensor(np.array(rewards))[None, :, None].float().to(device),
                }
            controller.set_batch(batch)
            action = controller.act(state)
        
        next_state, reward = env.transit(state, action)
        
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state
    
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    
    return states, actions, next_states, rewards


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

    test_loss = []
    train_loss = []

    # Metrics persistence to preserve curves across resumes
    metrics_path = os.path.join(experiment_dir, 'metrics.npz')
    if checkpoint_dir is not None and os.path.exists(metrics_path):
        try:
            loaded = np.load(metrics_path, allow_pickle=True)
            train_loss = list(loaded.get('train_loss', np.array([])).tolist())
            test_loss = list(loaded.get('test_loss', np.array([])).tolist())
            printw(f"Loaded previous metrics from {metrics_path} (epochs={len(train_loss)})")
        except Exception as e:
            printw(f"Warning: Failed to load metrics from {metrics_path}: {e}")

    # Limit to remaining epochs if resuming
    remaining_epochs = max(0, num_epochs - resume_epoch_offset)

    # Prepare environment generation parameters
    n_train_envs = int(.8 * n_envs)
    n_test_envs = n_envs - n_train_envs
    
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
        rollout_start_time = time.time()
        
        # For epoch 0 (before training), use None model (random rollout)
        # For subsequent epochs, use the current model
        rollout_model = None if current_epoch == 1 else model
        
        if env == 'bandit':
            # Create training environments
            train_envs = [bandit_env.sample(dim, horizon, var) for _ in range(n_train_envs)]
            test_envs = [bandit_env.sample(dim, horizon, var) for _ in range(n_test_envs)]
            
            # Collect rollout data
            train_trajs = collect_rollout_data_bandit(
                rollout_model, train_envs, horizon, n_hists, n_samples, sample_action
            )
            test_trajs = collect_rollout_data_bandit(
                rollout_model, test_envs, horizon, n_hists, n_samples, sample_action
            )
            
        elif env == 'linear_bandit':
            # Create training environments
            train_envs = [bandit_env.sample_linear(arms, horizon, var) for _ in range(n_train_envs)]
            test_envs = [bandit_env.sample_linear(arms, horizon, var) for _ in range(n_test_envs)]
            
            # Collect rollout data
            train_trajs = collect_rollout_data_linear_bandit(
                rollout_model, train_envs, horizon, n_hists, n_samples, arms, sample_action
            )
            test_trajs = collect_rollout_data_linear_bandit(
                rollout_model, test_envs, horizon, n_hists, n_samples, arms, sample_action
            )
            
        elif env.startswith('darkroom'):
            # Create training environments
            goals = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
            np.random.RandomState(seed=0).shuffle(goals)
            train_test_split = int(.8 * len(goals))
            train_goals = goals[:train_test_split]
            test_goals = goals[train_test_split:]
            
            train_goals = np.repeat(train_goals, max(1, n_train_envs // len(train_goals)), axis=0)[:n_train_envs]
            test_goals = np.repeat(test_goals, max(1, n_test_envs // len(test_goals)), axis=0)[:n_test_envs]
            
            if env == 'darkroom_heldout':
                train_envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in train_goals]
                test_envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in test_goals]
            else:
                train_envs = [darkroom_env.DarkroomEnvPermuted(dim, i, horizon) for i in range(n_train_envs)]
                test_envs = [darkroom_env.DarkroomEnvPermuted(dim, i + n_train_envs, horizon) for i in range(n_test_envs)]
            
            # Collect rollout data
            train_trajs = collect_rollout_data_darkroom(
                rollout_model, train_envs, horizon, n_hists, n_samples, sample_action
            )
            test_trajs = collect_rollout_data_darkroom(
                rollout_model, test_envs, horizon, n_hists, n_samples, sample_action
            )
            
        else:
            raise NotImplementedError(f"Environment {env} not yet supported for rollout training")
        
        # Save rollout data to temporary files
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir='datasets') as f_train:
            train_path = f_train.name
            pickle.dump(train_trajs, f_train)
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl', dir='datasets') as f_test:
            test_path = f_test.name
            pickle.dump(test_trajs, f_test)
        
        rollout_time = time.time() - rollout_start_time
        printw(f"Collected {len(train_trajs)} training trajectories and {len(test_trajs)} test trajectories")
        printw(f"Rollout time: {rollout_time:.2f}s")
        
        # STEP 2: Create datasets and dataloaders
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
            test_dataset = ImageDataset([test_path], config, transform)
        else:
            train_dataset = Dataset(train_path, config)
            test_dataset = Dataset(test_path, config)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, **params)
        test_loader = torch.utils.data.DataLoader(test_dataset, **params)
        
        printw(f"Num train batches: {len(train_loader)}")
        printw(f"Num test batches: {len(test_loader)}")
        
        # STEP 3: Evaluate on test set
        printw("Evaluating on test set...")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            epoch_test_entropy = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Test batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['optimal_actions']
                pred_actions, _ = model(batch)
                true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
                true_actions = true_actions.reshape(-1, action_dim)
                pred_actions = pred_actions.reshape(-1, action_dim)

                loss = loss_fn(pred_actions, true_actions)
                epoch_test_loss += loss.item() / horizon
                
                # Compute entropy
                pred_probs = torch.softmax(pred_actions, dim=-1)
                entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1).sum()
                epoch_test_entropy += entropy.item() / horizon

        test_loss.append(epoch_test_loss / len(test_dataset))
        test_entropy_val = (epoch_test_entropy / len(test_dataset))
        end_time = time.time()
        printw(f"\tTest loss: {test_loss[-1]}")
        printw(f"\tTest entropy: {test_entropy_val:.4f}")
        printw(f"\tEval time: {end_time - start_time:.2f}s")

        # STEP 4: Train for one epoch
        printw("Training...")
        epoch_train_loss = 0.0
        epoch_train_entropy = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Train batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch['optimal_actions']
            pred_actions, _ = model(batch)
            true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)
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
            "test_loss": test_loss[-1],
            "test_ce_loss": test_loss[-1],
            "test_entropy": test_entropy_val,
            "rollout_time": rollout_time,
            "eval_time": end_time - start_time,
        })

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
            os.unlink(test_path)
        except Exception as e:
            printw(f"Warning: Failed to delete temporary files: {e}")

        # Persist metrics
        if (epoch + 1) % 10 == 0:
            try:
                np.savez_compressed(
                    metrics_path,
                    train_loss=np.array(train_loss),
                    test_loss=np.array(test_loss),
                )
            except Exception as e:
                printw(f"Warning: Failed to save metrics to {metrics_path}: {e}")

    torch.save(model.state_dict(), f'{experiment_dir}/final_model.pt')
    wandb.finish()
    printw("Done.")
