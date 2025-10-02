import argparse
import os
import pickle
import yaml

import matplotlib.pyplot as plt
import torch
from IPython import embed

import common_args
from evals import eval_bandit, eval_linear_bandit, eval_darkroom
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_model_filename,
    build_linear_bandit_model_filename,
    build_darkroom_model_filename,
    build_miniworld_model_filename,
)
from envs import darkroom_env, bandit_env
import numpy as np
import scipy
import time
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_bandit_environments(n_eval, dim, var, bandit_type='uniform'):
    """Generate bandit environments for evaluation."""
    eval_envs = []
    for _ in range(n_eval):
        if bandit_type == 'uniform':
            env = bandit_env.sample(dim, 1, var)  # horizon=1 for evaluation
        elif bandit_type == 'bernoulli':
            env = bandit_env.sample_bernoulli(dim, 1)
        else:
            raise ValueError(f"Unknown bandit type: {bandit_type}")
        eval_envs.append(env)
    return eval_envs


def generate_linear_bandit_environments(n_eval, dim, lin_d, var, seed=1234):
    """Generate linear bandit environments for evaluation."""
    # Generate fixed features for arms (same as in train_online.py)
    rng = np.random.RandomState(seed=seed)
    arms = rng.normal(size=(dim, lin_d)) / np.sqrt(lin_d)
    
    eval_envs = []
    for _ in range(n_eval):
        env = bandit_env.sample_linear(arms, 1, var)  # horizon=1 for evaluation
        eval_envs.append(env)
    return eval_envs


def generate_darkroom_environments(n_eval, dim, env_type):
    """Generate darkroom environments for evaluation."""
    eval_envs = []
    
    if env_type == 'darkroom_heldout':
        # Generate goals for evaluation (different from training)
        goals = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
        np.random.RandomState(seed=0).shuffle(goals)
        train_test_split = int(.8 * len(goals))
        eval_goals = goals[train_test_split:]  # Use held-out goals
        
        # Repeat goals if we need more environments
        eval_goals = np.repeat(eval_goals, max(1, n_eval // len(eval_goals)), axis=0)[:n_eval]
        
        for goal in eval_goals:
            env = darkroom_env.DarkroomEnv(dim, goal, 1)  # horizon=1 for evaluation
            eval_envs.append(env)
            
    elif env_type == 'darkroom_permuted':
        for i in range(n_eval):
            env = darkroom_env.DarkroomEnvPermuted(dim, i, 1)  # horizon=1 for evaluation
            eval_envs.append(env)
    else:
        raise ValueError(f"Unknown darkroom type: {env_type}")
        
    return eval_envs


def generate_miniworld_environments(n_eval):
    """Generate miniworld environments for evaluation."""
    import gymnasium as gym
    import miniworld
    
    eval_envs = []
    for env_id in range(n_eval):
        gym_env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
        gym_env.env_id = env_id
        eval_envs.append(gym_env)
    return eval_envs


def generate_random_trajectory(env, env_type, horizon):
    """Generate a random trajectory for the given environment."""
    if env_type in ['bandit', 'bandit_bernoulli']:
        # For bandits, state is always [1]
        context_states = np.ones((horizon, 1))
        context_next_states = np.ones((horizon, 1))
        context_actions = np.zeros((horizon, env.dim))
        context_rewards = np.zeros((horizon,))
        
        for h in range(horizon):
            # Random action
            action_idx = np.random.randint(env.dim)
            context_actions[h, action_idx] = 1.0
            
            # Get reward from environment
            state = np.array([1])
            next_state, reward = env.transit(state, context_actions[h])
            context_rewards[h] = reward
            
    elif env_type == 'linear_bandit':
        # Similar to bandit but with linear structure
        context_states = np.ones((horizon, 1))
        context_next_states = np.ones((horizon, 1))
        context_actions = np.zeros((horizon, env.dim))
        context_rewards = np.zeros((horizon,))
        
        for h in range(horizon):
            # Random action
            action_idx = np.random.randint(env.dim)
            context_actions[h, action_idx] = 1.0
            
            # Get reward from environment
            state = np.array([1])
            next_state, reward = env.transit(state, context_actions[h])
            context_rewards[h] = reward
            
    elif env_type.startswith('darkroom'):
        # For darkroom, generate random states and actions
        context_states = np.zeros((horizon, 2))
        context_next_states = np.zeros((horizon, 2))
        context_actions = np.zeros((horizon, 5))
        context_rewards = np.zeros((horizon,))
        
        for h in range(horizon):
            # Random state and action
            state = env.sample_state()
            action = env.sample_action()
            next_state, reward = env.transit(state, action)
            
            context_states[h] = state
            context_actions[h] = action
            context_next_states[h] = next_state
            context_rewards[h] = reward
            
    elif env_type == 'miniworld':
        # For miniworld, this is more complex - simplified version
        context_states = np.zeros((horizon, 2))
        context_next_states = np.zeros((horizon, 2))
        context_actions = np.zeros((horizon, env.action_space.n))
        context_rewards = np.zeros((horizon,))
        
        # Reset environment
        env.reset()
        
        for h in range(horizon):
            # Random action
            action_idx = np.random.randint(env.action_space.n)
            context_actions[h, action_idx] = 1.0
            
            # Current state (simplified)
            current_state = env.agent.dir_vec[[0, -1]]
            context_states[h] = current_state
            
            # Step environment
            _, reward, done, _, _ = env.step(action_idx)
            context_rewards[h] = reward
            
            # Next state
            next_state = env.agent.dir_vec[[0, -1]]
            context_next_states[h] = next_state
            
            if done:
                env.reset()
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
        
    return context_states, context_actions, context_next_states, context_rewards


def convert_envs_to_trajs_format(envs, env_type, horizon):
    """Convert generated environments to the trajectory format expected by eval functions."""
    eval_trajs = []
    
    for env in envs:
        # Generate a random trajectory for this environment
        context_states, context_actions, context_next_states, context_rewards = generate_random_trajectory(env, env_type, horizon)
        
        if env_type in ['bandit', 'bandit_bernoulli']:
            traj = {
                'env': env,
                'means': env.means,
                'opt_a': env.opt_a,
                'dim': env.dim,
                'horizon': horizon,
                'context_states': context_states,
                'context_actions': context_actions,
                'context_next_states': context_next_states,
                'context_rewards': context_rewards,
            }
            if hasattr(env, 'var'):
                traj['var'] = env.var
                
        elif env_type == 'linear_bandit':
            traj = {
                'env': env,
                'arms': getattr(env, 'arms', None),
                'theta': env.theta,
                'means': env.means,
                'var': env.var,
                'opt_a': env.opt_a,
                'dim': env.dim,
                'horizon': horizon,
                'context_states': context_states,
                'context_actions': context_actions,
                'context_next_states': context_next_states,
                'context_rewards': context_rewards,
            }
            
        elif env_type.startswith('darkroom'):
            traj = {
                'env': env,
                'goal': env.goal,
                'dim': env.dim,
                'horizon': horizon,
                'context_states': context_states,
                'context_actions': context_actions,
                'context_next_states': context_next_states,
                'context_rewards': context_rewards,
            }
            if hasattr(env, 'perm_index'):
                traj['perm_index'] = env.perm_index
                
        elif env_type == 'miniworld':
            traj = {
                'env': env,
                'env_id': getattr(env, 'env_id', 0),
                'horizon': horizon,
                'context_states': context_states,
                'context_actions': context_actions,
                'context_next_states': context_next_states,
                'context_rewards': context_rewards,
            }
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
            
        eval_trajs.append(traj)
    
    return eval_trajs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Direct path to model file (e.g., models/bandits/100/epoch1.pt)')

    args = vars(parser.parse_args())

    # Store model_path separately before loading config
    model_path = args['model_path']

    # Load config from YAML file
    with open(args['config'], 'r') as f:
        config_args = yaml.safe_load(f)

    # Use config values as args, but keep model_path
    args = config_args
    args['model_path'] = model_path

    print("Args: ", args)

    n_envs = args.get('envs', 100)
    n_hists = args.get('hists', 1)
    H = args.get('H', 100)
    n_samples = args.get('samples', 1)
    dim = args.get('dim', 10)
    state_dim = dim
    action_dim = dim
    n_embd = args.get('embd', 32)
    n_head = args.get('head', 1)
    n_layer = args.get('layer', 3)
    lr = args.get('lr', 1e-3)
    epoch = args.get('epoch', -1)
    shuffle = args.get('shuffle', False)
    dropout = args.get('dropout', 0)
    var = args.get('var', 0.0)
    cov = args.get('cov', 0.0)
    test_cov = args.get('test_cov', -1.0)
    envname = args['env']
    horizon = args.get('hor', -1)
    n_eval = args.get('n_eval', 100)
    seed = args.get('seed', 0)
    lin_d = args.get('lin_d', 2)  # Only needed for linear_bandit
    # Remove online-specific parameters - we get the model directly via model_path
    
    # Since we load the model directly from model_path, we don't need confidence logic
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    if test_cov < 0:
        test_cov = cov
    if horizon < 0:
        horizon = H

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
    
    # Setup environment-specific configurations
    if envname == 'bandit':
        state_dim = 1
        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(envname, model_config)
        bandit_type = 'uniform'
        
    elif envname == 'bandit_bernoulli':
        state_dim = 1
        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(envname, model_config)
        bandit_type = 'bernoulli'
        
    elif envname == 'linear_bandit':
        state_dim = 1
        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(envname, model_config)
        
    elif envname.startswith('darkroom'):
        state_dim = 2
        action_dim = 5
        filename = build_darkroom_model_filename(envname, model_config)
        
    elif envname == 'miniworld':
        state_dim = 2
        action_dim = 4
        filename = build_miniworld_model_filename(envname, model_config)
    else:
        raise NotImplementedError

    config = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
    }

    # Load network from saved file
    if envname == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)
    
    # Use the required model_path parameter directly
    model_path = args['model_path']

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Generate environments instead of loading data
    print(f"Generating {n_eval} evaluation environments...")
    
    if envname in ['bandit', 'bandit_bernoulli']:
        eval_envs = generate_bandit_environments(n_eval, dim, var, bandit_type)
        save_filename = f'{os.path.splitext(os.path.basename(model_path))[0]}_testcov{test_cov}_hor{horizon}'
        
    elif envname == 'linear_bandit':
        eval_envs = generate_linear_bandit_environments(n_eval, dim, lin_d, var)
        save_filename = f'{os.path.splitext(os.path.basename(model_path))[0]}_testcov{test_cov}_hor{horizon}'
        
    elif envname.startswith('darkroom'):
        eval_envs = generate_darkroom_environments(n_eval, dim, envname)
        save_filename = f'{os.path.splitext(os.path.basename(model_path))[0]}_hor{horizon}'
        
    elif envname == 'miniworld':
        eval_envs = generate_miniworld_environments(n_eval)
        save_filename = f'{os.path.splitext(os.path.basename(model_path))[0]}_hor{horizon}'
    else:
        raise ValueError(f'Environment {envname} not supported')

    # Convert environments to trajectory format
    eval_trajs = convert_envs_to_trajs_format(eval_envs, envname, horizon)
    n_eval = len(eval_trajs)
    print(f"Generated {n_eval} evaluation trajectories")

    # Create output directories organized by model
    base_eval_dir = f"evals_online_models"
    model_specific_dir = f"{base_eval_dir}/{os.path.basename(model_path)}"
    
    # Create directory structure
    for subdir in ['', '/bar', '/online', '/graph']:
        dir_path = f'figs/{model_specific_dir}{subdir}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    
    evals_filename = model_specific_dir

    # Run evaluations (same as original eval.py)
    print("Running evaluations...")
    
    if envname == 'bandit' or envname == 'bandit_bernoulli':
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
            'bandit_type': bandit_type,
        }
        print("Running online bandit evaluation...")
        eval_bandit.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()
        plt.cla()
        plt.close()

        print("Running offline bandit evaluation...")
        eval_bandit.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

        print("Running offline graph bandit evaluation...")
        eval_bandit.offline_graph(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/graph/{save_filename}_graph.png')
        plt.clf()
        
    elif envname == 'linear_bandit':
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
        }
        print("Running online linear bandit evaluation...")
        eval_linear_bandit.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()
        plt.cla()
        plt.close()

        print("Running offline linear bandit evaluation...")
        eval_linear_bandit.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

        print("Running offline graph linear bandit evaluation...")
        eval_linear_bandit.offline_graph(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/graph/{save_filename}_graph.png')
        plt.clf()

    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        config = {
            'Heps': 40,
            'horizon': horizon,
            'H': H,
            'n_eval': min(20, n_eval),
            'dim': dim,
            'permuted': True if envname == 'darkroom_permuted' else False,
        }
        print("Running online darkroom evaluation...")
        eval_darkroom.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()

        del config['Heps']
        del config['horizon']
        config['n_eval'] = n_eval
        print("Running offline darkroom evaluation...")
        eval_darkroom.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

    elif envname == 'miniworld':
        from evals import eval_miniworld
        save_video = args.get('save_video', False)
        filename_prefix = f'videos/{save_filename}/{evals_filename}/'
        config = {
            'Heps': 40,
            'horizon': horizon,
            'H': H,
            'n_eval': min(20, n_eval),
            'save_video': save_video,
            'filename_template': filename_prefix + '{controller}_env{env_id}_ep{ep}_online.gif',
        }

        if save_video and not os.path.exists(f'videos/{save_filename}/{evals_filename}'):
            os.makedirs(f'videos/{save_filename}/{evals_filename}', exist_ok=True)

        print("Running online miniworld evaluation...")
        eval_miniworld.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()

        del config['Heps']
        del config['horizon']
        del config['H']
        config['n_eval'] = n_eval
        config['filename_template'] = filename_prefix + '{controller}_env{env_id}_offline.gif'
        
        print("Running offline miniworld evaluation...")
        start_time = time.time()
        eval_miniworld.offline(eval_trajs, model, **config)
        print(f'Offline evaluation took {time.time() - start_time} seconds')
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

    print(f"Evaluation complete! Results saved in figs/{evals_filename}/")
    print(f"Online plots: figs/{evals_filename}/online/")
    print(f"Bar plots: figs/{evals_filename}/bar/")
    print(f"Graph plots: figs/{evals_filename}/graph/")
    print(f"Results organized in: figs/{base_eval_dir}/")
