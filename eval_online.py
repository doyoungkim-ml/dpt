import argparse
import os
import pickle

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
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--online_suffix', type=str, default='_online', 
                       help='Suffix for online model files (e.g., "_online")')
    parser.add_argument('--confidence_type', type=str, default='', 
                       choices=['', 'linear', 'exponential', 'stepped', 'constant'],
                       help='Type of confidence function used in training')
    parser.add_argument('--confidence_start', type=float, default=0.3,
                       help='Starting confidence value for linear and stepped functions')
    parser.add_argument('--confidence_lambda', type=float, default=40,
                       help='Lambda parameter for exponential confidence')
    parser.add_argument('--max_position', type=int, default=-1,
                       help='Maximum position for stepped confidence')
    parser.add_argument('--confidence_value', type=float, default=1.0,
                       help='Constant confidence value')
    
    # Backward compatibility arguments
    parser.add_argument('--confidence', action='store_true',
                       help='Use linear confidence-trained model (deprecated, use --confidence_type linear)')
    parser.add_argument('--confidence_exp', action='store_true',
                       help='Use exponential confidence-trained model (deprecated, use --confidence_type exponential)')

    args = vars(parser.parse_args())
    print("Args: ", args)

    n_envs = args['envs']
    n_hists = args['hists']
    H = args['H']
    n_samples = args['samples']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
    horizon = args['hor']
    n_eval = args['n_eval']
    seed = args['seed']
    lin_d = args['lin_d']
    online_suffix = args['online_suffix']
    confidence_type = args['confidence_type']
    confidence_start = args['confidence_start']
    confidence_lambda = args['confidence_lambda']
    max_position = args['max_position']
    confidence_value = args['confidence_value']
    
    # Backward compatibility
    confidence = args['confidence']
    confidence_exp = args['confidence_exp']
    
    # Determine confidence type from arguments
    if confidence_exp and not confidence_type:
        confidence_type = 'exponential'
    elif confidence and not confidence_type:
        confidence_type = 'linear'
    
    # Determine confidence suffix and folder name
    if confidence_type:
        confidence_suffix = f"_{confidence_type}"
        if confidence_type in ['linear', 'stepped']:
            confidence_suffix += f"_start{confidence_start}"
        if confidence_type == 'exponential':
            confidence_suffix += f"_lambda{confidence_lambda}"
        if confidence_type == 'stepped':
            if max_position <= 0:
                max_position = horizon // 2 if horizon > 0 else 250  # default fallback
            confidence_suffix += f"_maxpos{max_position}"
        if confidence_type == 'constant':
            confidence_suffix += f"_val{confidence_value}"
        
        confidence_folder_name = confidence_type
    else:
        confidence_suffix = ""
        confidence_folder_name = "standard"
    
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
    
    tmp_filename = filename
    # Use appropriate suffix based on confidence type
    if confidence_type:
        # Update filename to include confidence parameters
        tmp_filename = tmp_filename.replace('.pt', f'{confidence_suffix}.pt')
        model_suffix = '_online_unified'
    else:
        model_suffix = online_suffix
    
    if epoch < 0:
        model_path = f'models/{tmp_filename}{model_suffix}.pt'
    else:
        model_path = f'models/{tmp_filename}{model_suffix}_epoch{epoch}.pt'
    
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

    # Create output directories organized by confidence function type
    base_eval_dir = f"evals_online_models"
    confidence_dir = f"{base_eval_dir}/{confidence_folder_name}"
    model_specific_dir = f"{confidence_dir}/{os.path.basename(model_path)}"
    
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
        save_video = args['save_video']
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
    print(f"Confidence type: {confidence_type if confidence_type else 'standard'}")
    print(f"Online plots: figs/{evals_filename}/online/")
    print(f"Bar plots: figs/{evals_filename}/bar/")
    print(f"Graph plots: figs/{evals_filename}/graph/")
    print(f"Results organized by confidence function in: figs/{base_eval_dir}/")
