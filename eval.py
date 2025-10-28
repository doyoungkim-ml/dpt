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
    build_bandit_data_filename,
    build_bandit_model_filename,
    build_linear_bandit_data_filename,
    build_linear_bandit_model_filename,
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
)
import numpy as np
import scipy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    n_envs = args.get('envs', 100000)
    n_hists = args.get('hists', 1)
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
    H = horizon
    n_eval = args.get('n_eval', 100)
    seed = args.get('seed', 0)
    lin_d = args.get('lin_d', 2)  # Only needed for linear_bandit
    
    tmp_seed = seed
    if seed == -1:
        tmp_seed = 0

    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)

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

    # Load network from saved file.
    # By default, load the final file, otherwise load specified epoch.
    if envname == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)
    
    # Use the required model_path parameter
    model_path = args['model_path']
    
    

    
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset_config = {
        'horizon': horizon,
        'dim': dim,
    }
    if envname in ['bandit', 'bandit_bernoulli']:
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        eval_filepath = build_bandit_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{filename}_testcov{test_cov}_hor{horizon}.pkl'
    elif envname in ['linear_bandit']:
        dataset_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        eval_filepath = build_linear_bandit_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{filename}_testcov{test_cov}_hor{horizon}.pkl'
    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        dataset_config.update({'rollin_type': 'uniform'})
        eval_filepath = build_darkroom_data_filename(
            envname, n_eval, dataset_config, mode=2)
        save_filename = f'{filename}_hor{horizon}.pkl'
    elif envname == 'miniworld':
        dataset_config.update({'rollin_type': 'uniform'})        
        eval_filepath = build_miniworld_data_filename(
            envname, 0, n_eval, dataset_config, mode=2)
        save_filename = f'{filename}_hor{horizon}.pkl'
    else:
        raise ValueError(f'Environment {envname} not supported')


    with open(eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)

    n_eval = min(n_eval, len(eval_trajs))


    evals_filename = f"evals_{model_path}"
    if not os.path.exists(f'figs/{evals_filename}'):
        os.makedirs(f'figs/{evals_filename}', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/bar'):
        os.makedirs(f'figs/{evals_filename}/bar', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/online'):
        os.makedirs(f'figs/{evals_filename}/online', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/graph'):
        os.makedirs(f'figs/{evals_filename}/graph', exist_ok=True)
    if not os.path.exists(f'figs/{evals_filename}/entropy'):
        os.makedirs(f'figs/{evals_filename}/entropy', exist_ok=True)

    # Online and offline evaluation.
    if envname == 'bandit' or envname == 'bandit_bernoulli':
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
            'bandit_type': bandit_type,
        }
        eval_bandit.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()
        plt.cla()
        plt.close()

        eval_bandit.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

        eval_bandit.offline_graph(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/graph/{save_filename}_graph.png')
        plt.clf()
        
        # Compute entropy metrics during online episodes (with baselines)
        entropy_results = eval_bandit.policy_entropy_online(eval_trajs, model, **config, include_baselines=True)
        
        # Plot entropy and cross-entropy over episodes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        episode_range = np.arange(len(entropy_results['lnr']['entropy_mean']))
        
        # Baseline colors
        baseline_color_map = {
            'lnr': 'tab:red',
            'emp': 'tab:blue',
            'ucb': 'tab:orange',
            'thmp': 'black',
        }
        
        # Plot entropy for all methods
        for method_name, method_data in entropy_results.items():
            color = baseline_color_map.get(method_name, 'gray')
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
        
        # Plot cross-entropy for all methods
        for method_name, method_data in entropy_results.items():
            color = baseline_color_map.get(method_name, 'gray')
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
        plt.savefig(f'figs/{evals_filename}/entropy/{save_filename}_entropy.png')
        plt.clf()
        plt.close()
        
    elif envname == 'linear_bandit':
        config = {
            'horizon': horizon,
            'var': var,
            'n_eval': n_eval,
        }

        with open(eval_filepath, 'rb') as f:
            eval_trajs = pickle.load(f)

        eval_linear_bandit.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()
        plt.cla()
        plt.close()

        eval_linear_bandit.offline(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()

        eval_linear_bandit.offline_graph(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/graph/{save_filename}_graph.png')
        plt.clf()

        # Compute entropy metrics during online episodes (with baselines)
        entropy_results = eval_linear_bandit.policy_entropy_online(eval_trajs, model, **config, include_baselines=True)
        
        # Plot entropy and cross-entropy over episodes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        episode_range = np.arange(len(entropy_results['lnr']['entropy_mean']))
        
        # Baseline colors
        baseline_color_map = {
            'lnr': 'tab:red',
            'thmp': 'black',
            'linucb': 'tab:orange',
        }
        
        # Plot entropy for all methods
        for method_name, method_data in entropy_results.items():
            color = baseline_color_map.get(method_name, 'gray')
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
        
        # Plot cross-entropy for all methods
        for method_name, method_data in entropy_results.items():
            color = baseline_color_map.get(method_name, 'gray')
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
        plt.savefig(f'figs/{evals_filename}/entropy/{save_filename}_entropy.png')
        plt.clf()
        plt.close()


    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        config = {
            'Heps': 40,
            'horizon': horizon,
            'H': H,
            'n_eval': min(20, n_eval),
            'dim': dim,
            'permuted': True if envname == 'darkroom_permuted' else False,
        }
        eval_darkroom.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()

        del config['Heps']
        del config['horizon']
        config['n_eval'] = n_eval
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
            os.makedirs(
                f'videos/{save_filename}/{evals_filename}', exist_ok=True)

        eval_miniworld.online(eval_trajs, model, **config)
        plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
        plt.clf()

        del config['Heps']
        del config['horizon']
        del config['H']
        config['n_eval'] = n_eval
        config['filename_template'] = filename_prefix + \
            '{controller}_env{env_id}_offline.gif'
        start_time = time.time()
        eval_miniworld.offline(eval_trajs, model, **config)
        print(f'Offline evaluation took {time.time() - start_time} seconds')
        plt.savefig(f'figs/{evals_filename}/bar/{save_filename}_bar.png')
        plt.clf()
