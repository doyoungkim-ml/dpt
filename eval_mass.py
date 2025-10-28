import argparse
import os
import pickle
import yaml
import re
from collections import defaultdict

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
    convert_to_tensor,
)
import numpy as np
import scipy
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_epoch_number(checkpoint_path):
    """Extract epoch number from checkpoint filename."""
    match = re.search(r'epoch(\d+)\.pt', checkpoint_path)
    if match:
        return int(match.group(1))
    return None


def get_checkpoint_files(checkpoint_dir):
    """Get all checkpoint files sorted by epoch number."""
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt') and filename.startswith('epoch'):
            epoch_num = extract_epoch_number(filename)
            if epoch_num is not None:
                checkpoints.append((epoch_num, os.path.join(checkpoint_dir, filename)))

    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def evaluate_checkpoint(model_path, args, model, eval_trajs, envname, config, dataset_config):
    """Evaluate a single checkpoint and return results."""

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    results = {}

    if envname == 'bandit' or envname == 'bandit_bernoulli':
        eval_config = {
            'horizon': config['horizon'],
            'var': args.get('var', 0.0),
            'n_eval': args.get('n_eval', 100),
            'bandit_type': 'uniform' if envname == 'bandit' else 'bernoulli',
        }

        # Get online results
        online_fig, online_data = eval_bandit.online(eval_trajs, model, **eval_config, return_data=True)
        results['online'] = online_data
        plt.close(online_fig)

        # Get offline bar results
        bar_fig, bar_data = eval_bandit.offline(eval_trajs, model, **eval_config, return_data=True)
        results['bar'] = bar_data
        plt.close(bar_fig)

        # Get offline graph results
        graph_fig, graph_data = eval_bandit.offline_graph(eval_trajs, model, **eval_config, return_data=True)
        results['graph'] = graph_data
        plt.close(graph_fig)

    elif envname == 'linear_bandit':
        eval_config = {
            'horizon': config['horizon'],
            'var': args.get('var', 0.0),
            'n_eval': args.get('n_eval', 100),
        }

        # Get online results
        online_fig, online_data = eval_linear_bandit.online(eval_trajs, model, **eval_config, return_data=True)
        results['online'] = online_data
        plt.close(online_fig)

        # Get offline bar results
        bar_fig, bar_data = eval_linear_bandit.offline(eval_trajs, model, **eval_config, return_data=True)
        results['bar'] = bar_data
        plt.close(bar_fig)

        # Get offline graph results
        graph_fig, graph_data = eval_linear_bandit.offline_graph(eval_trajs, model, **eval_config, return_data=True)
        results['graph'] = graph_data
        plt.close(graph_fig)

    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        eval_config = {
            'Heps': 40,
            'horizon': config['horizon'],
            'H': args.get('H', 100),
            'n_eval': min(20, args.get('n_eval', 100)),
            'dim': args.get('dim', 10),
            'permuted': True if envname == 'darkroom_permuted' else False,
        }

        # Get online results
        online_fig, online_data = eval_darkroom.online(eval_trajs, model, **eval_config, return_data=True)
        results['online'] = online_data
        plt.close(online_fig)

        # Get offline bar results
        del eval_config['Heps']
        del eval_config['horizon']
        eval_config['n_eval'] = args.get('n_eval', 100)
        bar_fig, bar_data = eval_darkroom.offline(eval_trajs, model, **eval_config, return_data=True)
        results['bar'] = bar_data
        plt.close(bar_fig)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing model checkpoints')

    cmd_args = vars(parser.parse_args())

    # Load config from YAML file
    with open(cmd_args['config'], 'r') as f:
        config_args = yaml.safe_load(f)

    # Use config values as args
    args = config_args
    checkpoint_dir = cmd_args['checkpoint_dir']

    print("Args: ", args)
    print("Checkpoint directory: ", checkpoint_dir)

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
    lin_d = args.get('lin_d', 2)

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

    # Create model
    if envname == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)

    # Load evaluation dataset
    dataset_config = {
        'horizon': horizon,
        'dim': dim,
    }

    if envname in ['bandit', 'bandit_bernoulli']:
        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        eval_filepath = build_bandit_data_filename(envname, n_eval, dataset_config, mode=2)
    elif envname in ['linear_bandit']:
        dataset_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        eval_filepath = build_linear_bandit_data_filename(envname, n_eval, dataset_config, mode=2)
    elif envname in ['darkroom_heldout', 'darkroom_permuted']:
        dataset_config = {
            'n_hists': n_hists,
            'n_samples': n_samples,
            'horizon': horizon,
            'dim': dim,
        }
        eval_filepath = build_darkroom_data_filename(envname, n_eval, dataset_config, mode=1)
    elif envname == 'miniworld':
        dataset_config.update({'rollin_type': 'uniform'})
        eval_filepath = build_miniworld_data_filename(envname, 0, n_eval, dataset_config, mode=2)
    else:
        raise ValueError(f'Environment {envname} not supported')

    with open(eval_filepath, 'rb') as f:
        eval_trajs = pickle.load(f)

    n_eval = min(n_eval, len(eval_trajs))

    # Get all checkpoints
    checkpoints = get_checkpoint_files(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    # Storage for all results across epochs
    all_online_results = {}  # epoch -> results dict
    all_bar_results = {}     # epoch -> baselines dict
    all_graph_results = {}   # epoch -> (horizons, regrets) data
    all_entropy_results = {} # epoch -> entropy/cross-entropy metrics
    epochs = []

    # Storage for non-learnable baselines (computed once)
    baseline_online_results = None
    baseline_bar_results = None
    baseline_graph_results = None
    baseline_entropy_results = None

    # Create output directories
    # Use normpath to remove trailing slashes, then get basename
    if checkpoint_dir.endswith('/'):
        checkpoint_dir = checkpoint_dir[:-1]
    mass_eval_dir = f"figs/mass_eval_{checkpoint_dir}"
    os.makedirs(f"{mass_eval_dir}/bar", exist_ok=True)
    os.makedirs(f"{mass_eval_dir}/online", exist_ok=True)
    os.makedirs(f"{mass_eval_dir}/graph", exist_ok=True)
    os.makedirs(f"{mass_eval_dir}/entropy", exist_ok=True)

    print(f"Output directory: {mass_eval_dir}")

    # Evaluate each checkpoint and collect data
    for epoch_num, checkpoint_path in checkpoints:
        print(f"\nEvaluating epoch {epoch_num}: {checkpoint_path}")
        epochs.append(epoch_num)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        if envname == 'bandit' or envname == 'bandit_bernoulli':
            eval_config = {
                'horizon': horizon,
                'var': var,
                'n_eval': n_eval,
                'bandit_type': 'uniform' if envname == 'bandit' else 'bernoulli',
            }

            # Collect data without plotting
            print(f"  Running evaluations...")

            # Online results - compute baselines only once
            from ctrls.ctrl_bandit import (BanditTransformerController, OptPolicy,
                                          EmpMeanPolicy, ThompsonSamplingPolicy, UCBPolicy)
            from envs.bandit_env import BanditEnvVec, BanditEnv
            from evals.eval_bandit import deploy_online_vec

            if baseline_online_results is None:
                print(f"  Computing non-learnable baselines (once)...")
                envs = []
                for i_eval in range(n_eval):
                    traj = eval_trajs[i_eval]
                    means = traj['means']
                    env = BanditEnv(means, horizon, var=var)
                    envs.append(env)

                vec_env = BanditEnvVec(envs)

                # Get results for non-learnable baselines
                all_means = {}

                opt_controller = OptPolicy(envs, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, opt_controller, horizon).T
                all_means['opt'] = cum_means

                emp_controller = EmpMeanPolicy(envs[0], online=True, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, emp_controller, horizon).T
                all_means['emp'] = cum_means

                ucb_controller = UCBPolicy(envs[0], const=1.0, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, ucb_controller, horizon).T
                all_means['ucb'] = cum_means

                thmp_controller = ThompsonSamplingPolicy(envs[0], std=var, sample=True,
                                                         prior_mean=0.5, prior_var=1/12.0,
                                                         warm_start=False, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, thmp_controller, horizon).T
                all_means['thmp'] = cum_means

                # Calculate regrets and cumulative regrets
                all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}
                cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}

                # Store mean and sem for each baseline
                baseline_online_results = {}
                for key in all_means_diff.keys():
                    if key != 'opt':
                        baseline_online_results[key] = {
                            'regret_mean': np.mean(all_means_diff[key], axis=0),
                            'regret_sem': scipy.stats.sem(all_means_diff[key], axis=0),
                            'cumregret_mean': np.mean(cumulative_regret[key], axis=0),
                            'cumregret_sem': scipy.stats.sem(cumulative_regret[key], axis=0)
                        }

            # Get lnr results for this epoch
            envs = []
            for i_eval in range(n_eval):
                traj = eval_trajs[i_eval]
                means = traj['means']
                env = BanditEnv(means, horizon, var=var)
                envs.append(env)

            vec_env = BanditEnvVec(envs)

            opt_controller = OptPolicy(envs, batch_size=len(envs))
            opt_cum_means = deploy_online_vec(vec_env, opt_controller, horizon).T

            lnr_controller = BanditTransformerController(model, sample=True, batch_size=len(envs))
            lnr_cum_means = deploy_online_vec(vec_env, lnr_controller, horizon).T

            # Calculate lnr regret
            lnr_regret = opt_cum_means - lnr_cum_means
            lnr_cumregret = np.cumsum(lnr_regret, axis=1)

            all_online_results[epoch_num] = {
                'lnr': {
                    'regret_mean': np.mean(lnr_regret, axis=0),
                    'regret_sem': scipy.stats.sem(lnr_regret, axis=0),
                    'cumregret_mean': np.mean(lnr_cumregret, axis=0),
                    'cumregret_sem': scipy.stats.sem(lnr_cumregret, axis=0)
                }
            }

            # Bar results
            bar_results = eval_bandit.offline(eval_trajs, model, **eval_config)
            all_bar_results[epoch_num] = bar_results
            plt.close()

            # Graph results - need to collect regret data across horizons
            horizons = np.linspace(1, horizon, 50, dtype=int)
            regret_data = {}
            regret_sem_data = {}
            for h in horizons:
                h_config = eval_config.copy()
                h_config['horizon'] = h
                h_baselines = eval_bandit.offline(eval_trajs, model, **h_config)
                plt.close()

                # Store regrets and standard errors for this horizon
                for key in h_baselines.keys():
                    if key not in regret_data:
                        regret_data[key] = []
                        regret_sem_data[key] = []
                    regret_data[key].append(np.mean(h_baselines[key]))
                    regret_sem_data[key].append(scipy.stats.sem(h_baselines[key]))

            all_graph_results[epoch_num] = (horizons, regret_data, regret_sem_data)

            # Entropy results - compute online entropy at each timestep
            # Compute baselines once
            if baseline_entropy_results is None:
                print(f"  Computing baseline entropies (once)...")
                baseline_entropy_results = eval_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=True)
                # Extract just the baseline parts
                baseline_entropy_results = {k: v for k, v in baseline_entropy_results.items() if k != 'lnr'}
            
            # Compute lnr entropy for this epoch
            entropy_results_lnr = eval_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=False)
            all_entropy_results[epoch_num] = entropy_results_lnr['lnr']

        elif envname == 'linear_bandit':
            eval_config = {
                'horizon': horizon,
                'var': var,
                'n_eval': n_eval,
            }

            print(f"  Running evaluations...")

            # Online results - compute baselines only once
            from ctrls.ctrl_bandit import BanditTransformerController, OptPolicy, ThompsonSamplingPolicy, LinUCBPolicy
            from envs.bandit_env import BanditEnvVec, LinearBanditEnv
            from evals.eval_linear_bandit import deploy_online_vec

            if baseline_online_results is None:
                print(f"  Computing non-learnable baselines (once)...")
                envs = []
                for i_eval in range(n_eval):
                    traj = eval_trajs[i_eval]
                    env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
                    envs.append(env)

                vec_env = BanditEnvVec(envs)

                # Get results for non-learnable baselines
                all_means = {}

                opt_controller = OptPolicy(envs, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, opt_controller, horizon).T
                all_means['opt'] = cum_means

                thmp_controller = ThompsonSamplingPolicy(envs[0], std=var, sample=True,
                                                         prior_mean=0.0, prior_var=1.0,
                                                         warm_start=False, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, thmp_controller, horizon).T
                all_means['thmp'] = cum_means

                linucb_controller = LinUCBPolicy(envs[0], const=1.0, batch_size=len(envs))
                cum_means = deploy_online_vec(vec_env, linucb_controller, horizon).T
                all_means['linucb'] = cum_means

                # Calculate regrets and cumulative regrets
                all_means_diff = {k: all_means['opt'] - v for k, v in all_means.items()}
                cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}

                # Store mean and sem for each baseline
                baseline_online_results = {}
                for key in all_means_diff.keys():
                    if key != 'opt':
                        baseline_online_results[key] = {
                            'regret_mean': np.mean(all_means_diff[key], axis=0),
                            'regret_sem': scipy.stats.sem(all_means_diff[key], axis=0),
                            'cumregret_mean': np.mean(cumulative_regret[key], axis=0),
                            'cumregret_sem': scipy.stats.sem(cumulative_regret[key], axis=0)
                        }

            # Get lnr results for this epoch
            envs = []
            for i_eval in range(n_eval):
                traj = eval_trajs[i_eval]
                env = LinearBanditEnv(traj['theta'], traj['arms'], horizon, var=var)
                envs.append(env)

            vec_env = BanditEnvVec(envs)

            opt_controller = OptPolicy(envs, batch_size=len(envs))
            opt_cum_means = deploy_online_vec(vec_env, opt_controller, horizon).T

            lnr_controller = BanditTransformerController(model, sample=True, batch_size=len(envs))
            lnr_cum_means = deploy_online_vec(vec_env, lnr_controller, horizon).T

            # Calculate lnr regret
            lnr_regret = opt_cum_means - lnr_cum_means
            lnr_cumregret = np.cumsum(lnr_regret, axis=1)

            all_online_results[epoch_num] = {
                'lnr': {
                    'regret_mean': np.mean(lnr_regret, axis=0),
                    'regret_sem': scipy.stats.sem(lnr_regret, axis=0),
                    'cumregret_mean': np.mean(lnr_cumregret, axis=0),
                    'cumregret_sem': scipy.stats.sem(lnr_cumregret, axis=0)
                }
            }

            # Bar results
            bar_results = eval_linear_bandit.offline(eval_trajs, model, **eval_config)
            all_bar_results[epoch_num] = bar_results
            plt.close()

            # Graph results
            horizons = np.linspace(1, horizon, 50, dtype=int)
            regret_data = {}
            regret_sem_data = {}
            for h in horizons:
                h_config = eval_config.copy()
                h_config['horizon'] = h
                h_baselines = eval_linear_bandit.offline(eval_trajs, model, **h_config)
                plt.close()

                for key in h_baselines.keys():
                    if key not in regret_data:
                        regret_data[key] = []
                        regret_sem_data[key] = []
                    regret_data[key].append(np.mean(h_baselines[key]))
                    regret_sem_data[key].append(scipy.stats.sem(h_baselines[key]))

            all_graph_results[epoch_num] = (horizons, regret_data, regret_sem_data)

            # Entropy results - compute online entropy at each timestep
            # Compute baselines once
            if baseline_entropy_results is None:
                print(f"  Computing baseline entropies (once)...")
                baseline_entropy_results = eval_linear_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=True)
                # Extract just the baseline parts
                baseline_entropy_results = {k: v for k, v in baseline_entropy_results.items() if k != 'lnr'}
            
            # Compute lnr entropy for this epoch
            entropy_results_lnr = eval_linear_bandit.policy_entropy_online(eval_trajs, model, **eval_config, include_baselines=False)
            all_entropy_results[epoch_num] = entropy_results_lnr['lnr']

        elif envname in ['darkroom_heldout', 'darkroom_permuted']:
            eval_config = {
                'Heps': 40,
                'horizon': horizon,
                'H': H,
                'n_eval': min(20, n_eval),
                'dim': dim,
                'permuted': True if envname == 'darkroom_permuted' else False,
            }

            print(f"  Running evaluations...")

            # Online results
            from ctrls.ctrl_darkroom import DarkroomOptPolicy, DarkroomTransformerController
            from envs.darkroom_env import DarkroomEnv, DarkroomEnvPermuted, DarkroomEnvVec
            from evals.eval_darkroom import deploy_online_vec

            # Create environments
            envs = []
            for i_eval in range(eval_config['n_eval']):
                traj = eval_trajs[i_eval]
                if eval_config['permuted']:
                    env = DarkroomEnvPermuted(dim, traj['perm_index'], horizon)
                else:
                    env = DarkroomEnv(dim, traj['goal'], horizon)
                envs.append(env)

            vec_env = DarkroomEnvVec(envs)

            # Get learner results
            lnr_controller = DarkroomTransformerController(model, batch_size=eval_config['n_eval'], sample=True)
            cum_means_lnr = deploy_online_vec(vec_env, lnr_controller, eval_config['Heps'], eval_config['H'], horizon)
            
            all_online_results[epoch_num] = {
                'lnr': {
                    'mean': np.mean(cum_means_lnr, axis=0),
                    'sem': scipy.stats.sem(cum_means_lnr, axis=0),
                    'all_data': cum_means_lnr
                }
            }

            # Bar results
            del eval_config['Heps']
            del eval_config['horizon']
            eval_config['n_eval'] = n_eval
            
            # Run offline evaluation
            all_rs_opt = []
            all_rs_lnr = []
            all_rs_lnr_greedy = []

            envs = []
            trajs = []

            for i_eval in range(n_eval):
                traj = eval_trajs[i_eval]
                
                if eval_config['permuted']:
                    env = DarkroomEnvPermuted(dim, traj['perm_index'], eval_config['H'])
                else:
                    env = DarkroomEnv(dim, traj['goal'], eval_config['H'])

                batch = {
                    'context_states': convert_to_tensor(traj['context_states'][None, :, :]),
                    'context_actions': convert_to_tensor(traj['context_actions'][None, :, :]),
                    'context_next_states': convert_to_tensor(traj['context_next_states'][None, :, :]),
                    'context_rewards': convert_to_tensor(traj['context_rewards'][None, :, None]),
                }

                true_opt = DarkroomOptPolicy(env)
                true_opt.set_batch(batch)
                _, _, _, rs_opt = env.deploy_eval(true_opt)
                all_rs_opt.append(np.sum(rs_opt))

                envs.append(env)
                trajs.append(traj)

            # Run learner evaluations in parallel
            vec_env = DarkroomEnvVec(envs)
            lnr = DarkroomTransformerController(model, batch_size=n_eval, sample=True)
            lnr_greedy = DarkroomTransformerController(model, batch_size=n_eval, sample=False)

            batch = {
                'context_states': convert_to_tensor([traj['context_states'] for traj in trajs]),
                'context_actions': convert_to_tensor([traj['context_actions'] for traj in trajs]),
                'context_next_states': convert_to_tensor([traj['context_next_states'] for traj in trajs]),
                'context_rewards': convert_to_tensor([traj['context_rewards'][:, None] for traj in trajs]),
            }
            lnr.set_batch(batch)
            lnr_greedy.set_batch(batch)

            _, _, _, rs_lnr = vec_env.deploy_eval(lnr)
            _, _, _, rs_lnr_greedy = vec_env.deploy_eval(lnr_greedy)
            all_rs_lnr = np.sum(rs_lnr, axis=-1)
            all_rs_lnr_greedy = np.sum(rs_lnr_greedy, axis=-1)

            all_bar_results[epoch_num] = {
                'opt': np.array(all_rs_opt),
                'lnr': np.array(all_rs_lnr),
                'lnr_greedy': np.array(all_rs_lnr_greedy)
            }

    # Now create the 3 aggregate plots
    print(f"\nCreating aggregate plots across {len(epochs)} epochs...")

    # 1. BAR PLOT: Show all epochs as grouped bars
    if len(all_bar_results) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define consistent colors for baselines
        baseline_color_map = {
            'emp': 'tab:blue',
            'ucb': 'tab:orange',
            'linucb': 'tab:orange',
            'thmp': 'black',
            'lcb': 'tab:red',
            'lnr_greedy': 'tab:cyan',
            'opt': 'tab:green'
        }

        # Get all baseline names from first epoch
        all_keys = list(all_bar_results[epochs[0]].keys())
        
        # Order baselines appropriately
        if 'lnr' in all_keys and 'lnr_greedy' in all_keys:
            # Darkroom case: lnr, lnr_greedy, opt
            baseline_names = ['lnr', 'lnr_greedy', 'opt']
        else:
            # Bandit case: lnr first, then non-learnable baselines (excluding opt)
            baseline_names = [k for k in all_keys if k not in ['opt', 'lnr']]
            baseline_names = ['lnr'] + baseline_names

        x = np.arange(len(baseline_names))
        width = 0.8 / len(epochs)  # Width of bars

        # Color map for epochs
        epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

        # Plot bars for each epoch
        for i, epoch_num in enumerate(epochs):
            means = []
            for name in baseline_names:
                means.append(np.mean(all_bar_results[epoch_num][name]))

            offset = width * (i - len(epochs)/2)

            # Only add legend label for first epoch (showing range)
            if i == 0:
                label = f'lnr (epochs {epochs[0]}-{epochs[-1]})'
            else:
                label = None

            # For lnr, use epoch color; for others use baseline color
            bar_colors = []
            for name in baseline_names:
                if name == 'lnr':
                    bar_colors.append(epoch_colors[i])
                else:
                    bar_colors.append(baseline_color_map.get(name, 'gray'))

            ax.bar(x + offset, means, width, color=bar_colors, label=label, alpha=0.8 if name != 'lnr' else 0.7)

        ax.set_xlabel('Baseline')
        ax.set_ylabel('Mean Reward')
        ax.set_title(f'Performance Comparison Across Training Epochs')
        ax.set_xticks(x)
        ax.set_xticklabels(baseline_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{mass_eval_dir}/bar/mass_eval_bar.png', dpi=150)
        plt.close()
        print(f"  Saved: {mass_eval_dir}/bar/mass_eval_bar.png")

    # 2. ONLINE PLOT: Two separate figures - suboptimality and cumulative regret (bandit) or returns (darkroom)
    if len(all_online_results) > 0:
        # Check if this is bandit (has baseline_online_results) or darkroom (no baselines)
        is_bandit = baseline_online_results is not None
        
        if is_bandit:
            n_episodes = len(baseline_online_results[list(baseline_online_results.keys())[0]]['regret_mean'])
        else:
            # Darkroom case - get from first epoch
            n_episodes = len(all_online_results[epochs[0]]['lnr']['mean'])
        
        episode_range = np.arange(n_episodes)

        # Define consistent colors for baselines
        baseline_color_map = {
            'emp': 'tab:blue',
            'ucb': 'tab:orange',
            'linucb': 'tab:orange',
            'thmp': 'black',
            'lcb': 'tab:red'
        }

        # Color map for epochs
        epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

        if is_bandit:
            # BANDIT CASE: Plot suboptimality and cumulative regret with baselines
            # FIGURE 1: Suboptimality (instant regret)
            fig1, ax1 = plt.subplots(figsize=(12, 6))

            # Plot lnr for each epoch first (so baselines appear on top)
            for i, epoch_num in enumerate(epochs):
                regret_mean = all_online_results[epoch_num]['lnr']['regret_mean']
                regret_sem = all_online_results[epoch_num]['lnr']['regret_sem']

                # Only add label for first epoch, showing the full range
                if i == 0:
                    label = f'lnr (epochs {epochs[0]}-{epochs[-1]})'
                else:
                    label = None

                ax1.plot(episode_range, regret_mean, label=label,
                        color=epoch_colors[i], linewidth=2, alpha=0.7)
                ax1.fill_between(episode_range, regret_mean - regret_sem, regret_mean + regret_sem,
                                alpha=0.15, color=epoch_colors[i])

            # Plot non-learnable baselines on top (with confidence bands)
            for baseline_name in baseline_online_results.keys():
                regret_mean = baseline_online_results[baseline_name]['regret_mean']
                regret_sem = baseline_online_results[baseline_name]['regret_sem']

                color = baseline_color_map.get(baseline_name, 'gray')
                ax1.plot(episode_range, regret_mean, label=baseline_name, linestyle='--',
                        linewidth=2, color=color)
                ax1.fill_between(episode_range, regret_mean - regret_sem, regret_mean + regret_sem,
                               alpha=0.2, color=color)

            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Suboptimality')
            ax1.set_title('Online Evaluation: Suboptimality')
            ax1.set_yscale('log')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{mass_eval_dir}/online/mass_eval_online_suboptimality.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {mass_eval_dir}/online/mass_eval_online_suboptimality.png")

            # FIGURE 2: Cumulative Regret
            fig2, ax2 = plt.subplots(figsize=(12, 6))

            # Plot lnr for each epoch first (so baselines appear on top)
            for i, epoch_num in enumerate(epochs):
                cumregret_mean = all_online_results[epoch_num]['lnr']['cumregret_mean']
                cumregret_sem = all_online_results[epoch_num]['lnr']['cumregret_sem']

                # Only add label for first epoch, showing the full range
                if i == 0:
                    label = f'lnr (epochs {epochs[0]}-{epochs[-1]})'
                else:
                    label = None

                ax2.plot(episode_range, cumregret_mean, label=label,
                        color=epoch_colors[i], linewidth=2, alpha=0.7)
                ax2.fill_between(episode_range, cumregret_mean - cumregret_sem, cumregret_mean + cumregret_sem,
                                alpha=0.15, color=epoch_colors[i])

            # Plot non-learnable baselines on top (with confidence bands)
            for baseline_name in baseline_online_results.keys():
                cumregret_mean = baseline_online_results[baseline_name]['cumregret_mean']
                cumregret_sem = baseline_online_results[baseline_name]['cumregret_sem']

                color = baseline_color_map.get(baseline_name, 'gray')
                ax2.plot(episode_range, cumregret_mean, label=baseline_name, linestyle='--',
                        linewidth=2, color=color)
                ax2.fill_between(episode_range, cumregret_mean - cumregret_sem, cumregret_mean + cumregret_sem,
                               alpha=0.2, color=color)

            ax2.set_xlabel('Episodes')
            ax2.set_ylabel('Cumulative Regret')
            ax2.set_title('Online Evaluation: Cumulative Regret Over Time')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{mass_eval_dir}/online/mass_eval_online_cumregret.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {mass_eval_dir}/online/mass_eval_online_cumregret.png")
        else:
            # DARKROOM CASE: Plot returns across epochs (no baselines)
            fig1, ax1 = plt.subplots(figsize=(12, 6))

            # Plot lnr for each epoch
            for i, epoch_num in enumerate(epochs):
                returns_mean = all_online_results[epoch_num]['lnr']['mean']
                returns_sem = all_online_results[epoch_num]['lnr']['sem']

                # Only add label for first epoch, showing the full range
                if i == 0:
                    label = f'lnr (epochs {epochs[0]}-{epochs[-1]})'
                else:
                    label = None

                ax1.plot(episode_range, returns_mean, label=label,
                        color=epoch_colors[i], linewidth=2, alpha=0.7)
                ax1.fill_between(episode_range, returns_mean - returns_sem, returns_mean + returns_sem,
                                alpha=0.15, color=epoch_colors[i])

            ax1.set_xlabel('Episodes')
            ax1.set_ylabel('Average Return')
            ax1.set_title('Online Evaluation: Returns Over Episodes')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{mass_eval_dir}/online/mass_eval_online_returns.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {mass_eval_dir}/online/mass_eval_online_returns.png")

    # 3. GRAPH PLOT: Suboptimality vs dataset size (similar to online but for offline)
    if len(all_graph_results) > 0:
        # Create single plot with all baselines
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define consistent colors for baselines
        baseline_color_map = {
            'emp': 'tab:blue',
            'ucb': 'tab:orange',
            'linucb': 'tab:orange',
            'thmp': 'black',
            'lcb': 'tab:red'
        }

        # Color map for epochs
        epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

        # Extract non-learnable baselines from the first epoch's graph results
        # (they should be the same across all epochs since they're not learned)
        first_epoch_horizons, first_epoch_data, first_epoch_sem = all_graph_results[epochs[0]]

        # Plot lnr for each epoch first (so baselines appear on top)
        for i, epoch_num in enumerate(epochs):
            horizons_ep, regret_data_ep, regret_sem_ep = all_graph_results[epoch_num]
            lnr_regrets = np.array([regret_data_ep['opt'][j] - regret_data_ep['lnr'][j]
                          for j in range(len(horizons_ep))])
            lnr_sems = np.array([np.sqrt(regret_sem_ep['opt'][j]**2 + regret_sem_ep['lnr'][j]**2)
                          for j in range(len(horizons_ep))])

            # Only add label for first epoch, showing the full range
            if i == 0:
                label = f'lnr (epochs {epochs[0]}-{epochs[-1]})'
            else:
                label = None

            ax.plot(horizons_ep, lnr_regrets, label=label,
                   color=epoch_colors[i], linewidth=2, alpha=0.7)
            ax.fill_between(horizons_ep, lnr_regrets - lnr_sems, lnr_regrets + lnr_sems,
                          alpha=0.15, color=epoch_colors[i])

        # Plot non-learnable baselines on top with confidence bands
        for baseline_name in first_epoch_data.keys():
            if baseline_name not in ['opt', 'lnr']:
                # Calculate regret (suboptimality)
                regrets = np.array([first_epoch_data['opt'][j] - first_epoch_data[baseline_name][j]
                          for j in range(len(first_epoch_horizons))])

                # Calculate SEM for regret (using error propagation)
                regret_sems = np.array([np.sqrt(first_epoch_sem['opt'][j]**2 + first_epoch_sem[baseline_name][j]**2)
                          for j in range(len(first_epoch_horizons))])

                color = baseline_color_map.get(baseline_name, 'gray')
                ax.plot(first_epoch_horizons, regrets, label=baseline_name, linestyle='--',
                       linewidth=2, color=color)
                ax.fill_between(first_epoch_horizons, regrets - regret_sems, regrets + regret_sems,
                              alpha=0.2, color=color)

        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Suboptimality')
        ax.set_title('Offline Evaluation: Suboptimality vs Dataset Size')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{mass_eval_dir}/graph/mass_eval_graph.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {mass_eval_dir}/graph/mass_eval_graph.png")

    # 4. ENTROPY PLOT: Show entropy and cross-entropy evolution during episodes
    if len(all_entropy_results) > 0:
        # Get episode range from first epoch
        first_epoch_result = all_entropy_results[epochs[0]]
        n_episodes = len(first_epoch_result['entropy_mean'])
        episode_range = np.arange(n_episodes)
        
        # Color map for epochs
        epoch_colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
        
        # Baseline colors
        baseline_color_map = {
            'emp': 'tab:blue',
            'ucb': 'tab:orange',
            'linucb': 'tab:orange',
            'thmp': 'black',
        }
        
        # FIGURE 1: Entropy over episodes
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot lnr for each epoch first (so baselines appear on top)
        for i, epoch_num in enumerate(epochs):
            entropy_mean = all_entropy_results[epoch_num]['entropy_mean']
            entropy_sem = all_entropy_results[epoch_num]['entropy_sem']
            
            # Only add label for first and last epoch to avoid clutter
            if i == 0:
                label = f'lnr Epoch {epochs[0]}'
            elif i == len(epochs) - 1:
                label = f'lnr Epoch {epochs[-1]}'
            else:
                label = None
            
            ax1.plot(episode_range, entropy_mean, label=label,
                    color=epoch_colors[i], linewidth=2, alpha=0.7)
            ax1.fill_between(episode_range, entropy_mean - entropy_sem, entropy_mean + entropy_sem,
                            alpha=0.15, color=epoch_colors[i])
        
        # Plot baseline curves on top (if available)
        if baseline_entropy_results is not None:
            for baseline_name, baseline_data in baseline_entropy_results.items():
                entropy_mean = baseline_data['entropy_mean']
                entropy_sem = baseline_data['entropy_sem']
                
                color = baseline_color_map.get(baseline_name, 'gray')
                ax1.plot(episode_range, entropy_mean, label=baseline_name, linestyle='--',
                        linewidth=2, color=color)
                ax1.fill_between(episode_range, entropy_mean - entropy_sem, entropy_mean + entropy_sem,
                               alpha=0.2, color=color)
        
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Entropy (nats)')
        ax1.set_title(f'Policy Entropy During Episodes')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{mass_eval_dir}/entropy/mass_eval_entropy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {mass_eval_dir}/entropy/mass_eval_entropy.png")
        
        # FIGURE 2: Cross-Entropy over episodes
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        # Plot lnr for each epoch first (so baselines appear on top)
        for i, epoch_num in enumerate(epochs):
            cross_entropy_mean = all_entropy_results[epoch_num]['cross_entropy_mean']
            cross_entropy_sem = all_entropy_results[epoch_num]['cross_entropy_sem']
            
            # Only add label for first and last epoch to avoid clutter
            if i == 0:
                label = f'lnr Epoch {epochs[0]}'
            elif i == len(epochs) - 1:
                label = f'lnr Epoch {epochs[-1]}'
            else:
                label = None
            
            ax2.plot(episode_range, cross_entropy_mean, label=label,
                    color=epoch_colors[i], linewidth=2, alpha=0.7)
            ax2.fill_between(episode_range, cross_entropy_mean - cross_entropy_sem, 
                            cross_entropy_mean + cross_entropy_sem,
                            alpha=0.15, color=epoch_colors[i])
        
        # Plot baseline curves on top (if available)
        if baseline_entropy_results is not None:
            for baseline_name, baseline_data in baseline_entropy_results.items():
                cross_entropy_mean = baseline_data['cross_entropy_mean']
                cross_entropy_sem = baseline_data['cross_entropy_sem']
                
                color = baseline_color_map.get(baseline_name, 'gray')
                ax2.plot(episode_range, cross_entropy_mean, label=baseline_name, linestyle='--',
                        linewidth=2, color=color)
                ax2.fill_between(episode_range, cross_entropy_mean - cross_entropy_sem, 
                               cross_entropy_mean + cross_entropy_sem,
                               alpha=0.2, color=color)
        
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Cross-Entropy (nats)')
        ax2.set_title(f'Cross-Entropy During Episodes')
        ax2.legend(fontsize=9, loc='best')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{mass_eval_dir}/entropy/mass_eval_crossentropy.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {mass_eval_dir}/entropy/mass_eval_crossentropy.png")

    print(f"\nMass evaluation completed!")
    print(f"Results saved to: {mass_eval_dir}")
    if len(all_bar_results) > 0:
        print(f"  - bar/mass_eval_bar.png: Performance comparison across epochs")
    if len(all_online_results) > 0:
        if baseline_online_results is not None:
            # Bandit case
            print(f"  - online/mass_eval_online_suboptimality.png: Suboptimality over episodes")
            print(f"  - online/mass_eval_online_cumregret.png: Cumulative regret over episodes")
        else:
            # Darkroom case
            print(f"  - online/mass_eval_online_returns.png: Returns over episodes")
    if len(all_graph_results) > 0:
        print(f"  - graph/mass_eval_graph.png: Lnr performance vs dataset size across epochs")
    if len(all_entropy_results) > 0:
        print(f"  - entropy/mass_eval_entropy.png: Policy entropy during episodes across epochs")
        print(f"  - entropy/mass_eval_crossentropy.png: Cross-entropy during episodes across epochs")
    # copy config.yaml, logs.txt, and train_loss.png to mass_eval_dir
