import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)  # or 'forkserver'

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
from IPython import embed
from pathlib import Path

import wandb
import torch
from torchvision.transforms import transforms

import numpy as np
import common_args
import random
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

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


from training_eval import log_evaluation_plots_to_wandb
EVAL_ENABLED = True


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

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
        name=f"offline_{config_basename}",
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
    lin_d = args.get('lin_d', 2)  # Only needed for linear_bandit
    
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

    dataset_config = {
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
    }
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

        dataset_config.update({'var': var, 'cov': cov, 'type': 'uniform'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'bandit_thompson':
        state_dim = 1

        dataset_config.update({'var': var, 'cov': cov, 'type': 'bernoulli'})
        path_train = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'linear_bandit':
        state_dim = 1

        dataset_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        path_train = build_linear_bandit_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_linear_bandit_data_filename(
            env, n_envs, dataset_config, mode=1)

        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(env, model_config)

    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5

        dataset_config.update({'rollin_type': 'uniform'})
        path_train = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=0)
        path_test = build_darkroom_data_filename(
            env, n_envs, dataset_config, mode=1)

        filename = build_darkroom_model_filename(env, model_config)

    elif env == 'miniworld':
        state_dim = 2   # direction vector is 2D, no position included
        action_dim = 4

        dataset_config.update({'rollin_type': 'uniform'})

        increment = 5000
        starts = np.arange(0, n_envs, increment)
        starts = np.array(starts)
        ends = starts + increment - 1

        paths_train = []
        paths_test = []
        for start_env_id, end_env_id in zip(starts, ends):
            path_train = build_miniworld_data_filename(
                env, start_env_id, end_env_id, dataset_config, mode=0)
            path_test = build_miniworld_data_filename(
                env, start_env_id, end_env_id, dataset_config, mode=1)

            paths_train.append(path_train)
            paths_test.append(path_test)

        filename = build_miniworld_model_filename(env, model_config)
        print(f"Generate filename: {filename}")

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

    # Save/Update config file in experiment directory (overwrite to reflect resume info)
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
        """
        A drop-in replacement for print that also writes to a log file.
        """
        # Use the standard print function to print to the console
        print(string)

        # Write the same output to the log file
        with open(log_filename, 'a') as f:
            print(string, file=f)




    if env == 'miniworld':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])



        params.update({'num_workers': 16,
                'prefetch_factor': 2,
                'persistent_workers': True,
                'pin_memory': True,
                'batch_size': 64,
                'worker_init_fn': worker_init_fn,
            })


        printw("Loading miniworld data...")
        train_dataset = ImageDataset(paths_train, config, transform)
        test_dataset = ImageDataset(paths_test, config, transform)
        printw("Done loading miniworld data")
    else:
        train_dataset = Dataset(path_train, config)
        test_dataset = Dataset(path_test, config)

    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    test_loader = torch.utils.data.DataLoader(test_dataset, **params)

    # Resume model weights if checkpoint_dir provided; also compute epoch offset
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

    printw("Num train batches: " + str(len(train_loader)))
    printw("Num test batches: " + str(len(test_loader)))

    # Limit to remaining epochs if resuming
    remaining_epochs = max(0, num_epochs - resume_epoch_offset)
    # Update config.yaml with resume metadata
    try:
        with open(config_path, 'w') as f:
            merged = dict(args)
            merged.update({
                'resume': checkpoint_dir is not None,
                'resumed_from_epoch': int(resume_epoch_offset),
                'remaining_epochs': int(remaining_epochs),
            })
            yaml.dump(merged, f, default_flow_style=False)
    except Exception:
        pass

    for epoch in range(remaining_epochs):
        # EVALUATION
        printw(f"Epoch: {resume_epoch_offset + epoch + 1}")
        start_time = time.time()
        with torch.no_grad():
            epoch_test_loss = 0.0
            epoch_test_entropy = 0.0
            for i, batch in enumerate(test_loader):
                print(f"Batch {i} of {len(test_loader)}", end='\r')
                batch = {k: v.to(device) for k, v in batch.items()}
                true_actions = batch['optimal_actions']
                pred_actions, _ = model(batch)  # Unpack tuple to get predictions
                true_actions = true_actions.unsqueeze(
                    1).repeat(1, pred_actions.shape[1], 1)
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
        printw(f"\tEval time: {end_time - start_time}")
        


        # TRAINING
        epoch_train_loss = 0.0
        epoch_train_entropy = 0.0
        start_time = time.time()

        for i, batch in enumerate(train_loader):
            print(f"Batch {i} of {len(train_loader)}", end='\r')
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch['optimal_actions']
            pred_actions, _ = model(batch)  # Unpack tuple to get predictions
            true_actions = true_actions.unsqueeze(
                1).repeat(1, pred_actions.shape[1], 1)
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
        printw(f"\tTrain time: {end_time - start_time}")
        
        # Log training metrics to wandb
        wandb.log({
            "epoch": resume_epoch_offset + epoch + 1,
            "train_ce_loss": train_loss[-1],  # For offline, CE loss equals total loss
            "train_entropy": train_entropy_val,
            "train_alpha": 1.0,  # Offline: alpha = 1
            "train_beta": 0.0,   # Offline: beta = 0
            "total_loss": train_loss[-1],
            "test_loss": test_loss[-1],
            "test_ce_loss": test_loss[-1],  # For offline, CE loss equals total loss
            "test_entropy": test_entropy_val,
            "eval_time": end_time - start_time,
        })


        # LOGGING
        current_epoch = resume_epoch_offset + epoch + 1
        if (epoch + 1) % args['eval_every'] == 0:
            torch.save(model.state_dict(),
                       f'{experiment_dir}/epoch{current_epoch}.pt')
            
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

        # LOGGING TO WANDB
        if (epoch + 1) % 10 == 0:
            printw(f"Epoch: {current_epoch}")
            printw(f"Test Loss:        {test_loss[-1]}")
            printw(f"Train Loss:       {train_loss[-1]}")
            printw("\n")

            # Persist metrics for future resumes
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
    print("Done.")
