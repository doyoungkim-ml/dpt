import argparse
import os
import pickle
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
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

def interpolate_positional_embeddings(model, new_horizon):
    """
    Interpolate positional embeddings to support longer sequences.
    
    Args:
        model: The transformer model
        new_horizon: The new horizon length to support
    """
    # Calculate new required positions
    new_n_positions = 4 * (1 + new_horizon)
    current_n_positions = model.transformer.wpe.weight.shape[0]
    
    if new_n_positions <= current_n_positions:
        # No interpolation needed
        return model
    
    print(f"Interpolating positional embeddings from {current_n_positions} to {new_n_positions} positions")
    
    # Get current positional embeddings
    current_pos_emb = model.transformer.wpe.weight.data  # [current_n_positions, n_embd]
    
    # Create new positional embedding tensor
    new_pos_emb = torch.zeros(new_n_positions, current_pos_emb.shape[1], 
                             dtype=current_pos_emb.dtype, device=current_pos_emb.device)
    
    # Use linear interpolation
    old_indices = torch.linspace(0, current_n_positions - 1, current_n_positions)
    new_indices = torch.linspace(0, current_n_positions - 1, new_n_positions)
    
    for i in range(current_pos_emb.shape[1]):  # For each embedding dimension
        # Interpolate this dimension
        new_pos_emb[:, i] = torch.from_numpy(
            np.interp(new_indices.numpy(), old_indices.numpy(), current_pos_emb[:, i].cpu().numpy())
        ).to(current_pos_emb.device)
    
    # Replace the positional embedding layer
    old_wpe = model.transformer.wpe
    model.transformer.wpe = torch.nn.Embedding(new_n_positions, old_wpe.embedding_dim)
    model.transformer.wpe.weight.data = new_pos_emb
    
    # Update the model's config
    model.horizon = new_horizon
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--extend_horizon', type=int, default=None, help='Extend to this horizon (if different from config)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override horizon if specified
    if args.extend_horizon is not None:
        config['hor'] = args.extend_horizon
        print(f"Extending horizon to {args.extend_horizon}")
    
    # Extract parameters
    envname = config['env']
    dim = config['dim']
    var = config['var']
    cov = config['cov']
    test_cov = config.get('test_cov', cov)
    n_embd = config['embd']
    n_layer = config['layer']
    n_head = config['head']
    dropout = config['dropout']
    horizon = config['hor']
    n_eval = config.get('n_eval', 100)
    seed = config.get('seed', 0)
    shuffle = config.get('shuffle', True)
    
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if test_cov < 0:
        test_cov = cov
    
    # Model configuration - use ORIGINAL training horizon for model creation
    original_horizon = 100  # The horizon the model was trained with
    model_config = {
        'shuffle': shuffle,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'horizon': original_horizon,  # Use original horizon for model creation
        'dim': dim,
    }
    
    if envname == 'bandit':
        state_dim = 1
        action_dim = 5
        model_config.update({'var': var, 'cov': cov})
    else:
        raise NotImplementedError(f"Environment {envname} not supported in this script")
    
    config_model = {
        'horizon': original_horizon,  # Use original horizon
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_envs': 1,
        'n_hists': 1,
        'n_samples': 1,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,  # Set to True so model returns single timestep output
    }
    
    # Create model with original horizon
    model = Transformer(config_model)
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Extend model for new horizon if needed
    if horizon != original_horizon:
        model = interpolate_positional_embeddings(model, horizon)
    
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Run evaluation
    dataset_config = {
        'horizon': horizon,
        'dim': dim,
        'var': var, 
        'cov': cov, 
        'type': 'uniform'
    }
    
    eval_filepath = build_bandit_data_filename(envname, n_eval, dataset_config, mode=2)
    
    # Build filename for saving results
    model_filename = os.path.basename(args.model_path).replace('.pt', '')
    save_filename = f'{model_filename}_extended_hor{horizon}.pkl'
    
    evals_filename = f'evals_{args.model_path.replace("/", "_").replace(".pt", "")}'
    os.makedirs(f'figs/{evals_filename}/online', exist_ok=True)
    os.makedirs(f'figs/{evals_filename}/graph', exist_ok=True)
    
    eval_trajs = pickle.load(open(eval_filepath, 'rb'))
    
    config_eval = {
        'n_eval': n_eval,
        'horizon': horizon,
        'var': var,
        'bandit_type': 'uniform',
    }
    
    print(f"Running evaluation with extended horizon {horizon}")
    eval_bandit.online(eval_trajs, model, **config_eval)
    
    plt.savefig(f'figs/{evals_filename}/online/{save_filename}.png')
    plt.clf()
    plt.cla()
    plt.close()
    
    print(f"Evaluation completed! Results saved to figs/{evals_filename}/online/{save_filename}.png")

if __name__ == '__main__':
    main()
