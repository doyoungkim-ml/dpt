import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

import argparse
import os
import time
from IPython import embed

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

import numpy as np
import common_args
import random
from net import Transformer, ImageTransformer
from utils import (
    build_bandit_model_filename,
    build_linear_bandit_model_filename,
    build_darkroom_model_filename,
    build_miniworld_model_filename,
)
from envs import darkroom_env, bandit_env
from collect_data import (
    rollin_bandit,
    rollin_linear_bandit_vec,
    rollin_mdp,
    rollin_mdp_miniworld,
    rand_pos_and_dir
)
from skimage.transform import resize
import gym

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OnlineEnvironmentManager:
    """Manages growing contexts for online environments with model-based sampling."""

    def __init__(self, envs, env_type, horizon, config, model=None, **kwargs):
        self.envs = envs
        self.env_type = env_type
        self.horizon = horizon
        self.config = config
        self.model = model
        self.kwargs = kwargs
        self.reset_contexts()

    def reset_contexts(self):
        """Reset all environment contexts at the start of each epoch."""
        self.contexts = {i: {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': []
        } for i in range(len(self.envs))}

    def step_and_learn(self, optimizer, loss_fn, action_dim, horizon, debug=False):
        """Step environments and learn from optimal actions using batched processing."""
        debug_info = {'env_steps': []} if debug else None
        
        # PHASE 1: Collect data from all environments
        env_data = []
        for env_idx in range(len(self.envs)):
            env = self.envs[env_idx]

            # Create current state
            if self.env_type in ['bandit', 'linear_bandit']:
                current_state = np.array([1])
            elif self.env_type.startswith('darkroom'):
                current_state = env.sample_state()
            elif self.env_type == 'miniworld':
                # For miniworld, place agent at random position
                init_pos, init_dir = rand_pos_and_dir(env)
                env.place_agent(pos=init_pos, dir=init_dir)
                current_state = env.agent.dir_vec[[0, -1]]

            # DEBUG: Current context info
            context_size = len(self.contexts[env_idx]['states']) if debug else 0
            recent_rewards = self.contexts[env_idx]['rewards'][-3:] if debug and self.contexts[env_idx]['rewards'] else []
                
            # 1. Get model's action prediction BEFORE adding to context
            if self.model is not None:
                model_action = self._get_model_action(env_idx, current_state, debug=debug)
            else:
                # Fallback to random action for early training
                model_action = self._get_random_action(env)

            # Execute action in environment
            if self.env_type in ['bandit', 'linear_bandit']:
                next_state, reward = env.transit(current_state, model_action)
            elif self.env_type.startswith('darkroom'):
                next_state, reward = env.transit(current_state, model_action)
            elif self.env_type == 'miniworld':
                # Convert one-hot to action index
                action_idx = np.argmax(model_action)
                _, reward, _, _, _ = env.step(action_idx)
                next_state = env.agent.dir_vec[[0, -1]]

            # 2. Add transition to context BEFORE learning
            self.contexts[env_idx]['states'].append(current_state)
            self.contexts[env_idx]['actions'].append(model_action)
            self.contexts[env_idx]['next_states'].append(next_state)
            self.contexts[env_idx]['rewards'].append(reward)

            # Truncate context if too long
            if len(self.contexts[env_idx]['states']) > self.horizon:
                for key in self.contexts[env_idx]:
                    self.contexts[env_idx][key] = self.contexts[env_idx][key][-self.horizon:]

            # Store data for batched processing
            env_data.append({
                'env_idx': env_idx,
                'env': env,
                'current_state': current_state,
                'model_action': model_action,
                'reward': reward,
                'next_state': next_state,
                'context_size': context_size,
                'recent_rewards': recent_rewards
            })

        # PHASE 2: Create batch of training samples
        training_samples = []
        for data in env_data:
            training_sample = self._create_training_sample(data['env'], data['env_idx'], data['current_state'])
            training_samples.append(training_sample)

        # PHASE 3: Batched forward and backward pass
        if self.env_type == 'miniworld':
            batch = batch_to_tensors(training_samples, self.config, self.env_type, self.kwargs.get('transform'))
        else:
            batch = batch_to_tensors(training_samples, self.config, self.env_type)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass for entire batch
        true_actions = batch['optimal_actions']  # Shape: [batch_size, action_dim]
        pred_actions = self.model(batch)         # Shape: [batch_size, horizon, action_dim]
        
        # Reshape for loss computation
        batch_size = true_actions.shape[0]
        true_actions = true_actions.unsqueeze(1).repeat(1, pred_actions.shape[1], 1)  # [batch_size, horizon, action_dim]
        true_actions = true_actions.reshape(-1, action_dim)  # [batch_size * horizon, action_dim]
        pred_actions = pred_actions.reshape(-1, action_dim)  # [batch_size * horizon, action_dim]

        # Single backward pass for entire batch
        optimizer.zero_grad()
        loss = loss_fn(pred_actions, true_actions)
        loss.backward()
        optimizer.step()

        # Compute average loss per sample
        total_loss = loss.item() / (batch_size * horizon)

        # PHASE 4: Debug output (show first 3 environments)
        if debug and len(env_data) > 0:
            # Show debug info for first 3 environments
            num_debug_envs = min(1, len(env_data))
            
            for debug_idx in range(num_debug_envs):
                data = env_data[debug_idx]
                env_idx = data['env_idx']
                
                # Extract predictions for this environment
                pred_probs = torch.softmax(pred_actions.view(batch_size, horizon, action_dim)[debug_idx, -1], dim=-1)
                optimal_action = training_samples[debug_idx]['optimal_action']
                optimal_action_idx = np.argmax(optimal_action)
                sampled_action_idx = np.argmax(data['model_action'])
                sampling_match = sampled_action_idx == optimal_action_idx
                
                # Format probability distribution for display
                if self.env_type in ['bandit', 'linear_bandit']:
                    prob_str = ", ".join([f"A{i}:{pred_probs[i].item():.3f}" for i in range(len(pred_probs))])
                else:
                    prob_str = ", ".join([f"A{i}:{pred_probs[i].item():.3f}" for i in range(len(pred_probs))])
                
                print(f"Batch: Env {env_idx}: Context={data['context_size']}, Probs=[{prob_str}], SampledAction={sampled_action_idx}, OptimalAction={optimal_action_idx}, Match={sampling_match}, Reward={data['reward']:.3f}, Loss={total_loss:.4f}, BatchSize={batch_size}")
                
                step_debug = {
                    'env_idx': env_idx,
                    'current_state': data['current_state'].tolist(),
                    'context_size': data['context_size'],
                    'recent_rewards': data['recent_rewards'],
                    'pred_probs': pred_probs.tolist(),
                    'sampled_action_idx': sampled_action_idx,
                    'optimal_action_idx': optimal_action_idx,
                    'sampling_match': sampling_match,
                    'reward': data['reward'],
                    'step_loss': total_loss,
                    'next_state': data['next_state'].tolist(),
                    'batch_size': batch_size
                }
                debug_info['env_steps'].append(step_debug)

        if debug:
            return total_loss, debug_info
        return total_loss

    def _get_model_action(self, env_idx, query_state, debug=False):
        """Get action from model given current context and query state."""
        if self.model is None:
            return self._get_random_action(self.envs[env_idx])

        with torch.no_grad():
            # Prepare batch with current context (add dummy optimal action for tensor conversion)
            sample = self._create_model_input_sample(env_idx, query_state)

            # Add dummy optimal action to make it compatible with batch_to_tensors
            if self.env_type in ['bandit', 'linear_bandit']:
                sample['optimal_action'] = np.zeros(self.envs[env_idx].dim)  # dummy
            elif self.env_type.startswith('darkroom'):
                sample['optimal_action'] = np.zeros(5)  # dummy
            elif self.env_type == 'miniworld':
                sample['optimal_action'] = np.zeros(self.envs[env_idx].action_space.n)  # dummy

            # Convert to tensor
            if self.env_type == 'miniworld':
                batch = batch_to_tensors([sample], self.config, self.env_type, self.kwargs.get('transform'))
            else:
                batch = batch_to_tensors([sample], self.config, self.env_type)

            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get model prediction
            self.model.eval()
            pred_actions = self.model(batch)

            # Sample action (you can use softmax sampling or argmax)
            if self.env_type == 'miniworld':
                # For discrete actions, use softmax sampling
                probs = torch.softmax(pred_actions[0, -1], dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
                action = np.zeros(self.envs[env_idx].action_space.n)
                action[action_idx] = 1.0
            else:
                # For continuous/bandit actions, use softmax sampling over action space
                probs = torch.softmax(pred_actions[0, -1], dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
                action = np.zeros(len(probs))
                action[action_idx] = 1.0

            self.model.train()
            return action

    def _get_random_action(self, env):
        """Get random action as fallback."""
        if self.env_type in ['bandit', 'linear_bandit']:
            action = np.zeros(env.dim)
            action[np.random.randint(env.dim)] = 1.0
        elif self.env_type.startswith('darkroom'):
            action = env.sample_action()
        elif self.env_type == 'miniworld':
            action = np.zeros(env.action_space.n)
            action[np.random.randint(env.action_space.n)] = 1.0
        return action

    def _create_model_input_sample(self, env_idx, query_state):
        """Create sample for model input (without optimal action)."""
        context = self.contexts[env_idx]

        # Pad context if needed
        context_len = len(context['states'])
        if context_len < self.horizon:
            pad_len = self.horizon - context_len
            if self.env_type == 'miniworld':
                dummy_image = np.zeros(self.kwargs.get('target_shape', (25, 25, 3)))
                dummy_state = np.zeros(2)
                dummy_action = np.zeros(self.envs[env_idx].action_space.n)

                padded_images = [dummy_image] * pad_len + context.get('images', [dummy_image] * context_len)
                padded_states = [dummy_state] * pad_len + context['states']
                padded_actions = [dummy_action] * pad_len + context['actions']
                padded_next_states = [dummy_state] * pad_len + context['next_states']
                padded_rewards = [0.0] * pad_len + context['rewards']

                return {
                    'query_image': env.render_obs(),
                    'query_state': query_state,
                    'context_images': np.array(padded_images),
                    'context_states': np.array(padded_states),
                    'context_actions': np.array(padded_actions),
                    'context_next_states': np.array(padded_next_states),
                    'context_rewards': np.array(padded_rewards),
                }
            else:
                if self.env_type in ['bandit', 'linear_bandit']:
                    dummy_state = np.array([1])
                    dummy_action = np.zeros(self.envs[env_idx].dim)
                else:  # darkroom
                    dummy_state = np.zeros(2)
                    dummy_action = np.zeros(5)

                padded_states = [dummy_state] * pad_len + context['states']
                padded_actions = [dummy_action] * pad_len + context['actions']
                padded_next_states = [dummy_state] * pad_len + context['next_states']
                padded_rewards = [0.0] * pad_len + context['rewards']
        else:
            padded_states = context['states']
            padded_actions = context['actions']
            padded_next_states = context['next_states']
            padded_rewards = context['rewards']

        return {
            'query_state': query_state,
            'context_states': np.array(padded_states),
            'context_actions': np.array(padded_actions),
            'context_next_states': np.array(padded_next_states),
            'context_rewards': np.array(padded_rewards),
        }

    def _create_training_sample(self, env, env_idx, query_state):
        """Create training sample with optimal action as target."""
        sample = self._create_model_input_sample(env_idx, query_state)

        # Add optimal action as target
        if self.env_type in ['bandit', 'linear_bandit']:
            sample['optimal_action'] = env.opt_a
            sample['means'] = env.means
            if self.env_type == 'linear_bandit':
                sample['arms'] = self.kwargs.get('arms')
                sample['theta'] = env.theta
                sample['var'] = env.var
        elif self.env_type.startswith('darkroom'):
            sample['optimal_action'] = env.opt_action(query_state)
            sample['goal'] = env.goal
            if hasattr(env, 'perm_index'):
                sample['perm_index'] = env.perm_index
        elif self.env_type == 'miniworld':
            obs = sample['query_image'] if 'query_image' in sample else env.render_obs()
            action = env.opt_a(obs, env.agent.pos, env.agent.dir_vec)
            one_hot_action = np.zeros(env.action_space.n)
            one_hot_action[action] = 1
            sample['optimal_action'] = one_hot_action
            sample['env_id'] = getattr(env, 'env_id', 0)

        return sample




def batch_to_tensors(batch_data, config, env_type, transform=None):
    """Convert batch data to tensors suitable for model input."""
    if env_type == 'miniworld':
        return batch_to_tensors_miniworld(batch_data, config, transform)
    else:
        return batch_to_tensors_standard(batch_data, config)


def batch_to_tensors_standard(batch_data, config):
    """Convert standard (non-image) batch data to tensors."""
    horizon = config['horizon']
    state_dim = config['state_dim']
    action_dim = config['action_dim']

    batch_size = len(batch_data)

    # Initialize tensors
    query_states = torch.zeros(batch_size, state_dim)
    optimal_actions = torch.zeros(batch_size, action_dim)
    context_states = torch.zeros(batch_size, horizon, state_dim)
    context_actions = torch.zeros(batch_size, horizon, action_dim)
    context_next_states = torch.zeros(batch_size, horizon, state_dim)
    context_rewards = torch.zeros(batch_size, horizon, 1)

    # Create zeros tensor like in Dataset class
    zeros = torch.zeros(state_dim ** 2 + action_dim + 1)

    for i, sample in enumerate(batch_data):
        query_states[i] = torch.from_numpy(sample['query_state']).float()
        optimal_actions[i] = torch.from_numpy(sample['optimal_action']).float()
        context_states[i] = torch.from_numpy(sample['context_states']).float()
        context_actions[i] = torch.from_numpy(sample['context_actions']).float()
        context_next_states[i] = torch.from_numpy(sample['context_next_states']).float()
        context_rewards[i] = torch.from_numpy(sample['context_rewards']).float().unsqueeze(-1)

    return {
        'query_states': query_states,
        'optimal_actions': optimal_actions,
        'context_states': context_states,
        'context_actions': context_actions,
        'context_next_states': context_next_states,
        'context_rewards': context_rewards,
        'zeros': zeros.unsqueeze(0).repeat(batch_size, 1),
    }


def batch_to_tensors_miniworld(batch_data, config, transform):
    """Convert miniworld batch data to tensors."""
    horizon = config['horizon']
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    image_size = config['image_size']

    batch_size = len(batch_data)

    # Initialize tensors
    query_images = torch.zeros(batch_size, 3, image_size, image_size)
    query_states = torch.zeros(batch_size, state_dim)
    optimal_actions = torch.zeros(batch_size, action_dim)
    context_images = torch.zeros(batch_size, horizon, 3, image_size, image_size)
    context_states = torch.zeros(batch_size, horizon, state_dim)
    context_actions = torch.zeros(batch_size, horizon, action_dim)
    context_rewards = torch.zeros(batch_size, horizon, 1)

    # Create zeros tensor like in Dataset class
    zeros = torch.zeros(state_dim ** 2 + action_dim + 1)

    for i, sample in enumerate(batch_data):
        # Query data
        query_img = transform(sample['query_image'])
        query_images[i] = query_img
        query_states[i] = torch.from_numpy(sample['query_state']).float()
        optimal_actions[i] = torch.from_numpy(sample['optimal_action']).float()

        # Context data
        for h in range(horizon):
            context_img = transform(sample['context_images'][h])
            context_images[i, h] = context_img

        context_states[i] = torch.from_numpy(sample['context_states']).float()
        context_actions[i] = torch.from_numpy(sample['context_actions']).float()
        context_rewards[i] = torch.from_numpy(sample['context_rewards']).float().unsqueeze(-1)

    return {
        'query_images': query_images,
        'query_states': query_states,
        'optimal_actions': optimal_actions,
        'context_images': context_images,
        'context_states': context_states,
        'context_actions': context_actions,
        'context_rewards': context_rewards,
        'zeros': zeros.unsqueeze(0).repeat(batch_size, 1),
    }


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples_per_iter', type=int, default=64, help='Number of samples to generate per iteration')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with detailed step-by-step logging')

    args = vars(parser.parse_args())
    print("Args: ", args)

    env = args['env']
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['H'] * args['envs']
    horizon = args['H']
    dim = args['dim']
    state_dim = dim
    action_dim = dim
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    num_epochs = args['num_epochs']
    seed = args['seed']
    lin_d = args['lin_d']
    samples_per_iter = args['samples_per_iter']
    debug_mode = args['debug']

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

    # Create model config like in train.py
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

    # Setup environments for online sampling
    if env == 'bandit':
        state_dim = 1
        train_envs = [bandit_env.sample(dim, horizon, var) for _ in range(n_envs)]

        model_config.update({'var': var, 'cov': cov})
        filename = build_bandit_model_filename(env, model_config)

    elif env == 'linear_bandit':
        state_dim = 1
        # Generate fixed features for arms
        rng = np.random.RandomState(seed=1234)
        arms = rng.normal(size=(dim, lin_d)) / np.sqrt(lin_d)
        train_envs = [bandit_env.sample_linear(arms, horizon, var) for _ in range(n_envs)]

        model_config.update({'lin_d': lin_d, 'var': var, 'cov': cov})
        filename = build_linear_bandit_model_filename(env, model_config)

    elif env.startswith('darkroom'):
        state_dim = 2
        action_dim = 5
        # Generate goals for training environments
        goals = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
        np.random.RandomState(seed=0).shuffle(goals)
        train_test_split = int(.8 * len(goals))
        train_goals = goals[:train_test_split]
        train_goals = np.repeat(train_goals, max(1, n_envs // len(train_goals)), axis=0)[:n_envs]

        if env == 'darkroom_heldout':
            train_envs = [darkroom_env.DarkroomEnv(dim, goal, horizon) for goal in train_goals]
        else:
            train_envs = [darkroom_env.DarkroomEnvPermuted(dim, i, horizon) for i in range(n_envs)]

        filename = build_darkroom_model_filename(env, model_config)

    elif env == 'miniworld':
        import gymnasium as gym
        import miniworld

        state_dim = 2
        action_dim = 4
        target_shape = (25, 25, 3)
        gym_env = gym.make('MiniWorld-OneRoomS6FastMultiFourBoxesFixedInit-v0')
        train_env_ids = list(range(n_envs))

        filename = build_miniworld_model_filename(env, model_config)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    log_filename = f'figs/loss/{filename}_online_logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        print(string)
        with open(log_filename, 'a') as f:
            print(string, file=f)

    train_loss = []
    total_iterations = 0

    # Create online environment manager
    if env == 'miniworld':
        env_manager = OnlineEnvironmentManager(
            [gym_env] * len(train_env_ids), env, horizon, config, model,
            transform=transform, target_shape=target_shape, env_ids=train_env_ids
        )
    elif env == 'linear_bandit':
        env_manager = OnlineEnvironmentManager(
            train_envs, env, horizon, config, model, arms=arms
        )
    else:
        env_manager = OnlineEnvironmentManager(
            train_envs, env, horizon, config, model
        )

    printw(f"Starting online training for {env}")
    printw(f"Model filename: {filename}")

    for epoch in range(num_epochs):
        printw(f"Epoch: {epoch + 1}")
        start_time = time.time()

        # Reset environment contexts at start of each epoch
        env_manager.reset_contexts()
        env_manager.model = model  # Update model reference

        epoch_train_loss = 0.0
        epoch_iterations = 0


        for iteration in range(horizon):
            # Step all environments and do immediate gradient updates
            # Enable debug for first few iterations if debug mode is on
            should_debug = debug_mode #and (epoch == 0 and iteration < 10)
            
            if should_debug:
                iteration_loss, debug_info = env_manager.step_and_learn(optimizer, loss_fn, action_dim, horizon, debug=True)
                printw(f"\n=== DEBUG: Epoch {epoch+1}, Iteration {iteration} ===")
            else:
                iteration_loss = env_manager.step_and_learn(optimizer, loss_fn, action_dim, horizon)

            epoch_train_loss += iteration_loss
            epoch_iterations += 1
            total_iterations += 1

            if iteration % 10 == 0:
                print(f"Epoch {epoch+1}, Iteration {iteration}/{horizon}, Loss: {iteration_loss:.6f}", end='\r')

        train_loss.append(epoch_train_loss / epoch_iterations)
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]:.6f}")
        printw(f"\tTrain time: {end_time - start_time:.2f}s")
        printw(f"\tTotal iterations: {total_iterations}")

        # Checkpointing
        if (epoch + 1) % 50 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), f'models/{filename}_online_epoch{epoch+1}.pt')

        # Plotting
        if (epoch + 1) % 10 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Train Loss: {train_loss[-1]:.6f}")
            printw("\n")

            plt.yscale('log')
            plt.plot(train_loss[1:], label="Train Loss")
            plt.legend()
            plt.title(f"Online Training Loss - {env}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.savefig(f"figs/loss/{filename}_online_train_loss.png")
            plt.clf()

    torch.save(model.state_dict(), f'models/{filename}_online.pt')
    printw("Online training completed!")