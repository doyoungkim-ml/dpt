import pickle

import numpy as np
import torch

from utils import convert_to_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class Dataset(torch.utils.data.Dataset):
    """Dataset class."""

    def __init__(self, path, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config

        # if path is not a list
        if not isinstance(path, list):
            path = [path]

        self.trajs = []
        for p in path:
            with open(p, 'rb') as f:
                self.trajs += pickle.load(f)
            
        context_states = []
        context_actions = []
        context_next_states = []
        context_rewards = []
        query_states = []
        optimal_actions = []
        optimal_actions_per_state = []  # NEW: optimal actions for each context state (for MDPs)

        for traj in self.trajs:
            context_states.append(traj['context_states'])
            context_actions.append(traj['context_actions'])
            # context_next_states is optional (not used by transformer model)
            if 'context_next_states' in traj:
                context_next_states.append(traj['context_next_states'])
            context_rewards.append(traj['context_rewards'])

            query_states.append(traj['query_state'])
            
            # Check if optimal_actions (plural) exists - this is for MDPs where each state has its own optimal action
            if 'optimal_actions' in traj:
                optimal_actions_per_state.append(traj['optimal_actions'])
                # Also keep single optimal_action for backward compatibility
                optimal_actions.append(traj.get('optimal_action', traj['optimal_actions'][0]))
            else:
                # Fall back to single optimal_action (for bandit environments)
                optimal_actions.append(traj['optimal_action'])
                optimal_actions_per_state.append(None)

        context_states = np.array(context_states)
        context_actions = np.array(context_actions)
        if len(context_next_states) > 0:
            context_next_states = np.array(context_next_states)
        else:
            context_next_states = None
        context_rewards = np.array(context_rewards)
        if len(context_rewards.shape) < 3:
            context_rewards = context_rewards[:, :, None]
        query_states = np.array(query_states)
        optimal_actions = np.array(optimal_actions)
        
        # Convert optimal_actions_per_state to array if at least one trajectory has it
        # If all trajectories have it, convert to array; otherwise keep as None
        has_optimal_actions_per_state = any(x is not None for x in optimal_actions_per_state)
        if has_optimal_actions_per_state:
            # Check if all trajectories have it (for consistent batching)
            all_have_it = all(x is not None for x in optimal_actions_per_state)
            if all_have_it:
                optimal_actions_per_state = np.array(optimal_actions_per_state)
            else:
                # Mixed case: some have it, some don't - set to None to avoid issues
                # In practice, this shouldn't happen if we're consistent about MDP vs bandit
                optimal_actions_per_state = None
        else:
            optimal_actions_per_state = None

        self.dataset = {
            'query_states': convert_to_tensor(query_states, store_gpu=self.store_gpu),
            'optimal_actions': convert_to_tensor(optimal_actions, store_gpu=self.store_gpu),
            'context_states': convert_to_tensor(context_states, store_gpu=self.store_gpu),
            'context_actions': convert_to_tensor(context_actions, store_gpu=self.store_gpu),
            'context_rewards': convert_to_tensor(context_rewards, store_gpu=self.store_gpu),
        }
        
        if context_next_states is not None:
            self.dataset['context_next_states'] = convert_to_tensor(context_next_states, store_gpu=self.store_gpu)
        else:
            self.dataset['context_next_states'] = None
            
        # Store optimal_actions_per_state if available (for MDPs)
        if optimal_actions_per_state is not None:
            self.dataset['optimal_actions_per_state'] = convert_to_tensor(optimal_actions_per_state, store_gpu=self.store_gpu)
        else:
            self.dataset['optimal_actions_per_state'] = None

        self.zeros = np.zeros(
            config['state_dim'] ** 2 + config['action_dim'] + 1
        )
        self.zeros = convert_to_tensor(self.zeros, store_gpu=self.store_gpu)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset['query_states'])

    def __getitem__(self, index):
        'Generates one sample of data'
        res = {
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }
        
        if self.dataset['context_next_states'] is not None:
            res['context_next_states'] = self.dataset['context_next_states'][index]
            
        # Add optimal_actions_per_state if available (for MDPs)
        if self.dataset['optimal_actions_per_state'] is not None:
            res['optimal_actions_per_state'] = self.dataset['optimal_actions_per_state'][index]

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            if 'context_next_states' in res:
                res['context_next_states'] = res['context_next_states'][perm]
            # Also shuffle optimal_actions_per_state if available
            if 'optimal_actions_per_state' in res:
                res['optimal_actions_per_state'] = res['optimal_actions_per_state'][perm]

        return res


class ImageDataset(Dataset):
    """"Dataset class for image-based data."""

    def __init__(self, paths, config, transform):
        config['store_gpu'] = False
        super().__init__(paths, config)
        self.transform = transform
        self.config = config

        context_filepaths = []
        query_images = []

        for traj in self.trajs:
            context_filepaths.append(traj['context_images'])
            query_image = self.transform(traj['query_image']).float()
            query_images.append(query_image)

        self.dataset.update({
            'context_filepaths': context_filepaths,
            'query_images': torch.stack(query_images),
        })

    def __getitem__(self, index):
        'Generates one sample of data'
        filepath = self.dataset['context_filepaths'][index]
        context_images = np.load(filepath)
        context_images = [self.transform(images) for images in context_images]
        context_images = torch.stack(context_images).float()

        query_images = self.dataset['query_images'][index]

        res = {
            'context_images': context_images,#.to(device),
            'context_states': self.dataset['context_states'][index],
            'context_actions': self.dataset['context_actions'][index],
            'context_rewards': self.dataset['context_rewards'][index],
            'query_images': query_images,#.to(device),
            'query_states': self.dataset['query_states'][index],
            'optimal_actions': self.dataset['optimal_actions'][index],
            'zeros': self.zeros,
        }
        
        if self.dataset['context_next_states'] is not None:
            res['context_next_states'] = self.dataset['context_next_states'][index]

        if self.shuffle:
            perm = torch.randperm(self.horizon)
            res['context_images'] = res['context_images'][perm]
            res['context_states'] = res['context_states'][perm]
            res['context_actions'] = res['context_actions'][perm]
            res['context_rewards'] = res['context_rewards'][perm]
            if 'context_next_states' in res:
                res['context_next_states'] = res['context_next_states'][perm]

        return res
