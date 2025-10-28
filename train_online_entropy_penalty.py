import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn', force=True)

import argparse
import os
import time
import yaml
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


def get_beta_value(confidence_type, iteration, horizon, **kwargs):
    """Calculate beta (entropy penalty coefficient) based on the specified confidence function type.

    Beta decreases over time (opposite of confidence in the original script).
    For linear: starts at confidence_start and decreases to 0.
    For stepped: starts at confidence_start, decreases to 0 at max_position, then stays at 0.
    For linear_interpolate: starts at confidence_start and decreases to 0 (used with interpolated loss).
    For self_referential_entropy: beta = alpha * H[π(·|s_h, D_{h-1})]
    For reward_variance: beta = σ²(r_{1:h-1}) / (μ(r_{1:h-1})² + ε)
    For inverse_sqrt: beta = beta_0 / sqrt(h+1)
    For reward_prediction_error: beta = |r_h - E[r_{1:h-1}]| / (std(r_{1:h-1}) + ε)
    """
    if confidence_type == 'linear':
        confidence_start = kwargs.get('confidence_start', 0.1)
        # Linearly decrease from confidence_start to 0
        return confidence_start * (1 - iteration / horizon)

    elif confidence_type == 'stepped':
        confidence_start = kwargs.get('confidence_start', 0.1)
        max_position = kwargs.get('max_position', 40)
        if iteration >= max_position:
            return 0.0
        else:
            # Linear decrease from confidence_start to 0 up to max_position
            return confidence_start * (1 - iteration / max_position)

    elif confidence_type == 'linear_interpolate':
        confidence_start = kwargs.get('confidence_start', 0.1)
        # Linearly decrease from confidence_start to 0
        # Used with loss = (1-beta) * ce_loss - beta * entropy
        return confidence_start * (1 - iteration / horizon)

    elif confidence_type == 'constant':
        return kwargs.get('confidence_value', 0.1)

    elif confidence_type == 'self_referential_entropy':
        # beta = alpha * H[π(·|s_h, D_{h-1})]
        # Policy entropy directly controls regularization
        # High entropy → policy already exploring → maintain it
        # Low entropy → policy converging → let it exploit
        policy_entropy = kwargs.get('policy_entropy', 0.0)
        alpha = kwargs.get('alpha', 0.05)  # Scaling factor (REDUCED from 0.1)
        return alpha * policy_entropy

    elif confidence_type == 'reward_variance':
        # beta = α * min(σ²(r_{1:h-1}) / (μ(r_{1:h-1})² + ε), β_max)
        # Coefficient-of-variation style normalization with clipping
        reward_mean = kwargs.get('reward_mean', 0.0)
        reward_var = kwargs.get('reward_var', 0.0)
        epsilon = kwargs.get('epsilon', 1e-6)
        alpha = kwargs.get('alpha', 1.0)  # Scaling factor
        beta_max = kwargs.get('beta_max', 0.3)  # Maximum beta to prevent explosion

        if abs(reward_mean) < epsilon:
            # If mean is too small, return base value
            return beta_max * 0.5

        raw_beta = reward_var / (reward_mean ** 2 + epsilon)
        return min(alpha * raw_beta, beta_max)

    elif confidence_type == 'inverse_sqrt':
        # beta = beta_0 / sqrt(h+1)
        # From online learning theory - optimal regret bounds
        # FIXED: Use much smaller beta_0 to avoid over-regularization
        beta_0 = kwargs.get('beta_0', 0.5)  # REDUCED from 1.0
        return beta_0 / np.sqrt(iteration + 1)

    elif confidence_type == 'reward_prediction_error':
        # beta = α * min(|r_actual - r_predicted|, β_max)
        # Reward prediction error: actual vs predicted rewards with clipping
        reward_prediction_error = kwargs.get('reward_prediction_error', 0.0)
        alpha = kwargs.get('alpha', 0.5)  # Scaling factor
        beta_max = kwargs.get('beta_max', 0.5)  # Maximum beta
        
        # Simple formula: β = α * min(error, β_max)
        return min(alpha * reward_prediction_error, beta_max)

    else:
        raise ValueError(f"Unknown confidence type: {confidence_type}")


class OnlineEnvironmentManager:
    """Manages growing contexts for online environments with entropy-penalized cross-entropy loss."""

    def __init__(self, envs, env_type, horizon, config, model=None, confidence_type='linear', **kwargs):
        self.envs = envs
        self.env_type = env_type
        self.horizon = horizon
        self.config = config
        self.model = model
        self.confidence_type = confidence_type
        self.confidence_kwargs = kwargs
        self.kwargs = kwargs
        self.reset_contexts()

    def reset_contexts(self):
        """Reset all environment contexts at the start of each epoch."""
        self.contexts = {i: {
            'states': [],
            'actions': [],
            'rewards': []
        } for i in range(len(self.envs))}

    def step_and_learn(self, optimizer, loss_fn, action_dim, horizon, current_iteration, debug=False):
        """Step environments and learn using cross-entropy loss minus beta-weighted entropy penalty."""
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

            # DEBUG: Current context info (BEFORE adding new transition)
            context_size = len(self.contexts[env_idx]['states']) if debug else 0
            recent_rewards = self.contexts[env_idx]['rewards'][-3:] if debug and self.contexts[env_idx]['rewards'] else []
                
            # Store data for batched processing
            env_data.append({
                'env_idx': env_idx,
                'env': env,
                'current_state': current_state,
                'model_action': None,  # Will be filled by batched prediction
                'reward': None,  # Will be filled after action execution
                'next_state': None,  # Will be filled after action execution
                'context_size': context_size,
                'recent_rewards': recent_rewards
            })

        # PHASE 1B: Batched action prediction for all environments at once
        if self.model is not None:
            # Create input samples for all environments
            inference_samples = []
            for data in env_data:
                sample = self._create_model_input_sample(data['env_idx'], data['current_state'])
                
                # Add dummy optimal action to make it compatible with batch_to_tensors
                if self.env_type in ['bandit', 'linear_bandit']:
                    sample['optimal_action'] = np.zeros(data['env'].dim)  # dummy
                elif self.env_type.startswith('darkroom'):
                    sample['optimal_action'] = np.zeros(5)  # dummy
                elif self.env_type == 'miniworld':
                    sample['optimal_action'] = np.zeros(data['env'].action_space.n)  # dummy
                
                inference_samples.append(sample)
            
            # Batch inference: single forward pass for all environments
            with torch.no_grad():
                if self.env_type == 'miniworld':
                    inference_batch = batch_to_tensors(inference_samples, self.config, self.env_type, self.kwargs.get('transform'))
                else:
                    inference_batch = batch_to_tensors(inference_samples, self.config, self.env_type)
                
                inference_batch = {k: v.to(device) for k, v in inference_batch.items()}
                
                self.model.eval()
                model_outputs = self.model(inference_batch)
                pred_actions = model_outputs[0]
                self.model.train()
            
            # Extract actions for each environment
            for i, data in enumerate(env_data):
                env_idx = data['env_idx']
                context_len = len(self.contexts[env_idx]['states'])
                sample_pos = min(context_len, pred_actions.shape[1] - 1)
                
                # Sample action from predicted distribution
                if self.env_type == 'miniworld':
                    probs = torch.softmax(pred_actions[i, sample_pos], dim=-1)
                    action_idx = torch.multinomial(probs, 1).item()
                    action = np.zeros(data['env'].action_space.n)
                    action[action_idx] = 1.0
                else:
                    probs = torch.softmax(pred_actions[i, sample_pos], dim=-1)
                    action_idx = torch.multinomial(probs, 1).item()
                    action = np.zeros(len(probs))
                    action[action_idx] = 1.0
                
                data['model_action'] = action
        else:
            # Fallback to random actions
            for data in env_data:
                data['model_action'] = self._get_random_action(data['env'])
        
        # PHASE 1C: Execute actions in environments
        for data in env_data:
            env = data['env']
            current_state = data['current_state']
            model_action = data['model_action']
            
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
            
            data['reward'] = reward
            data['next_state'] = next_state

        # PHASE 2: Create batch of training samples
        # NOTE: At this point, contexts contain past (s, a, r) tuples
        # We'll add the current state to create the full sequence for prediction
        for data in env_data:
            # Add current state (query state) and dummy action/reward for prediction
            self.contexts[data['env_idx']]['states'].append(data['current_state'])
            self.contexts[data['env_idx']]['actions'].append(data['model_action'])  # Will be replaced with dummy
            self.contexts[data['env_idx']]['rewards'].append(0.0)  # Dummy, will predict
            
            # Truncate if too long
            if len(self.contexts[data['env_idx']]['states']) > self.horizon + 1:  # +1 for query state
                for key in self.contexts[data['env_idx']]:
                    self.contexts[data['env_idx']][key] = self.contexts[data['env_idx']][key][-(self.horizon + 1):]
        
        training_samples = []
        for data in env_data:
            training_sample = self._create_training_sample(data['env'], data['env_idx'], data['current_state'])
            training_samples.append(training_sample)
        
        # Now update contexts with real rewards (after training samples created)
        for data in env_data:
            # Update the last reward to be the actual reward
            if len(self.contexts[data['env_idx']]['rewards']) > 0:
                self.contexts[data['env_idx']]['rewards'][-1] = data['reward']

        # PHASE 3: Batched forward and backward pass
        if self.env_type == 'miniworld':
            batch = batch_to_tensors(training_samples, self.config, self.env_type, self.kwargs.get('transform'))
        else:
            batch = batch_to_tensors(training_samples, self.config, self.env_type)

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass for entire batch
        true_actions = batch['optimal_actions']  # Shape: [batch_size, action_dim]
        for key, value in batch.items():
            #printw the first batch element of the value
            printw(f"{key}: {value[0]}")
        model_outputs = self.model(batch)        # Returns (pred_actions, pred_rewards)
        pred_actions, pred_rewards = model_outputs[0], model_outputs[1]  # Shape: [batch_size, max_seq_len, action_dim], [batch_size, max_seq_len, 1]
        printw(f"pred_actions: {pred_actions[0]}")
        printw(f"pred_rewards: {pred_rewards[0]}")
        loss_positions = batch['loss_positions'] # Shape: [batch_size]
        
        # Extract predictions at specific positions for each batch item
        batch_size = true_actions.shape[0]
        selected_pred_actions = torch.zeros_like(true_actions)  # [batch_size, action_dim]
        selected_pred_rewards = torch.zeros(batch_size, 1).to(device)  # [batch_size, 1]

        for i in range(batch_size):
            pos = loss_positions[i].item()
            selected_pred_actions[i] = pred_actions[i, pos]  # Get prediction at specific position
            selected_pred_rewards[i] = pred_rewards[i, pos]   # Get reward prediction at specific position

        # Compute statistics needed for different beta schedules
        beta_kwargs = dict(self.confidence_kwargs)

        # 1. Policy entropy (for self_referential_entropy)
        if self.confidence_type == 'self_referential_entropy':
            pred_probs = torch.softmax(selected_pred_actions, dim=-1)
            policy_entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1).mean().item()
            beta_kwargs['policy_entropy'] = policy_entropy

        # 2. Reward prediction error (for reward_prediction_error only)
        if self.confidence_type == 'reward_prediction_error':
            # Get actual rewards from the current step (stored in env_data)
            actual_rewards = np.array([data['reward'] for data in env_data])
            # Get predicted rewards from the model (detach to avoid gradient issues)
            predicted_rewards = selected_pred_rewards.detach().cpu().numpy().flatten()
            # Mean absolute prediction error for the current position
            # printw(f"actual rewards: {actual_rewards}")
            # printw(f"predicted rewards: {predicted_rewards}")
            # printw(f"reward prediction error: {np.abs(predicted_rewards - actual_rewards)}")
            reward_prediction_error = np.mean(np.abs(predicted_rewards - actual_rewards))
            beta_kwargs['reward_prediction_error'] = reward_prediction_error
            
        # Reward statistics (for reward_variance - keep this for other confidence types)
        elif self.confidence_type == 'reward_variance':
            # Aggregate rewards from all environments up to current iteration
            all_rewards = []
            current_rewards = []
            for env_idx in range(len(self.envs)):
                env_rewards = self.contexts[env_idx]['rewards']
                if len(env_rewards) > 0:
                    all_rewards.extend(env_rewards[:-1])  # All but last (history)
                    current_rewards.append(env_rewards[-1])  # Current reward

            if len(all_rewards) > 0:
                reward_mean = np.mean(all_rewards)
                reward_var = np.var(all_rewards)
                reward_std = np.std(all_rewards)
                beta_kwargs['reward_mean'] = reward_mean
                beta_kwargs['reward_var'] = reward_var
                beta_kwargs['reward_std'] = reward_std

        # Calculate beta (entropy penalty coefficient)
        beta = get_beta_value(
            self.confidence_type, current_iteration, horizon,
            **beta_kwargs
        )

        # ENTROPY-PENALIZED CROSS-ENTROPY LOSS
        # For linear/stepped/constant: Loss = CE(pred, true) - beta * H(pred)
        # For linear_interpolate: Loss = (1-beta) * CE(pred, true) - beta * H(pred)
        # where H(pred) = -sum(p * log(p)) is the entropy of model output
        
        # Standard cross-entropy loss
        ce_loss = loss_fn(selected_pred_actions, true_actions)
        
        # Get true rewards for reward prediction loss
        true_rewards = torch.zeros(batch_size, 1).to(device)
        for i, data in enumerate(env_data):
            true_rewards[i] = data['reward']
        
        # Reward prediction loss (MSE)
        reward_loss = torch.nn.functional.mse_loss(selected_pred_rewards, true_rewards)
        
        # Calculate entropy of model predictions
        pred_probs = torch.softmax(selected_pred_actions, dim=-1)
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=-1)
        total_entropy = entropy.sum()
        
        # Final loss: depends on confidence type
        # Total loss = action_loss + reward_loss - beta * entropy
        if self.confidence_type == 'linear_interpolate':
            # Interpolated loss: (1-beta) * CE + reward_loss - beta * entropy
            loss = (1 - beta) * ce_loss + reward_loss - beta * total_entropy
        else:
            # Standard entropy penalty: CE + reward_loss - beta * entropy
            loss = ce_loss + reward_loss - beta * total_entropy
        
        # Single backward pass for entire batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute average loss per sample
        total_loss = loss.item() / batch_size

        # PHASE 4: Debug output (show first 3 environments)
        if debug and len(env_data) > 0:
            # Show debug info for first environment
            num_debug_envs = min(1, len(env_data))
            
            for debug_idx in range(num_debug_envs):
                data = env_data[debug_idx]
                env_idx = data['env_idx']
                
                # Extract predictions for this environment
                pred_probs_display = torch.softmax(pred_actions.view(batch_size, horizon, action_dim)[debug_idx, -1], dim=-1)
                optimal_action = training_samples[debug_idx]['optimal_action']
                optimal_action_idx = np.argmax(optimal_action)
                sampled_action_idx = np.argmax(data['model_action'])
                sampling_match = sampled_action_idx == optimal_action_idx
                
                # Format probability distribution for display
                if self.env_type in ['bandit', 'linear_bandit']:
                    prob_str = ", ".join([f"A{i}:{pred_probs_display[i].item():.3f}" for i in range(len(pred_probs_display))])
                else:
                    prob_str = ", ".join([f"A{i}:{pred_probs_display[i].item():.3f}" for i in range(len(pred_probs_display))])
                    
                # Format confidence function info for display
                conf_info = f"{self.confidence_type}"
                if self.confidence_type in ['linear', 'linear_interpolate']:
                    conf_info += f"(start={self.confidence_kwargs.get('confidence_start', 0.1)})"
                elif self.confidence_type == 'stepped':
                    conf_info += f"(start={self.confidence_kwargs.get('confidence_start', 0.1)}, max_pos={self.confidence_kwargs.get('max_position', 40)})"
                    
                avg_entropy = (total_entropy / batch_size).item()
                ce_loss_avg = (ce_loss / batch_size).item()
                    
                print(f"Batch: Env {env_idx}: Context={data['context_size']}, Beta={beta:.3f} ({conf_info}), Probs=[{prob_str}], SampledAction={sampled_action_idx}, OptimalAction={optimal_action_idx}, Match={sampling_match}, Reward={data['reward']:.3f}, CE_Loss={ce_loss_avg:.4f}, Entropy={avg_entropy:.4f}, Total_Loss={total_loss:.4f}, BatchSize={batch_size}")
                
                step_debug = {
                    'env_idx': env_idx,
                    'current_state': data['current_state'].tolist(),
                    'context_size': data['context_size'],
                    'recent_rewards': data['recent_rewards'],
                    'beta': beta,
                    'confidence_type': self.confidence_type,
                    'pred_probs': pred_probs_display.tolist(),
                    'sampled_action_idx': sampled_action_idx,
                    'optimal_action_idx': optimal_action_idx,
                    'sampling_match': sampling_match,
                    'reward': data['reward'],
                    'ce_loss': ce_loss_avg,
                    'entropy': avg_entropy,
                    'step_loss': total_loss,
                    'next_state': data['next_state'].tolist(),
                    'batch_size': batch_size
                }
                debug_info['env_steps'].append(step_debug)

        if debug:
            return total_loss, debug_info, beta
        return total_loss, beta

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


            model_outputs = self.model(batch)
            
            pred_actions = model_outputs[0]

            # Get the position to sample from (last non-padded position)
            context_len = len(self.contexts[env_idx]['states'])
            sample_pos = min(context_len, pred_actions.shape[1] - 1)
            
            # Sample action (you can use softmax sampling or argmax)
            if self.env_type == 'miniworld':
                # For discrete actions, use softmax sampling
                probs = torch.softmax(pred_actions[0, sample_pos], dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
                action = np.zeros(self.envs[env_idx].action_space.n)
                action[action_idx] = 1.0
            else:
                # For continuous/bandit actions, use softmax sampling over action space
                probs = torch.softmax(pred_actions[0, sample_pos], dim=-1)
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
        """Create sample for model input - just use the context which already includes the current state."""
        context = self.contexts[env_idx]
        context_len = len(context['states'])
        
        # Handle empty context case - should have at least current state
        if context_len == 0:
            context_len = 1
            states = np.array([query_state])
            
            # Determine action dimension
            if self.env_type in ['bandit', 'linear_bandit']:
                action_dim = self.envs[env_idx].dim
            elif self.env_type.startswith('darkroom'):
                action_dim = 5
            elif self.env_type == 'miniworld':
                action_dim = self.envs[env_idx].action_space.n
            else:
                action_dim = self.config['action_dim']
            
            actions = np.zeros((1, action_dim))
            rewards = np.array([0.0])
            
            if self.env_type == 'miniworld':
                env = self.envs[env_idx]
                return {
                    'context_images': np.array([env.render_obs()]),
                    'context_states': states,
                    'context_actions': actions,
                    'context_rewards': rewards,
                    'loss_position': 0,
                    'context_length': context_len,
                }
            else:
                return {
                    'context_states': states,
                    'context_actions': actions,
                    'context_rewards': rewards,
                    'loss_position': 0,
                    'context_length': context_len,
                }
        
        if self.env_type == 'miniworld':
            env = self.envs[env_idx]
            context_images = context.get('images', [])
            
            # Get sequences (already includes current state at the end)
            states = np.array(context['states'])
            actions = np.array(context['actions'])
            rewards = np.array(context['rewards'])
            images = np.array(context_images) if context_images else None
            
            # Find position to compute loss (last position with current state)
            loss_position = context_len - 1
            
            return {
                'context_images': images,
                'context_states': states,
                'context_actions': actions,
                'context_rewards': rewards,
                'loss_position': loss_position,
                'context_length': context_len,
            }
        else:
            # Get sequences (already includes current state at the end)
            states = np.array(context['states'])
            actions = np.array(context['actions'])
            rewards = np.array(context['rewards'])
            
            # Find position to compute loss (last position with current state)
            loss_position = context_len - 1
            
            return {
                'context_states': states,
                'context_actions': actions,
                'context_rewards': rewards,
                'loss_position': loss_position,
                'context_length': context_len,
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
    """Convert standard (non-image) batch data to tensors with variable-length sequences."""
    state_dim = config['state_dim']
    action_dim = config['action_dim']

    batch_size = len(batch_data)

    # Get maximum sequence length in batch
    max_len = max(sample['context_length'] for sample in batch_data)
    
    # Initialize tensors with max length
    optimal_actions = torch.zeros(batch_size, action_dim)
    context_states = torch.zeros(batch_size, max_len, state_dim)
    context_actions = torch.zeros(batch_size, max_len, action_dim)
    context_rewards = torch.zeros(batch_size, max_len, 1)
    loss_positions = torch.zeros(batch_size, dtype=torch.long)
    
    for i, sample in enumerate(batch_data):
        optimal_actions[i] = torch.from_numpy(sample['optimal_action']).float()
        seq_len = sample['context_length']
        
        # Fill sequences (context already includes current state at the end)
        context_states[i, :seq_len] = torch.from_numpy(sample['context_states']).float()
        context_actions[i, :seq_len] = torch.from_numpy(sample['context_actions']).float()
        
        # Handle context_rewards shape
        rewards_array = sample['context_rewards']
        if len(rewards_array.shape) == 1:
            rewards_array = rewards_array.reshape(-1, 1)
        context_rewards[i, :seq_len] = torch.from_numpy(rewards_array).float()
        
        loss_positions[i] = sample['loss_position']

    return {
        'optimal_actions': optimal_actions,
        'context_states': context_states,
        'context_actions': context_actions,
        'context_rewards': context_rewards,
        'loss_positions': loss_positions,
        'context_lengths': torch.tensor([sample['context_length'] for sample in batch_data], dtype=torch.long),
    }


def batch_to_tensors_miniworld(batch_data, config, transform):
    """Convert miniworld batch data to tensors with variable-length sequences."""
    state_dim = config['state_dim']
    action_dim = config['action_dim']
    image_size = config['image_size']

    batch_size = len(batch_data)
    
    # Get maximum sequence length in batch
    max_len = max(sample['context_length'] for sample in batch_data)

    # Initialize tensors with max length
    optimal_actions = torch.zeros(batch_size, action_dim)
    context_images = torch.zeros(batch_size, max_len, 3, image_size, image_size)
    context_states = torch.zeros(batch_size, max_len, state_dim)
    context_actions = torch.zeros(batch_size, max_len, action_dim)
    context_rewards = torch.zeros(batch_size, max_len, 1)
    loss_positions = torch.zeros(batch_size, dtype=torch.long)

    for i, sample in enumerate(batch_data):
        optimal_actions[i] = torch.from_numpy(sample['optimal_action']).float()
        seq_len = sample['context_length']
        
        # Transform images
        for h in range(seq_len):
            if sample['context_images'] is not None and len(sample['context_images']) > 0:
                context_img = transform(sample['context_images'][h])
                context_images[i, h] = context_img
        
        # Fill sequences
        context_states[i, :seq_len] = torch.from_numpy(sample['context_states']).float()
        context_actions[i, :seq_len] = torch.from_numpy(sample['context_actions']).float()
        
        # Handle context_rewards shape
        rewards_array = sample['context_rewards']
        if len(rewards_array.shape) == 1:
            rewards_array = rewards_array.reshape(-1, 1)
        context_rewards[i, :seq_len] = torch.from_numpy(rewards_array).float()
        
        loss_positions[i] = sample['loss_position']

    return {
        'optimal_actions': optimal_actions,
        'context_images': context_images,
        'context_states': context_states,
        'context_actions': context_actions,
        'context_rewards': context_rewards,
        'loss_positions': loss_positions,
        'context_lengths': torch.tensor([sample['context_length'] for sample in batch_data], dtype=torch.long),
    }


if __name__ == '__main__':
    if not os.path.exists('figs/loss'):
        os.makedirs('figs/loss', exist_ok=True)
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')

    args = vars(parser.parse_args())

    # Load config from YAML file
    with open(args['config'], 'r') as f:
        config_args = yaml.safe_load(f)

    # Use config values as args
    args = config_args

    print("Args: ", args)

    # Determine experiment identifier (job ID or timestamp)
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        experiment_id = job_id
    else:
        experiment_id = str(int(time.time()))

    print(f"Experiment ID: {experiment_id}")

    env = args['env']
    n_envs = args.get('envs', 100)
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
    num_epochs = args.get('n_epoch', 100)  # Online uses n_epoch
    seed = args.get('seed', 0)
    lin_d = args.get('lin_d', 2)  # Only needed for linear_bandit
    samples_per_iter = args.get('samples_per_iter', 64)
    debug_mode = args.get('debug', False)

    # Beta (entropy penalty) parameters
    confidence_type = args.get('confidence_type', 'linear')
    confidence_start = args.get('confidence_start', 0.1)  # Default 0.1
    max_position = args.get('max_position', 40)  # Default 40
    if max_position <= 0:
        max_position = 40
    confidence_value = args.get('confidence_value', 0.1)

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
        'confidence_type': confidence_type,
    }

    # Add confidence-specific parameters to model config
    if confidence_type in ['linear', 'stepped', 'linear_interpolate']:
        model_config['confidence_start'] = confidence_start
    if confidence_type == 'stepped':
        model_config['max_position'] = max_position
    if confidence_type == 'constant':
        model_config['confidence_value'] = confidence_value

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

    # Update filename to include entropy penalty information
    filename_suffix = f"_entropy_{confidence_type}"
    if confidence_type in ['linear', 'stepped', 'linear_interpolate']:
        filename_suffix += f"_start{confidence_start}"
    if confidence_type == 'stepped':
        filename_suffix += f"_maxpos{max_position}"
    if confidence_type == 'constant':
        filename_suffix += f"_val{confidence_value}"
    
    filename = filename.replace('.pt', f'{filename_suffix}.pt')

    # Create experiment directory structure
    experiment_dir = f'models/{env}/{experiment_id}'
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config file in experiment directory
    config_path = f'{experiment_dir}/config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(args, f, default_flow_style=False)

    log_filename = f'{experiment_dir}/logs.txt'
    with open(log_filename, 'w') as f:
        pass
    def printw(string):
        print(string)
        with open(log_filename, 'a') as f:
            print(string, file=f)

    train_loss = []
    step_losses = []  # Store loss for every step (iteration)
    beta_schedule = []
    total_iterations = 0

    # Create confidence kwargs dictionary
    confidence_kwargs = {
        'confidence_start': confidence_start,
        'max_position': max_position,
        'confidence_value': confidence_value,
    }

    # Create online environment manager with entropy penalty loss
    if env == 'miniworld':
        env_manager = OnlineEnvironmentManager(
            [gym_env] * len(train_env_ids), env, horizon, config, model,
            confidence_type=confidence_type, transform=transform, target_shape=target_shape, 
            env_ids=train_env_ids, **confidence_kwargs
        )
    elif env == 'linear_bandit':
        env_manager = OnlineEnvironmentManager(
            train_envs, env, horizon, config, model, confidence_type=confidence_type, 
            arms=arms, **confidence_kwargs
        )
    else:
        env_manager = OnlineEnvironmentManager(
            train_envs, env, horizon, config, model, confidence_type=confidence_type, 
            **confidence_kwargs
        )

    printw(f"Starting online training with entropy penalty (beta schedule: {confidence_type}) for {env}")
    printw(f"Model filename: {filename}")
    printw(f"Confidence type: {confidence_type}")
    if confidence_type in ['linear', 'stepped', 'linear_interpolate']:
        printw(f"Confidence start (beta_0): {confidence_start}")
    if confidence_type == 'stepped':
        printw(f"Max position: {max_position}")
    if confidence_type == 'constant':
        printw(f"Confidence value (beta): {confidence_value}")
    if confidence_type == 'linear_interpolate':
        printw(f"Loss formula: (1-beta) * CE - beta * Entropy")

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
                iteration_loss, debug_info, beta = env_manager.step_and_learn(optimizer, loss_fn, action_dim, horizon, iteration, debug=True)
                printw(f"\n=== DEBUG: Epoch {epoch+1}, Iteration {iteration} ===")
            else:
                iteration_loss, beta = env_manager.step_and_learn(optimizer, loss_fn, action_dim, horizon, iteration)

            epoch_train_loss += iteration_loss
            epoch_iterations += 1
            total_iterations += 1
            step_losses.append(iteration_loss)  # Store step-level loss
            beta_schedule.append(beta)

            if iteration % 1 == 0:
                conf_info = confidence_type
                if confidence_type == 'stepped':
                    conf_info += f"(max_pos={max_position})"
                printw(f"Epoch {epoch+1}, Iteration {iteration}/{horizon}, Loss: {iteration_loss:.6f}, Beta: {beta:.3f} ({conf_info})")

        train_loss.append(epoch_train_loss / epoch_iterations)
        end_time = time.time()
        printw(f"\tTrain loss: {train_loss[-1]:.6f}")
        printw(f"\tTrain time: {end_time - start_time:.2f}s")
        printw(f"\tTotal iterations: {total_iterations}")
        printw(f"\tFinal beta: {beta:.3f}")

        # Checkpointing
        if (epoch + 1) % 1 == 0 or (env == 'linear_bandit' and (epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), f'{experiment_dir}/epoch{epoch+1}.pt')

        # Plotting
        if (epoch + 1) % 1 == 0:
            printw(f"Epoch: {epoch + 1}")
            printw(f"Train Loss: {train_loss[-1]:.6f}")
            printw("\n")

            # Plot training loss
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.yscale('log')
            # Plot loss for every step, not every epoch
            if len(step_losses) > 1:
                plt.plot(step_losses, label="Train Loss")
            plt.legend()
            plt.title(f"Online Training Loss (Entropy Penalty) - {env} ({confidence_type.title()} Beta)")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            
            # Plot beta schedule
            plt.subplot(1, 2, 2)
            if len(beta_schedule) > 0:
                plt.plot(beta_schedule, label=f"Beta Schedule ({confidence_type})")
            plt.legend()
            plt.title(f"{confidence_type.title()} Beta Schedule - {env}")
            plt.xlabel("Iteration")
            plt.ylabel("Beta (Entropy Penalty Coefficient)")
            # Adjust y-axis to fit all data
            if len(beta_schedule) > 0:
                plt.ylim(bottom=0)
            
            plt.tight_layout()
            plt.savefig(f"{experiment_dir}/train_loss.png")
            plt.clf()

    torch.save(model.state_dict(), f'{experiment_dir}/final_model.pt')
    printw("Online training with entropy penalty completed!")

