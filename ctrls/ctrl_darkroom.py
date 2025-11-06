import numpy as np
import scipy
import torch

from ctrls.ctrl_bandit import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class DarkroomOptPolicy(Controller):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.goal = env.goal

    def reset(self):
        return

    def act(self, state):
        return self.env.opt_action(state)        


class DarkroomZigzagPolicy(Controller):
    """
    Zigzag exploration policy for darkroom.
    Systematically explores the grid in a zigzag pattern:
    - Moves right until hitting boundary, then moves up one and goes left
    - Repeats until goal is found, then stays at goal
    - Ensures complete exploration of the grid
    """
    def __init__(self, envs, batch_size=1):
        super().__init__()
        self.envs = envs if isinstance(envs, list) else [envs]
        self.batch_size = batch_size
        self.dim = envs[0].dim if isinstance(envs, list) else envs.dim
        # Track current position and direction for each environment
        # Initialize on first call
        self.current_y = None
        self.going_right = None
        self.found_goal = None
        self.initialized = False
        
    def reset(self):
        """Reset internal state for each environment"""
        self.current_y = [0] * self.batch_size
        self.going_right = [True] * self.batch_size
        self.found_goal = [False] * self.batch_size
        self.initialized = True
        
    def set_batch(self, batch):
        """Interface compatibility - not used for zigzag policy"""
        pass
        
    def act(self, states):
        """
        Act based on current state and zigzag strategy.
        states: list of states from vectorized environment
        """
        if not self.initialized:
            self.reset()
        
        # Handle both list and single state (for compatibility)
        if not isinstance(states, list):
            states = [states]
        
        actions = []
        for env_idx, state in enumerate(states):
            state = np.array(state)
            env = self.envs[env_idx] if isinstance(self.envs, list) else self.envs
            
            # Check if we're at the goal
            if np.all(state == env.goal):
                self.found_goal[env_idx] = True
                # Stay at goal
                action = np.zeros(env.action_dim)
                action[4] = 1.0  # Stay action
                actions.append(action)
                continue
            
            # If we already found the goal, stay
            if self.found_goal[env_idx]:
                action = np.zeros(env.action_dim)
                action[4] = 1.0  # Stay action
                actions.append(action)
                continue
            
            # Zigzag strategy
            current_x, current_y = int(state[0]), int(state[1])
            
            # Update our tracked y position if we moved up
            if current_y > self.current_y[env_idx]:
                self.current_y[env_idx] = current_y
                # When we move up, switch direction
                self.going_right[env_idx] = not self.going_right[env_idx]
            
            action = np.zeros(env.action_dim)
            
            if self.going_right[env_idx]:
                # Going right: move right if not at boundary, else move up
                if current_x < self.dim - 1:
                    action[0] = 1.0  # Right
                else:
                    # At right boundary, move up
                    if current_y < self.dim - 1:
                        action[2] = 1.0  # Up
                        self.current_y[env_idx] = current_y + 1
                        self.going_right[env_idx] = False
                    else:
                        # At top-right corner, go left
                        action[1] = 1.0  # Left
                        self.going_right[env_idx] = False
            else:
                # Going left: move left if not at boundary, else move up
                if current_x > 0:
                    action[1] = 1.0  # Left
                else:
                    # At left boundary, move up
                    if current_y < self.dim - 1:
                        action[2] = 1.0  # Up
                        self.current_y[env_idx] = current_y + 1
                        self.going_right[env_idx] = True
                    else:
                        # At top-left corner, go right
                        action[0] = 1.0  # Right
                        self.going_right[env_idx] = True
            
            actions.append(action)
        
        return np.array(actions)


class DarkroomTransformerController(Controller):
    def __init__(self, model, batch_size=1, sample=False):
        self.model = model
        self.state_dim = model.config['state_dim']
        self.action_dim = model.config['action_dim']
        self.horizon = model.horizon
        self.zeros = torch.zeros(
            batch_size, self.state_dim ** 2 + self.action_dim + 1).float().to(device)
        self.sample = sample
        self.temp = 1.0
        self.batch_size = batch_size

    def act(self, state):
        self.batch['zeros'] = self.zeros

        states = torch.tensor(np.array(state)).float().to(device)
        if self.batch_size == 1:
            states = states[None, :]
        self.batch['query_states'] = states

        # Model returns (pred_actions, pred_rewards) tuple, unpack it
        pred_actions, _ = self.model(self.batch)
        # Extract predictions at the last position
        if self.batch_size == 1:
            actions = pred_actions[0, -1, :].cpu().detach().numpy()
        else:
            actions = pred_actions[:, -1, :].cpu().detach().numpy()

        if self.sample:
            if self.batch_size > 1:
                action_indices = []
                for idx in range(self.batch_size):
                    probs = scipy.special.softmax(actions[idx] / self.temp)
                    sampled_action = np.random.choice(
                        np.arange(self.action_dim), p=probs)
                    action_indices.append(sampled_action)
            else:
                probs = scipy.special.softmax(actions / self.temp)
                action_indices = [np.random.choice(
                    np.arange(self.action_dim), p=probs)]
        else:
            action_indices = np.argmax(actions, axis=-1)

        actions = np.zeros((self.batch_size, self.action_dim))
        actions[np.arange(self.batch_size), action_indices] = 1.0
        if self.batch_size == 1:
            actions = actions[0]
        return actions
