import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
from IPython import embed
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        # Set a large max_position_embeddings to support long sequences
        
        config = GPT2Config(
            n_positions=3 * (self.horizon + 1),  # Allow for long sequences with room
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=1,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=True,  # Enable caching for inference
        )
        self.transformer = GPT2Model(config)
        
        # # Disable positional embeddings since we're using dynamic length
        # # and Decision Transformer doesn't use positional embeddings
        # self.transformer.wpe.weight.requires_grad = False

        # Separate embeddings for state, action, and reward (like language modeling)
        self.embed_state = nn.Linear(self.state_dim, self.n_embd)
        self.embed_action = nn.Linear(self.action_dim, self.n_embd)
        self.embed_reward = nn.Linear(1, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        
        # Separate prediction heads for different indices (state=0, action=1, reward=2)
        self.predict_state = nn.Linear(self.n_embd, self.state_dim)
        self.predict_action = nn.Linear(self.n_embd, self.action_dim)
        self.predict_reward = nn.Linear(self.n_embd, 1)

    def forward(self, x, past_key_values=None, return_dict=False):
        """Forward pass.
        
        Args:
            x: Input dict with context_states, context_actions, context_rewards
            past_key_values: Optional cached key/values for incremental decoding
            return_dict: If True, returns dict with 'hidden_state' and 'past_key_values'
                 If False (default), returns (pred_actions, pred_rewards) for backward compatibility
        """
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']
        
        batch_size = context_states.shape[0]
        seq_len = context_states.shape[1]
        
        # Embed each modality separately
        state_embeds = self.embed_state(context_states)  # [batch, seq_len, n_embd]
        action_embeds = self.embed_action(context_actions)  # [batch, seq_len, n_embd]
        reward_embeds = self.embed_reward(context_rewards)  # [batch, seq_len, n_embd]
        
        # Stack them as separate positions: (s_0, a_0, r_0, s_1, a_1, r_1, ...)
        stacked_inputs = torch.stack(
            (state_embeds, action_embeds, reward_embeds), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.n_embd)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Create attention mask for variable-length sequences
        attention_mask = None
        if 'context_lengths' in x:
            # Create mask based on actual sequence lengths
            context_lengths = x['context_lengths']  # [batch]
            attention_mask = torch.zeros(batch_size, 3 * seq_len, device=context_states.device)
            for i in range(batch_size):
                actual_len = context_lengths[i].item()
                attention_mask[i, :(3 * actual_len)] = 1.0
        
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=self.test if past_key_values is None else True,
        )
        
        hidden_state = transformer_outputs['last_hidden_state']
        
        # Reshape output: [batch, seq_len, 3, n_embd] where dim 2 is (state, action, reward)
        x = hidden_state.reshape(batch_size, seq_len, 3, self.n_embd).permute(0, 2, 1, 3)
        
        # If return_dict, return transformer outputs for caching
        if return_dict:
            return {
                'hidden_state': x,  # [batch, 3, seq_len, n_embd] 
                'past_key_values': transformer_outputs.get('past_key_values'),
            }
        
        # Otherwise, return predictions for backward compatibility with training code
        # Extract predictions from state positions (index 0) for actions and rewards
        pred_actions = self.predict_action(x[:, 0])  # [batch, seq_len, action_dim]
        pred_rewards = self.predict_reward(x[:, 0])  # [batch, seq_len, 1]
        if self.test:
            return pred_actions
        else:
            return pred_actions, pred_rewards


class ImageTransformer(Transformer):
    """Transformer class for image-based data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']
        size = (size - 3) // 2 + 1
        size = (size - 3) // 1 + 1

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        # Separate embeddings for image+state, action, and reward
        self.embed_image_state = nn.Linear(self.im_embd + self.state_dim, self.n_embd)
        self.embed_action = nn.Linear(self.action_dim, self.n_embd)
        self.embed_reward = nn.Linear(1, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        
        # Separate prediction heads for different indices (image+state=0, action=1, reward=2)
        self.predict_action = nn.Linear(self.n_embd, self.action_dim)
        self.predict_reward = nn.Linear(self.n_embd, 1)
        
        # # Disable positional embeddings since we're using dynamic length
        # # and Decision Transformer doesn't use positional embeddings
        # self.transformer.wpe.weight.requires_grad = False

    def forward(self, x, past_key_values=None, return_dict=False):
        """Forward pass.
        
        Returns (pred_actions, pred_rewards) by default for backward compatibility,
        or dict with 'hidden_state' and 'past_key_values' if return_dict=True.
        """
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = context_images.shape[0]
        seq_len = context_images.shape[1]

        # Encode images
        image_seq = context_images.view(-1, *context_images.size()[2:])
        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, seq_len, self.im_embd)

        # Combine image and state embeddings
        image_state_embeds = self.embed_image_state(
            torch.cat([image_enc_seq, context_states], dim=2)
        )  # [batch, seq_len, n_embd]
        
        action_embeds = self.embed_action(context_actions)  # [batch, seq_len, n_embd]
        reward_embeds = self.embed_reward(context_rewards)  # [batch, seq_len, n_embd]

        # Stack them as separate positions: (img+state_0, a_0, r_0, img+state_1, a_1, r_1, ...)
        stacked_inputs = torch.stack(
            (image_state_embeds, action_embeds, reward_embeds), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.n_embd)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create attention mask for variable-length sequences
        attention_mask = None
        if 'context_lengths' in x:
            # Create mask based on actual sequence lengths
            context_lengths = x['context_lengths']  # [batch]
            attention_mask = torch.zeros(batch_size, 3 * seq_len, device=context_images.device)
            for i in range(batch_size):
                actual_len = context_lengths[i].item()
                attention_mask[i, :(3 * actual_len)] = 1.0

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=self.test if past_key_values is None else True,
        )
        
        hidden_state = transformer_outputs['last_hidden_state']
        
        # Reshape output: [batch, seq_len, 3, n_embd] where dim 2 is (image+state, action, reward)
        x = hidden_state.reshape(batch_size, seq_len, 3, self.n_embd).permute(0, 2, 1, 3)
        
        # If return_dict, return transformer outputs for caching
        if return_dict:
            return {
                'hidden_state': x,  # [batch, 3, seq_len, n_embd]
                'past_key_values': transformer_outputs.get('past_key_values'),
            }
        
        # Otherwise, return predictions for backward compatibility
        pred_actions = self.predict_action(x[:, 0])  # [batch, seq_len, action_dim]
        pred_rewards = self.predict_reward(x[:, 0])  # [batch, seq_len, 1]
        
        return pred_actions, pred_rewards
