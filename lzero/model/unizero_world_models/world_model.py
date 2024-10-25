import logging
from typing import Any, Tuple
from typing import Optional
from typing import Union, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical, Independent, Normal
from torch.distributions import TransformedDistribution, TanhTransform

from lzero.model.common import SimNorm
from lzero.model.utils import cal_dormant_ratio
from .kv_caching import KeysValues
from .slicer import Head, PolicyHeadCont
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import LossWithIntermediateLosses, init_weights
from .utils import WorldModelOutput, hash_state

logging.getLogger().setLevel(logging.DEBUG)


class WorldModel(nn.Module):
    """
    Overview:
        The WorldModel class is responsible for the scalable latent world model of UniZero (https://arxiv.org/abs/2406.10667),
        which is used to predict the next latent state, rewards, policy, and value based on the current latent state and action.
        The world model consists of three main components:
            - a tokenizer, which encodes observations into embeddings,
            - a transformer, which processes the input sequences,
            - and heads, which generate the logits for observations, rewards, policy, and value.
    """

    def __init__(self, config: TransformerConfig, tokenizer) -> None:
        """
        Overview:
            Initialize the WorldModel class.
        Arguments:
            - config (:obj:`TransformerConfig`): The configuration for the transformer.
            - tokenizer (:obj:`Tokenizer`): The tokenizer.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.transformer = Transformer(self.config)

        if self.config.device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Move all modules to the specified device
        print(f"self.device: {self.device}")
        self.to(self.device)

        # Initialize configuration parameters
        self._initialize_config_parameters()

        # Initialize patterns for block masks
        self._initialize_patterns()

        self.hidden_size = config.embed_dim // config.num_heads

        # Position embedding
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim, device=self.device)
        self.precompute_pos_emb_diff_kv()
        print(f"self.pos_emb.weight.device: {self.pos_emb.weight.device}")
        self.continuous_action_space = self.config.continuous_action_space

        # Initialize action embedding table
        if self.continuous_action_space:
            # TODO: check the effect of SimNorm
            self.act_embedding_table = nn.Sequential(
                nn.Linear(config.action_space_size, config.embed_dim, device=self.device, bias=False),
                SimNorm(simnorm_dim=self.group_size))
        else:
            # for discrete action space
            self.act_embedding_table = nn.Embedding(config.action_space_size, config.embed_dim, device=self.device)
            print(f"self.act_embedding_table.weight.device: {self.act_embedding_table.weight.device}")

        # Head modules
        self.head_rewards = self._create_head(self.act_tokens_pattern, self.support_size)
        self.head_observations = self._create_head(self.all_but_last_latent_state_pattern, self.obs_per_embdding_dim,
                                                   self.sim_norm)  # NOTE: we add a sim_norm to the head for observations
        if self.continuous_action_space:
            self.sigma_type = self.config.sigma_type
            if self.sigma_type == 'fixed':
                self.fixed_sigma_value = self.config.fixed_sigma_value
            else:
                self.fixed_sigma_value = 0.5
            self.bound_type = self.config.bound_type
            self.head_policy = self._create_head_cont(self.value_policy_tokens_pattern, self.action_space_size)
        else:
            self.head_policy = self._create_head(self.value_policy_tokens_pattern, self.action_space_size)
        self.head_value = self._create_head(self.value_policy_tokens_pattern, self.support_size)

        # Apply weight initialization, the order is important
        self.apply(lambda module: init_weights(module, norm_type=self.config.norm_type))
        self._initialize_last_layer()

        # Cache structures
        self._initialize_cache_structures()

        # Projection input dimension
        self._initialize_projection_input_dim()

        # Hit count and query count statistics
        self._initialize_statistics()

        # Initialize keys and values for transformer
        self._initialize_transformer_keys_values()

        self.latent_recon_loss = torch.tensor(0., device=self.device)
        self.perceptual_loss = torch.tensor(0., device=self.device)

        # TODO: check the size of the shared pool
        # for self.kv_cache_recurrent_infer
        # If needed, recurrent_infer should store the results of the one MCTS search. 
        self.shared_pool_size = int(50*self.env_num)
        self.shared_pool_recur_infer = [None] * self.shared_pool_size
        self.shared_pool_index = 0

        # for self.kv_cache_init_infer
        # In contrast, init_infer only needs to retain the results of the most recent step.
        # self.shared_pool_size_init = int(2*self.env_num)
        self.shared_pool_size_init = int(2) # NOTE: 过多会导致检索到错误的kvcache吗
        self.shared_pool_init_infer = [[None] * self.shared_pool_size_init for _ in range(self.env_num)]
        self.shared_pool_index_init_envs = [0 for _ in range(self.env_num)]

        # for self.kv_cache_wm
        self.shared_pool_size_wm = int(self.env_num)
        self.shared_pool_wm = [None] * self.shared_pool_size_wm
        self.shared_pool_index_wm = 0

        self.reanalyze_phase = False

    #@profile
    def custom_copy_kv_cache_to_shared_init_envs(self, src_kv: KeysValues, env_id) -> int:
        """
        Overview:
            Efficiently copy the contents of a KeysValues object to the shared pool.
        Arguments:
            - src_kv (:obj:`KeysValues`): The source KeysValues object to copy from.
        Returns:
            - index (:obj:`int`): The index of the copied KeysValues object in the shared pool.
        """
        src_kv_shape = src_kv._keys_values[0]._k_cache._cache.shape
        
        if self.shared_pool_init_infer[env_id][self.shared_pool_index_init_envs[env_id]] is None:
            self.shared_pool_init_infer[env_id][self.shared_pool_index_init_envs[env_id]] = KeysValues(
                src_kv_shape[0],  # n
                src_kv_shape[1],  # num_heads
                src_kv_shape[2],  # max_tokens
                src_kv_shape[3] * src_kv_shape[1],  # embed_dim
                len(src_kv),  # num_layers
                src_kv._keys_values[0]._k_cache._cache.device,  # device
            )
        
        dst_kv = self.shared_pool_init_infer[env_id][self.shared_pool_index_init_envs[env_id]]
        
        for src_layer, dst_layer in zip(src_kv._keys_values, dst_kv._keys_values):
            # Copy the key and value caches using torch.copy_()
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        index = self.shared_pool_index_init_envs[env_id]
        self.shared_pool_index_init_envs[env_id] = (self.shared_pool_index_init_envs[env_id] + 1) % self.shared_pool_size_init
        
        return index

    #@profile
    def custom_copy_kv_cache_to_shared_wm(self, src_kv: KeysValues) -> int:
        """
        Overview:
            Efficiently copy the contents of a KeysValues object to the shared pool.
        Arguments:
            - src_kv (:obj:`KeysValues`): The source KeysValues object to copy from.
        Returns:
            - index (:obj:`int`): The index of the copied KeysValues object in the shared pool.
        """
        src_kv_shape = src_kv._keys_values[0]._k_cache._cache.shape
        
        if self.shared_pool_wm[self.shared_pool_index_wm] is None:
            self.shared_pool_wm[self.shared_pool_index_wm] = KeysValues(
                src_kv_shape[0],  # n
                src_kv_shape[1],  # num_heads
                src_kv_shape[2],  # max_tokens
                src_kv_shape[3] * src_kv_shape[1],  # embed_dim
                len(src_kv),  # num_layers
                src_kv._keys_values[0]._k_cache._cache.device,  # device
            )
        
        dst_kv = self.shared_pool_wm[self.shared_pool_index_wm]
        
        for src_layer, dst_layer in zip(src_kv._keys_values, dst_kv._keys_values):
            # Copy the key and value caches using torch.copy_()
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        self.shared_pool_index_wm = (self.shared_pool_index_wm + 1) % self.shared_pool_size_wm
        
        return dst_kv

    #@profile
    def custom_copy_kv_cache_to_shared_recur(self, src_kv: KeysValues) -> int:
        """
        Overview:
            Efficiently copy the contents of a KeysValues object to the shared pool.
        Arguments:
            - src_kv (:obj:`KeysValues`): The source KeysValues object to copy from.
        Returns:
            - index (:obj:`int`): The index of the copied KeysValues object in the shared pool.
        """
        src_kv_shape = src_kv._keys_values[0]._k_cache._cache.shape
        
        if self.shared_pool_recur_infer[self.shared_pool_index] is None:
            self.shared_pool_recur_infer[self.shared_pool_index] = KeysValues(
                src_kv_shape[0],  # n
                src_kv_shape[1],  # num_heads
                src_kv_shape[2],  # max_tokens
                src_kv_shape[3] * src_kv_shape[1],  # embed_dim
                len(src_kv),  # num_layers
                src_kv._keys_values[0]._k_cache._cache.device,  # device
            )
        
        dst_kv = self.shared_pool_recur_infer[self.shared_pool_index]
        
        for src_layer, dst_layer in zip(src_kv._keys_values, dst_kv._keys_values):
            # Copy the key and value caches using torch.copy_()
            dst_layer._k_cache._cache.copy_(src_layer._k_cache._cache)
            dst_layer._v_cache._cache.copy_(src_layer._v_cache._cache)
            dst_layer._k_cache._size = src_layer._k_cache._size
            dst_layer._v_cache._size = src_layer._v_cache._size
        
        index = self.shared_pool_index
        self.shared_pool_index = (self.shared_pool_index + 1) % self.shared_pool_size
        
        return index


    def _initialize_config_parameters(self) -> None:
        """Initialize configuration parameters."""
        self.policy_entropy_weight = self.config.policy_entropy_weight
        self.predict_latent_loss_type = self.config.predict_latent_loss_type
        self.group_size = self.config.group_size
        self.num_groups = self.config.embed_dim // self.group_size
        self.obs_type = self.config.obs_type
        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.gamma = self.config.gamma
        self.context_length = self.config.context_length
        self.dormant_threshold = self.config.dormant_threshold
        self.analysis_dormant_ratio = self.config.analysis_dormant_ratio
        self.num_observations_tokens = self.config.tokens_per_block - 1
        self.latent_recon_loss_weight = self.config.latent_recon_loss_weight
        self.perceptual_loss_weight = self.config.perceptual_loss_weight
        self.support_size = self.config.support_size
        self.action_space_size = self.config.action_space_size
        self.max_cache_size = self.config.max_cache_size
        self.env_num = self.config.env_num
        self.num_layers = self.config.num_layers
        self.obs_per_embdding_dim = self.config.embed_dim
        self.sim_norm = SimNorm(simnorm_dim=self.group_size)

    def _initialize_patterns(self) -> None:
        """Initialize patterns for block masks."""
        self.all_but_last_latent_state_pattern = torch.ones(self.config.tokens_per_block)
        self.all_but_last_latent_state_pattern[-2] = 0
        self.act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.act_tokens_pattern[-1] = 1
        self.value_policy_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        self.value_policy_tokens_pattern[-2] = 1

    def _create_head(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        modules = [
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(self.config.embed_dim, output_dim)
        ]
        if norm_layer:
            modules.append(norm_layer)
        return Head(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=nn.Sequential(*modules)
        )

    def _create_head_cont(self, block_mask: torch.Tensor, output_dim: int, norm_layer=None) -> Head:
        """Create head modules for the transformer."""
        from ding.model.common import ReparameterizationHead
        self.fc_policy_head = ReparameterizationHead(
            input_size=self.config.embed_dim,
            output_size=output_dim,
            layer_num=2,  # TODO: check the effect of layer_num
            sigma_type=self.sigma_type,
            activation=nn.GELU(approximate='tanh'),
            fixed_sigma_value=self.fixed_sigma_value,
            norm_type=None,
            bound_type=self.bound_type
        )
        return PolicyHeadCont(
            max_blocks=self.config.max_blocks,
            block_mask=block_mask,
            head_module=self.fc_policy_head
        )

    def _initialize_last_layer(self) -> None:
        """Initialize the last linear layer."""
        last_linear_layer_init_zero = True  # TODO
        if last_linear_layer_init_zero:
            if self.continuous_action_space:
                module_to_initialize = [self.head_value, self.head_rewards, self.head_observations]
            else:
                module_to_initialize = [self.head_policy, self.head_value, self.head_rewards, self.head_observations]
            for head in module_to_initialize:
                for layer in reversed(head.head_module):
                    if isinstance(layer, nn.Linear):
                        nn.init.zeros_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
                        break

    def _initialize_cache_structures(self) -> None:
        """Initialize cache structures for past keys and values."""
        # self.past_kv_cache_init_infer = collections.OrderedDict()
        # self.past_kv_cache_recurrent_infer = collections.OrderedDict()
        # self.past_kv_cache_init_infer_envs = [collections.OrderedDict() for _ in range(self.env_num)]
        # TODO: check
        from collections import defaultdict
        self.past_kv_cache_recurrent_infer = defaultdict(dict)
        self.past_kv_cache_init_infer_envs = [defaultdict(dict) for _ in range(self.env_num)]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []

    def _initialize_projection_input_dim(self) -> None:
        """Initialize the projection input dimension based on the number of observation tokens."""
        if self.num_observations_tokens == 16:
            self.projection_input_dim = 128
        elif self.num_observations_tokens == 1:
            self.projection_input_dim = self.obs_per_embdding_dim

    def _initialize_statistics(self) -> None:
        """Initialize counters for hit count and query count statistics."""
        self.hit_count = 0
        self.total_query_count = 0
        self.length_largethan_maxminus5_context_cnt = 0
        self.length_largethan_maxminus7_context_cnt = 0
        self.root_hit_cnt = 0
        self.root_total_query_cnt = 0

    def _initialize_transformer_keys_values(self) -> None:
        """Initialize keys and values for the transformer."""
        self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1,
                                                                                     max_tokens=self.context_length)
        self.keys_values_wm_single_env_tmp = self.transformer.generate_empty_keys_values(n=1,
                                                                                     max_tokens=self.context_length)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=self.env_num,
                                                                          max_tokens=self.context_length)

    def precompute_pos_emb_diff_kv(self):
        """ Precompute positional embedding differences for key and value. """
        if self.context_length <= 2:
            # If context length is 2 or less, no context is present
            return

        # Precompute positional embedding matrices for inference in collect/eval stages, not for training
        self.positional_embedding_k = [
            self._get_positional_embedding(layer, 'key')
            for layer in range(self.config.num_layers)
        ]
        self.positional_embedding_v = [
            self._get_positional_embedding(layer, 'value')
            for layer in range(self.config.num_layers)
        ]

        # Precompute all possible positional embedding differences
        self.pos_emb_diff_k = []
        self.pos_emb_diff_v = []

        for layer in range(self.config.num_layers):
            layer_pos_emb_diff_k = {}
            layer_pos_emb_diff_v = {}

            for start in [2]:
                for end in [self.context_length - 1]:
                    original_pos_emb_k = self.positional_embedding_k[layer][:, :, start:end, :]
                    new_pos_emb_k = self.positional_embedding_k[layer][:, :, :end - start, :]
                    layer_pos_emb_diff_k[(start, end)] = new_pos_emb_k - original_pos_emb_k

                    original_pos_emb_v = self.positional_embedding_v[layer][:, :, start:end, :]
                    new_pos_emb_v = self.positional_embedding_v[layer][:, :, :end - start, :]
                    layer_pos_emb_diff_v[(start, end)] = new_pos_emb_v - original_pos_emb_v

            self.pos_emb_diff_k.append(layer_pos_emb_diff_k)
            self.pos_emb_diff_v.append(layer_pos_emb_diff_v)

    #@profile
    def _get_positional_embedding(self, layer, attn_type) -> torch.Tensor:
        """
         Helper function to get positional embedding for a given layer and attention type.

         Arguments:
         - layer (:obj:`int`): Layer index.
         - attn_type (:obj:`str`): Attention type, either 'key' or 'value'.

         Returns:
         - torch.Tensor: The positional embedding tensor.
         """
        attn_func = getattr(self.transformer.blocks[layer].attn, attn_type)
        if torch.cuda.is_available():
            return attn_func(self.pos_emb.weight).view(
                1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
            ).transpose(1, 2).to(self.device).detach()
        else:
            return attn_func(self.pos_emb.weight).view(
                1, self.config.max_tokens, self.num_heads, self.embed_dim // self.num_heads
            ).transpose(1, 2).detach()

    #@profile
    def forward(self, obs_embeddings_or_act_tokens: Dict[str, Union[torch.Tensor, tuple]],
                past_keys_values: Optional[torch.Tensor] = None,
                kvcache_independent: bool = False, is_init_infer: bool = True,
                valid_context_lengths: Optional[torch.Tensor] = None) -> WorldModelOutput:
        """
        Forward pass for the model.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing observation embeddings or action tokens.
            - past_keys_values (:obj:`Optional[torch.Tensor]`): Previous keys and values for transformer.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - is_init_infer (:obj:`bool`): Initialize inference.
            - valid_context_lengths (:obj:`Optional[torch.Tensor]`): Valid context lengths.
        Returns:
            - WorldModelOutput: Model output containing logits for observations, rewards, policy, and value.
        """
        # Determine previous steps based on key-value caching method
        if kvcache_independent:
            prev_steps = torch.tensor([0 if past_keys_values is None else past_kv.size for past_kv in past_keys_values],
                                      device=self.device)
        else:
            prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # Reset valid_context_lengths during initial inference
        if is_init_infer:
            valid_context_lengths = None

        # Process observation embeddings
        if 'obs_embeddings' in obs_embeddings_or_act_tokens:
            obs_embeddings = obs_embeddings_or_act_tokens['obs_embeddings']
            if len(obs_embeddings.shape) == 2:
                obs_embeddings = obs_embeddings.unsqueeze(1)
            num_steps = obs_embeddings.size(1)
            sequences = self._add_position_embeddings(obs_embeddings, prev_steps, num_steps, kvcache_independent,
                                                      is_init_infer, valid_context_lengths)

        # Process action tokens
        elif 'act_tokens' in obs_embeddings_or_act_tokens:
            act_tokens = obs_embeddings_or_act_tokens['act_tokens']
            if self.continuous_action_space:
                num_steps = 1
                act_tokens = act_tokens.float()
                if len(act_tokens.shape) == 2:
                    act_tokens = act_tokens.unsqueeze(1)
            else:
                if len(act_tokens.shape) == 3:
                    act_tokens = act_tokens.squeeze(1)
                num_steps = act_tokens.size(1)
            act_embeddings = self.act_embedding_table(act_tokens)
            sequences = self._add_position_embeddings(act_embeddings, prev_steps, num_steps, kvcache_independent,
                                                      is_init_infer, valid_context_lengths)

        # Process combined observation embeddings and action tokens
        else:
            if self.continuous_action_space:
                sequences, num_steps = self._process_obs_act_combined_cont(obs_embeddings_or_act_tokens, prev_steps)
            else:
                sequences, num_steps = self._process_obs_act_combined(obs_embeddings_or_act_tokens, prev_steps)

        # Pass sequences through transformer
        x = self._transformer_pass(sequences, past_keys_values, kvcache_independent, valid_context_lengths)

        # Generate logits
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        # logits_ends is None
        return WorldModelOutput(x, logits_observations, logits_rewards, None, logits_policy, logits_value)

    #@profile
    def _add_position_embeddings(self, embeddings, prev_steps, num_steps, kvcache_independent, is_init_infer,
                                 valid_context_lengths):
        """
        Add position embeddings to the input embeddings.

        Arguments:
            - embeddings (:obj:`torch.Tensor`): Input embeddings.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
            - num_steps (:obj:`int`): Number of steps.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - is_init_infer (:obj:`bool`): Initialize inference.
            - valid_context_lengths (:obj:`torch.Tensor`): Valid context lengths.
        Returns:
            - torch.Tensor: Embeddings with position information added.
        """
        if kvcache_independent:
            steps_indices = prev_steps + torch.arange(num_steps, device=embeddings.device)
            position_embeddings = self.pos_emb(steps_indices).view(-1, num_steps, embeddings.shape[-1])
            return embeddings + position_embeddings
        else:
            if is_init_infer:
                return embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=self.device))
            else:
                valid_context_lengths = torch.tensor(self.keys_values_wm_size_list_current, device=self.device)
                position_embeddings = self.pos_emb(
                    valid_context_lengths + torch.arange(num_steps, device=self.device)).unsqueeze(1)
                return embeddings + position_embeddings

    #@profile
    def _process_obs_act_combined_cont(self, obs_embeddings_or_act_tokens, prev_steps):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        if self.continuous_action_space:
            act_tokens = act_tokens.float()
            if len(act_tokens.shape) == 2:  # TODO
                act_tokens = act_tokens.unsqueeze(-1)

        # B, L, E
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        # B, L*2, E
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        return obs_act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=self.device)), num_steps

    #@profile
    def _process_obs_act_combined(self, obs_embeddings_or_act_tokens, prev_steps):
        """
        Process combined observation embeddings and action tokens.

        Arguments:
            - obs_embeddings_or_act_tokens (:obj:`dict`): Dictionary containing combined observation embeddings and action tokens.
            - prev_steps (:obj:`torch.Tensor`): Previous steps.
        Returns:
            - torch.Tensor: Combined observation and action embeddings with position information added.
        """
        obs_embeddings, act_tokens = obs_embeddings_or_act_tokens['obs_embeddings_and_act_tokens']
        if len(obs_embeddings.shape) == 3:
            obs_embeddings = obs_embeddings.view(act_tokens.shape[0], act_tokens.shape[1], self.num_observations_tokens,
                                                 -1)

        num_steps = int(obs_embeddings.size(1) * (obs_embeddings.size(2) + 1))
        act_embeddings = self.act_embedding_table(act_tokens)

        B, L, K, E = obs_embeddings.size()
        obs_act_embeddings = torch.empty(B, L * (K + 1), E, device=self.device)

        for i in range(L):
            obs = obs_embeddings[:, i, :, :]
            act = act_embeddings[:, i, 0, :].unsqueeze(1)
            obs_act = torch.cat([obs, act], dim=1)
            obs_act_embeddings[:, i * (K + 1):(i + 1) * (K + 1), :] = obs_act

        return obs_act_embeddings + self.pos_emb(prev_steps + torch.arange(num_steps, device=self.device)), num_steps

    #@profile
    def _transformer_pass(self, sequences, past_keys_values, kvcache_independent, valid_context_lengths):
        """
        Pass sequences through the transformer.

        Arguments:
            - sequences (:obj:`torch.Tensor`): Input sequences.
            - past_keys_values (:obj:`Optional[torch.Tensor]`): Previous keys and values for transformer.
            - kvcache_independent (:obj:`bool`): Whether to use independent key-value caching.
            - valid_context_lengths (:obj:`torch.Tensor`): Valid context lengths.
        Returns:
            - torch.Tensor: Transformer output.
        """
        if kvcache_independent:
            x = [self.transformer(sequences[k].unsqueeze(0), past_kv,
                                  valid_context_lengths=valid_context_lengths[k].unsqueeze(0)) for k, past_kv in
                 enumerate(past_keys_values)]
            return torch.cat(x, dim=0)
        else:
            return self.transformer(sequences, past_keys_values, valid_context_lengths=valid_context_lengths)

    #@profile
    @torch.no_grad()
    def reset_for_initial_inference(self, obs_act_dict: torch.FloatTensor) -> torch.FloatTensor:
        """
        Reset the model state based on initial observations and actions.

        Arguments:
            - obs_act_dict (:obj:`torch.FloatTensor`): A dictionary containing 'obs', 'action', and 'current_obs'.
        Returns:
            - torch.FloatTensor: The outputs from the world model and the latent state.
        """
        # Extract observations, actions, and current observations from the dictionary.
        if isinstance(obs_act_dict, dict):
            batch_obs = obs_act_dict['obs']  # obs_act_dict['obs'] is at timestep t
            batch_action = obs_act_dict['action'] # obs_act_dict['action'] is at timestep t
            batch_current_obs = obs_act_dict['current_obs'] # obs_act_dict['current_obs'] is at timestep t+1

        # Encode observations to latent embeddings.
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_obs)

        if batch_current_obs is not None:
            # ================ Collect and Evaluation Phase ================
            # Encode current observations to latent embeddings
            current_obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch_current_obs)
            # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
            self.latent_state = current_obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action,
                                                                                   current_obs_embeddings)
        else:
            # ================ calculate the target value in Train phase ================
            self.latent_state = obs_embeddings
            outputs_wm = self.wm_forward_for_initial_infererence(obs_embeddings, batch_action, None)

        return outputs_wm, self.latent_state

    #@profile
    @torch.no_grad()
    def wm_forward_for_initial_infererence(self, last_obs_embeddings: torch.LongTensor,
                                                             batch_action=None,
                                                             current_obs_embeddings=None) -> torch.FloatTensor:
        """
        Refresh key-value pairs with the initial latent state for inference.

        Arguments:
            - last_obs_embeddings (:obj:`torch.LongTensor`): The latent state embeddings.
            - batch_action (optional): Actions taken.
            - current_obs_embeddings (optional): Current observation embeddings.
        Returns:
            - torch.FloatTensor: The outputs from the world model.
        """
        n, num_observations_tokens, _ = last_obs_embeddings.shape
        if n <= self.env_num and current_obs_embeddings is not None:
            # ================ Collect and Evaluation Phase ================
            if current_obs_embeddings is not None:
                if self.continuous_action_space:
                    first_step_flag = not isinstance(batch_action[0], np.ndarray)
                else:
                    first_step_flag = max(batch_action) == -1
                if first_step_flag:
                    # First step in an episode
                    self.keys_values_wm = self.transformer.generate_empty_keys_values(n=current_obs_embeddings.shape[0],
                                                                                      max_tokens=self.context_length)
                    # print(f"current_obs_embeddings.device: {current_obs_embeddings.device}")
                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)
                else:
                    # current_obs_embeddings is the new latent_state, containing information from ready_env_num environments
                    ready_env_num = current_obs_embeddings.shape[0]
                    self.keys_values_wm_list = []
                    self.keys_values_wm_size_list = []
                    for i in range(ready_env_num):
                        # Retrieve latent state for a single environment
                        # TODO: len(last_obs_embeddings) may smaller than len(current_obs_embeddings), because some environments may have done

                        state_single_env = last_obs_embeddings[i]
                        # Compute hash value using latent state for a single environment
                        cache_key = hash_state(state_single_env.view(-1).cpu().numpy())  # last_obs_embeddings[i] is torch.Tensor

                        # Retrieve cached value
                        cache_index = self.past_kv_cache_init_infer_envs[i].get(cache_key)
                        if cache_index is not None:
                            matched_value = self.shared_pool_init_infer[i][cache_index]
                        else:
                            matched_value = None

                        self.root_total_query_cnt += 1
                        if matched_value is not None:
                            # If a matching value is found, add it to the list
                            self.root_hit_cnt += 1
                            # NOTE: deepcopy is needed because forward modifies matched_value in place
                            self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                            self.keys_values_wm_size_list.append(matched_value.size)
                        else:
                            # Reset using zero values
                            self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(n=1,
                                                                                                         max_tokens=self.context_length)
                            outputs_wm = self.forward({'obs_embeddings': state_single_env.unsqueeze(0)},
                                                      past_keys_values=self.keys_values_wm_single_env,
                                                      is_init_infer=True)
                            self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                            self.keys_values_wm_size_list.append(1)

                    # Input self.keys_values_wm_list, output self.keys_values_wm
                    self.keys_values_wm_size_list_current = self.trim_and_pad_kv_cache(is_init_infer=True)

                    batch_action = batch_action[:ready_env_num]
                    # # only for debug
                    # if ready_env_num < self.env_num:
                    #     print(f'init inference ready_env_num: {ready_env_num} < env_num: {self.env_num}')
                    if self.continuous_action_space:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(1)
                    else:
                        act_tokens = torch.from_numpy(np.array(batch_action)).to(last_obs_embeddings.device).unsqueeze(-1)
                    outputs_wm = self.forward({'act_tokens': act_tokens}, past_keys_values=self.keys_values_wm,
                                              is_init_infer=True)

                    outputs_wm = self.forward({'obs_embeddings': current_obs_embeddings},
                                              past_keys_values=self.keys_values_wm, is_init_infer=True)

                    # Copy and store keys_values_wm for a single environment
                    self.update_cache_context(current_obs_embeddings, is_init_infer=True)

        elif batch_action is not None and current_obs_embeddings is None:
            # ================ calculate the target value in Train phase ================
            # [192, 16, 64] -> [32, 6, 16, 64]
            last_obs_embeddings = last_obs_embeddings.contiguous().view(batch_action.shape[0], -1, num_observations_tokens,
                                                          self.obs_per_embdding_dim)  # (BL, K) for unroll_step=1

            last_obs_embeddings = last_obs_embeddings[:, :-1, :]
            batch_action = torch.from_numpy(batch_action).to(last_obs_embeddings.device)
            if self.continuous_action_space:
                act_tokens = batch_action
            else:
                act_tokens = rearrange(batch_action, 'b l -> b l 1')

            # select the last timestep for each sample
            # This will select the last column while keeping the dimensions unchanged, and the target policy/value in the final step itself is not used.
            last_steps_act = act_tokens[:, -1:, :]
            act_tokens = torch.cat((act_tokens, last_steps_act), dim=1)

            outputs_wm = self.forward({'obs_embeddings_and_act_tokens': (last_obs_embeddings, act_tokens)})

            # select the last timestep for each sample
            last_steps_value = outputs_wm.logits_value[:, -1:, :]
            outputs_wm.logits_value = torch.cat((outputs_wm.logits_value, last_steps_value), dim=1)

            last_steps_policy = outputs_wm.logits_policy[:, -1:, :]
            outputs_wm.logits_policy = torch.cat((outputs_wm.logits_policy, last_steps_policy), dim=1)

            # Reshape your tensors
            # outputs_wm.logits_value.shape (B, H, 101) = (B*H, 101)
            outputs_wm.logits_value = rearrange(outputs_wm.logits_value, 'b t e -> (b t) e')
            outputs_wm.logits_policy = rearrange(outputs_wm.logits_policy, 'b t e -> (b t) e')

        return outputs_wm

    #@profile
    @torch.no_grad()
    def forward_initial_inference(self, obs_act_dict):
        """
        Perform initial inference based on the given observation-action dictionary.

        Arguments:
            - obs_act_dict (:obj:`dict`): Dictionary containing observations and actions.
        Returns:
            - tuple: A tuple containing output sequence, latent state, logits rewards, logits policy, and logits value.
        """
        # UniZero has context in the root node
        outputs_wm, latent_state = self.reset_for_initial_inference(obs_act_dict)
        self.past_kv_cache_recurrent_infer.clear()

        return (outputs_wm.output_sequence, latent_state, outputs_wm.logits_rewards,
                outputs_wm.logits_policy, outputs_wm.logits_value)

    #@profile
    @torch.no_grad()
    def forward_recurrent_inference(self, state_action_history, simulation_index=0,
                                    latent_state_index_in_search_path=[]):
        """
        Perform recurrent inference based on the state-action history.

        Arguments:
            - state_action_history (:obj:`list`): List containing tuples of state and action history.
            - simulation_index (:obj:`int`, optional): Index of the current simulation. Defaults to 0.
            - latent_state_index_in_search_path (:obj:`list`, optional): List containing indices of latent states in the search path. Defaults to [].
        Returns:
            - tuple: A tuple containing output sequence, updated latent state, reward, logits policy, and logits value.
        """
        latest_state, action = state_action_history[-1]
        ready_env_num = latest_state.shape[0]

        self.keys_values_wm_list = []
        self.keys_values_wm_size_list = []
        self.keys_values_wm_size_list = self.retrieve_or_generate_kvcache(latest_state, ready_env_num, simulation_index)

        latent_state_list = []
        if not self.continuous_action_space:
            token = action.reshape(-1, 1)
        else:
            token = action.reshape(-1, self.action_space_size)

        # ======= Print statistics for debugging =============
        # min_size = min(self.keys_values_wm_size_list)
        # if min_size >= self.config.max_tokens - 5:
        #     self.length_largethan_maxminus5_context_cnt += len(self.keys_values_wm_size_list)
        # if min_size >= self.config.max_tokens - 7:
        #     self.length_largethan_maxminus7_context_cnt += len(self.keys_values_wm_size_list)
        # if self.total_query_count > 0 and self.total_query_count % 10000 == 0:
        #     self.hit_freq = self.hit_count / self.total_query_count
        #     print('total_query_count:', self.total_query_count)
        #     length_largethan_maxminus5_context_cnt_ratio = self.length_largethan_maxminus5_context_cnt / self.total_query_count
        #     print('recurrent largethan_maxminus5_context:', self.length_largethan_maxminus5_context_cnt)
        #     print('recurrent largethan_maxminus5_context_ratio:', length_largethan_maxminus5_context_cnt_ratio)
        #     length_largethan_maxminus7_context_cnt_ratio = self.length_largethan_maxminus7_context_cnt / self.total_query_count
        #     print('recurrent largethan_maxminus7_context_ratio:', length_largethan_maxminus7_context_cnt_ratio)
        #     print('recurrent largethan_maxminus7_context:', self.length_largethan_maxminus7_context_cnt)

        # Trim and pad kv_cache: modify self.keys_values_wm in-place
        self.keys_values_wm_size_list = self.trim_and_pad_kv_cache(is_init_infer=False)
        self.keys_values_wm_size_list_current = self.keys_values_wm_size_list

        for k in range(2):
            # action_token obs_token
            if k == 0:
                obs_embeddings_or_act_tokens = {'act_tokens': token}
            else:
                obs_embeddings_or_act_tokens = {'obs_embeddings': token}

            # Perform forward pass
            outputs_wm = self.forward(
                obs_embeddings_or_act_tokens,
                past_keys_values=self.keys_values_wm,
                kvcache_independent=False,
                is_init_infer=False
            )

            self.keys_values_wm_size_list_current = [i + 1 for i in self.keys_values_wm_size_list_current]

            if k == 0:
                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = outputs_wm.logits_observations
                if len(token.shape) != 3:
                    token = token.unsqueeze(1)  # (8,1024) -> (8,1,1024)
                latent_state_list.append(token)

        del self.latent_state  # Very important to minimize cuda memory usage
        self.latent_state = torch.cat(latent_state_list, dim=1)  # (B, K)

        self.update_cache_context(
            self.latent_state,
            is_init_infer=False,
            simulation_index=simulation_index,
            latent_state_index_in_search_path=latent_state_index_in_search_path
        )

        return (outputs_wm.output_sequence, self.latent_state, reward, outputs_wm.logits_policy, outputs_wm.logits_value)


    #@profile
    def trim_and_pad_kv_cache(self, is_init_infer=True) -> list:
        """
        Adjusts the key-value cache for each environment to ensure they all have the same size.

        In a multi-environment setting, the key-value cache (kv_cache) for each environment is stored separately.
        During recurrent inference, the kv_cache sizes may vary across environments. This method pads each kv_cache
        to match the largest size found among them, facilitating batch processing in the transformer forward pass.

        Arguments:
            - is_init_infer (:obj:`bool`): Indicates if this is an initial inference. Default is True.
        Returns:
            - list: Updated sizes of the key-value caches.
        """
        # Find the maximum size among all key-value caches
        max_size = max(self.keys_values_wm_size_list)

        # Iterate over each layer of the transformer
        for layer in range(self.num_layers):
            kv_cache_k_list = []
            kv_cache_v_list = []

            # Enumerate through each environment's key-value pairs
            for idx, keys_values in enumerate(self.keys_values_wm_list):
                k_cache = keys_values[layer]._k_cache._cache
                v_cache = keys_values[layer]._v_cache._cache

                effective_size = self.keys_values_wm_size_list[idx]
                pad_size = max_size - effective_size

                # If padding is required, trim the end and pad the beginning of the cache
                if pad_size > 0:
                    k_cache_trimmed = k_cache[:, :, :-pad_size, :]
                    v_cache_trimmed = v_cache[:, :, :-pad_size, :]
                    k_cache_padded = F.pad(k_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                    v_cache_padded = F.pad(v_cache_trimmed, (0, 0, pad_size, 0), "constant", 0)
                else:
                    k_cache_padded = k_cache
                    v_cache_padded = v_cache

                kv_cache_k_list.append(k_cache_padded)
                kv_cache_v_list.append(v_cache_padded)

            # Stack the caches along a new dimension and remove any extra dimensions
            self.keys_values_wm._keys_values[layer]._k_cache._cache = torch.stack(kv_cache_k_list, dim=0).squeeze(1)
            self.keys_values_wm._keys_values[layer]._v_cache._cache = torch.stack(kv_cache_v_list, dim=0).squeeze(1)

            # Update the cache size to the maximum size
            self.keys_values_wm._keys_values[layer]._k_cache._size = max_size
            self.keys_values_wm._keys_values[layer]._v_cache._size = max_size

        return self.keys_values_wm_size_list

    #@profile
    def update_cache_context(self, latent_state, is_init_infer=True, simulation_index=0,
                             latent_state_index_in_search_path=[], valid_context_lengths=None):
        """
        Update the cache context with the given latent state.

        Arguments:
            - latent_state (:obj:`torch.Tensor`): The latent state tensor.
            - is_init_infer (:obj:`bool`): Flag to indicate if this is the initial inference.
            - simulation_index (:obj:`int`): Index of the simulation.
            - latent_state_index_in_search_path (:obj:`list`): List of indices in the search path.
            - valid_context_lengths (:obj:`list`): List of valid context lengths.
        """
        if self.context_length <= 2:
            # No context to update if the context length is less than or equal to 2.
            return
        for i in range(latent_state.size(0)):
            # ============ Iterate over each environment ============
            cache_key = hash_state(latent_state[i].view(-1).cpu().numpy())  # latent_state[i] is torch.Tensor
            context_length = self.context_length

            if not is_init_infer:
                # ============ Internal Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding
                current_max_context_length = max(self.keys_values_wm_size_list_current)
                trim_size = current_max_context_length - self.keys_values_wm_size_list_current[i]
                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============
                    # cache shape [batch_size, num_heads, sequence_length, features]
                    k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                    v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                    if trim_size > 0:
                        # Trim invalid leading zeros as per effective length
                        # Remove the first trim_size zero kv items
                        k_cache_trimmed = k_cache_current[:, trim_size:, :]
                        v_cache_trimmed = v_cache_current[:, trim_size:, :]
                        # If effective length < current_max_context_length, pad the end of cache with 'trim_size' zeros
                        k_cache_padded = F.pad(k_cache_trimmed, (0, 0, 0, trim_size), "constant",
                                               0)  # Pad with 'trim_size' zeros at end of cache
                        v_cache_padded = F.pad(v_cache_trimmed, (0, 0, 0, trim_size), "constant", 0)
                    else:
                        k_cache_padded = k_cache_current
                        v_cache_padded = v_cache_current

                    # Update cache of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                    # Update size of self.keys_values_wm_single_env
                    self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm_size_list_current[i]
                    self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm_size_list_current[i]

                    # ============ NOTE: Very Important ============
                    if self.keys_values_wm_single_env._keys_values[layer]._k_cache._size >= context_length - 1:
                        # Keep only the last self.context_length-3 timesteps of context
                        # For memory environments, training is for H steps, recurrent_inference might exceed H steps
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache
                        v_cache_current = self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, :, 2:context_length - 1, :].squeeze(0)
                        v_cache_trimmed = v_cache_current[:, :, 2:context_length - 1, :].squeeze(0)

                        # Index pre-computed positional encoding differences
                        pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                        pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                        # ============ NOTE: Very Important ============
                        # Apply positional encoding correction to k and v
                        k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                        v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update single environment cache
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)

                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            else:
                # ============ Root Node ============
                # Retrieve KV from global KV cache self.keys_values_wm to single environment KV cache self.keys_values_wm_single_env, ensuring correct positional encoding

                for layer in range(self.num_layers):
                    # ============ Apply trimming and padding to each layer of kv_cache ============

                    if self.keys_values_wm._keys_values[layer]._k_cache._size < context_length - 1:  # Keep only the last self.context_length-1 timesteps of context
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = \
                        self.keys_values_wm._keys_values[layer]._k_cache._cache[i].unsqueeze(
                            0)  # Shape torch.Size([2, 100, 512])
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = \
                        self.keys_values_wm._keys_values[layer]._v_cache._cache[i].unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = \
                        self.keys_values_wm._keys_values[layer]._k_cache._size
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = \
                        self.keys_values_wm._keys_values[layer]._v_cache._size
                    else:
                        # Assuming cache dimension is [batch_size, num_heads, sequence_length, features]
                        k_cache_current = self.keys_values_wm._keys_values[layer]._k_cache._cache[i]
                        v_cache_current = self.keys_values_wm._keys_values[layer]._v_cache._cache[i]

                        # Remove the first 2 steps, keep the last self.context_length-3 steps
                        k_cache_trimmed = k_cache_current[:, 2:context_length - 1, :]
                        v_cache_trimmed = v_cache_current[:, 2:context_length - 1, :]

                        # Index pre-computed positional encoding differences
                        pos_emb_diff_k = self.pos_emb_diff_k[layer][(2, context_length - 1)]
                        pos_emb_diff_v = self.pos_emb_diff_v[layer][(2, context_length - 1)]
                        # ============ NOTE: Very Important ============
                        # Apply positional encoding correction to k and v
                        k_cache_trimmed += pos_emb_diff_k.squeeze(0)
                        v_cache_trimmed += pos_emb_diff_v.squeeze(0)

                        # Pad the last 3 steps along the third dimension with zeros
                        # F.pad parameters (0, 0, 0, 3) specify padding amounts for each dimension: (left, right, top, bottom). For 3D tensor, they correspond to (dim2 left, dim2 right, dim1 left, dim1 right).
                        padding_size = (0, 0, 0, 3)
                        k_cache_padded = F.pad(k_cache_trimmed, padding_size, 'constant', 0)
                        v_cache_padded = F.pad(v_cache_trimmed, padding_size, 'constant', 0)
                        # Update cache of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._cache = k_cache_padded.unsqueeze(0)
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._cache = v_cache_padded.unsqueeze(0)
                        # Update size of self.keys_values_wm_single_env
                        self.keys_values_wm_single_env._keys_values[layer]._k_cache._size = context_length - 3
                        self.keys_values_wm_single_env._keys_values[layer]._v_cache._size = context_length - 3

            if is_init_infer:
                # Store the latest key-value cache for initial inference
                cache_index = self.custom_copy_kv_cache_to_shared_init_envs(self.keys_values_wm_single_env, i)
                self.past_kv_cache_init_infer_envs[i][cache_key] = cache_index
            else:
                # Store the latest key-value cache for recurrent inference
                cache_index = self.custom_copy_kv_cache_to_shared_recur(self.keys_values_wm_single_env)
                self.past_kv_cache_recurrent_infer[cache_key] = cache_index


    #@profile
    def retrieve_or_generate_kvcache(self, latent_state: list, ready_env_num: int,
                                     simulation_index: int = 0) -> list:
        """
        Retrieves or generates key-value caches for each environment based on the latent state.

        For each environment, this method either retrieves a matching cache from the predefined
        caches if available, or generates a new cache if no match is found. The method updates
        the internal lists with these caches and their sizes.

        Arguments:
            - latent_state (:obj:`list`): List of latent states for each environment.
            - ready_env_num (:obj:`int`): Number of environments ready for processing.
            - simulation_index (:obj:`int`, optional): Index for simulation tracking. Default is 0.
        Returns:
            - list: Sizes of the key-value caches for each environment.
        """
        for i in range(ready_env_num):
            self.total_query_count += 1
            state_single_env = latent_state[i]  # latent_state[i] is np.array
            cache_key = hash_state(state_single_env)

            if self.reanalyze_phase:
                # TODO: check if this is correct
                matched_value = None
            else:
                # Try to retrieve the cached value from past_kv_cache_init_infer_envs
                cache_index = self.past_kv_cache_init_infer_envs[i].get(cache_key)
                if cache_index is not None:
                    matched_value = self.shared_pool_init_infer[i][cache_index]
                else:
                    matched_value = None

                # If not found, try to retrieve from past_kv_cache_recurrent_infer
                if matched_value is None:
                    matched_value = self.shared_pool_recur_infer[self.past_kv_cache_recurrent_infer.get(cache_key)]

            if matched_value is not None:
                # If a matching cache is found, add it to the lists
                self.hit_count += 1
                # Perform a deep copy because the transformer's forward pass might modify matched_value in-place
                self.keys_values_wm_list.append(self.custom_copy_kv_cache_to_shared_wm(matched_value))
                self.keys_values_wm_size_list.append(matched_value.size)
            else:
                # If no matching cache is found, generate a new one using zero reset
                self.keys_values_wm_single_env = self.transformer.generate_empty_keys_values(
                    n=1, max_tokens=self.context_length
                )
                self.forward(
                    {'obs_embeddings': torch.from_numpy(state_single_env).unsqueeze(0).to(self.device)},
                    past_keys_values=self.keys_values_wm_single_env, is_init_infer=True
                )
                self.keys_values_wm_list.append(self.keys_values_wm_single_env)
                self.keys_values_wm_size_list.append(1)

        return self.keys_values_wm_size_list


    #@profile
    def compute_loss(self, batch, target_tokenizer: Tokenizer = None, inverse_scalar_transform_handle=None,
                     **kwargs: Any) -> LossWithIntermediateLosses:
        # Encode observations into latent state representations
        obs_embeddings = self.tokenizer.encode_to_obs_embeddings(batch['observations'])

        # ========= for visual analysis =========
        # Uncomment the lines below for visual analysis in Pong
        # self.plot_latent_tsne_each_and_all_for_pong(obs_embeddings, suffix='pong_H10_H4_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='pong_H10_H4_tsne')
        # Uncomment the lines below for visual analysis in visual match
        # self.plot_latent_tsne_each_and_all(obs_embeddings, suffix='visual_match_memlen1-60-15_tsne')
        # self.save_as_image_with_timestep(batch['observations'], suffix='visual_match_memlen1-60-15_tsne')

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # Calculate dormant ratio of the encoder
            shape = batch['observations'].shape  # (..., C, H, W)
            inputs = batch['observations'].contiguous().view(-1, *shape[-3:])  # (32,5,3,64,64) -> (160,3,64,64)
            dormant_ratio_encoder = cal_dormant_ratio(self.tokenizer.representation_network, inputs.detach(),
                                                      percentage=self.dormant_threshold)
            # self.past_kv_cache_init_infer.clear()
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_encoder = torch.tensor(0.)

        # Calculate the L2 norm of the latent state roots
        latent_state_l2_norms = torch.norm(obs_embeddings, p=2, dim=2).mean()

        if self.obs_type == 'image':
            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # original_images, reconstructed_images = batch['observations'], reconstructed_images
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # Calculate reconstruction loss and perceptual loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            # perceptual_loss = self.tokenizer.perceptual_loss(batch['observations'].reshape(-1, 3, 64, 64), reconstructed_images) # NOTE: for stack=1
            
            latent_recon_loss = self.latent_recon_loss
            perceptual_loss = self.perceptual_loss

        elif self.obs_type == 'vector':
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

            # Reconstruct observations from latent state representations
            # reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings.reshape(-1, self.embed_dim))
            # # Calculate reconstruction loss
            # latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 25),
            #                                                        reconstructed_images)
            latent_recon_loss = torch.tensor(0., device=batch['observations'].device,
                                             dtype=batch['observations'].dtype)

        elif self.obs_type == 'image_memory':
            # Reconstruct observations from latent state representations
            reconstructed_images = self.tokenizer.decode_to_obs(obs_embeddings)
            original_images, reconstructed_images = batch['observations'], reconstructed_images

            #  ========== for visualization ==========
            # Uncomment the lines below for visual analysis
            # target_policy = batch['target_policy']
            # target_predict_value = inverse_scalar_transform_handle(batch['target_value'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            # true_rewards = inverse_scalar_transform_handle(batch['rewards'].reshape(-1, 101)).reshape(
            #     batch['observations'].shape[0], batch['observations'].shape[1], 1)
            #  ========== for visualization ==========

            # Calculate reconstruction loss and perceptual loss
            latent_recon_loss = self.tokenizer.reconstruction_loss(batch['observations'].reshape(-1, 3, 5, 5),
                                                                   reconstructed_images)
            perceptual_loss = torch.tensor(0., device=batch['observations'].device,
                                           dtype=batch['observations'].dtype)

        # Action tokens
        if self.continuous_action_space:
            act_tokens = batch['actions']
        else:
            act_tokens = rearrange(batch['actions'], 'b l -> b l 1')

        # Forward pass to obtain predictions for observations, rewards, and policies
        outputs = self.forward({'obs_embeddings_and_act_tokens': (obs_embeddings, act_tokens)})

        # ========= logging for analysis =========
        if self.analysis_dormant_ratio:
            # Calculate dormant ratio of the world model
            dormant_ratio_world_model = cal_dormant_ratio(self, {
                'obs_embeddings_and_act_tokens': (obs_embeddings.detach(), act_tokens.detach())},
                                                          percentage=self.dormant_threshold)
            # self.past_kv_cache_init_infer.clear()
            self.past_kv_cache_recurrent_infer.clear()
            self.keys_values_wm_list.clear()
            torch.cuda.empty_cache()
        else:
            dormant_ratio_world_model = torch.tensor(0.)

        #  ========== for visualization ==========
        # Uncomment the lines below for visualization
        # predict_policy = outputs.logits_policy
        # predict_policy = F.softmax(outputs.logits_policy, dim=-1)
        # predict_value = inverse_scalar_transform_handle(outputs.logits_value.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # predict_rewards = inverse_scalar_transform_handle(outputs.logits_rewards.reshape(-1, 101)).reshape(batch['observations'].shape[0], batch['observations'].shape[1], 1)
        # import pdb; pdb.set_trace()
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=[], suffix='pong_H10_H4_0613')

        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_success_episode')
        # visualize_reward_value_img_policy(original_images, reconstructed_images, target_predict_value, true_rewards, target_policy, predict_value, predict_rewards, predict_policy, not_plot_timesteps=list(np.arange(4,60)), suffix='visual_match_memlen1-60-15/one_fail_episode')
        #  ========== for visualization ==========

        # For training stability, use target_tokenizer to compute the true next latent state representations
        with torch.no_grad():
            target_obs_embeddings = target_tokenizer.encode_to_obs_embeddings(batch['observations'])

        # Compute labels for observations, rewards, and ends
        labels_observations, labels_rewards, _ = self.compute_labels_world_model(target_obs_embeddings,
                                                                                           batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        # Reshape the logits and labels for observations
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        labels_observations = labels_observations.reshape(-1, self.projection_input_dim)

        # Compute prediction loss for observations. Options: MSE and Group KL
        if self.predict_latent_loss_type == 'mse':
            # MSE loss, directly compare logits and labels
            loss_obs = torch.nn.functional.mse_loss(logits_observations, labels_observations, reduction='none').mean(
                -1)
        elif self.predict_latent_loss_type == 'group_kl':
            # Group KL loss, group features and calculate KL divergence within each group
            batch_size, num_features = logits_observations.shape
            epsilon = 1e-6
            logits_reshaped = logits_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon
            labels_reshaped = labels_observations.reshape(batch_size, self.num_groups, self.group_size) + epsilon

            loss_obs = F.kl_div(logits_reshaped.log(), labels_reshaped, reduction='none').sum(dim=-1).mean(dim=-1)

            #  ========== for debugging ==========
            # print('loss_obs:', loss_obs.mean())
            # assert not torch.isnan(loss_obs).any(), "loss_obs contains NaN values"
            # assert not torch.isinf(loss_obs).any(), "loss_obs contains Inf values"
            # for name, param in self.tokenizer.encoder.named_parameters():
            #     print('name, param.mean(), param.std():', name, param.mean(), param.std())

        # Apply mask to loss_obs
        mask_padding_expanded = batch['mask_padding'][:, 1:].contiguous().view(-1)
        loss_obs = (loss_obs * mask_padding_expanded)

        # Compute labels for policy and value
        labels_policy, labels_value = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                   batch['target_policy'],
                                                                                   batch['mask_padding'])

        # Compute losses for rewards, policy, and value
        loss_rewards = self.compute_cross_entropy_loss(outputs, labels_rewards, batch, element='rewards')

        if not self.continuous_action_space:
            loss_policy, orig_policy_loss, policy_entropy = self.compute_cross_entropy_loss(outputs, labels_policy,
                                                                                            batch,
                                                                                            element='policy')
        else:
            # NOTE: for continuous action space
            if self.config.policy_loss_type == 'simple':
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont_simple(outputs, batch)
            else:
                orig_policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma = self._calculate_policy_loss_cont(outputs, batch)
            
            loss_policy = orig_policy_loss + self.policy_entropy_weight * policy_entropy_loss
            policy_entropy = - policy_entropy_loss

        loss_value = self.compute_cross_entropy_loss(outputs, labels_value, batch, element='value')

        # ==== TODO: calculate the new priorities for each transition. ====
        # value_priority = L1Loss(reduction='none')(labels_value.squeeze(-1), outputs['logits_value'][:, 0])
        # value_priority = value_priority.data.cpu().numpy() + 1e-6

        # Compute timesteps
        timesteps = torch.arange(batch['actions'].shape[1], device=batch['actions'].device)
        # Compute discount coefficients for each timestep
        discounts = self.gamma ** timesteps

        if batch['mask_padding'].sum() == 0:
            assert False, "mask_padding is all zeros"

        # Group losses into first step, middle step, and last step
        first_step_losses = {}
        middle_step_losses = {}
        last_step_losses = {}
        # batch['mask_padding'] indicates mask status for future H steps, exclude masked losses to maintain accurate mean statistics
        # Group losses for each loss item
        for loss_name, loss_tmp in zip(
                ['loss_obs', 'loss_rewards', 'loss_value', 'loss_policy', 'orig_policy_loss', 'policy_entropy'],
                [loss_obs, loss_rewards, loss_value, loss_policy, orig_policy_loss, policy_entropy]
        ):
            if loss_name == 'loss_obs':
                seq_len = batch['actions'].shape[1] - 1
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, 1:seq_len]
            else:
                seq_len = batch['actions'].shape[1]
                # Get the corresponding mask_padding
                mask_padding = batch['mask_padding'][:, :seq_len]

            # Adjust loss shape to (batch_size, seq_len)
            loss_tmp = loss_tmp.view(-1, seq_len)

            # First step loss
            first_step_mask = mask_padding[:, 0]
            first_step_losses[loss_name] = loss_tmp[:, 0][first_step_mask].mean()

            # Middle step loss
            middle_step_index = seq_len // 2
            middle_step_mask = mask_padding[:, middle_step_index]
            middle_step_losses[loss_name] = loss_tmp[:, middle_step_index][middle_step_mask].mean()

            # Last step loss
            last_step_mask = mask_padding[:, -1]
            last_step_losses[loss_name] = loss_tmp[:, -1][last_step_mask].mean()

        # Discount reconstruction loss and perceptual loss
        discounted_latent_recon_loss = latent_recon_loss
        discounted_perceptual_loss = perceptual_loss

        # Calculate overall discounted loss
        discounted_loss_obs = (loss_obs.view(-1, batch['actions'].shape[1] - 1) * discounts[1:]).sum()/ batch['mask_padding'][:,1:].sum()
        discounted_loss_rewards = (loss_rewards.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_value = (loss_value.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_loss_policy = (loss_policy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_orig_policy_loss = (orig_policy_loss.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()
        discounted_policy_entropy = (policy_entropy.view(-1, batch['actions'].shape[1]) * discounts).sum()/ batch['mask_padding'].sum()

        if self.continuous_action_space:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                loss_obs=discounted_loss_obs,
                loss_rewards=discounted_loss_rewards,
                loss_value=discounted_loss_value,
                loss_policy=discounted_loss_policy,
                latent_recon_loss=discounted_latent_recon_loss,
                perceptual_loss=discounted_perceptual_loss,
                orig_policy_loss=discounted_orig_policy_loss,
                policy_entropy=discounted_policy_entropy,
                first_step_losses=first_step_losses,
                middle_step_losses=middle_step_losses,
                last_step_losses=last_step_losses,
                dormant_ratio_encoder=dormant_ratio_encoder,
                dormant_ratio_world_model=dormant_ratio_world_model,
                latent_state_l2_norms=latent_state_l2_norms,
                policy_mu=mu,
                policy_sigma=sigma,
                target_sampled_actions=target_sampled_actions,
            )
        else:
            return LossWithIntermediateLosses(
                latent_recon_loss_weight=self.latent_recon_loss_weight,
                perceptual_loss_weight=self.perceptual_loss_weight,
                loss_obs=discounted_loss_obs,
                loss_rewards=discounted_loss_rewards,
                loss_value=discounted_loss_value,
                loss_policy=discounted_loss_policy,
                latent_recon_loss=discounted_latent_recon_loss,
                perceptual_loss=discounted_perceptual_loss,
                orig_policy_loss=discounted_orig_policy_loss,
                policy_entropy=discounted_policy_entropy,
                first_step_losses=first_step_losses,
                middle_step_losses=middle_step_losses,
                last_step_losses=last_step_losses,
                dormant_ratio_encoder=dormant_ratio_encoder,
                dormant_ratio_world_model=dormant_ratio_world_model,
                latent_state_l2_norms=latent_state_l2_norms,
            )

    # TODO: test correctness
    def _calculate_policy_loss_cont_simple(self, outputs, batch: dict):
        """
        Simplified policy loss calculation for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.

        Returns:
            - policy_loss (:obj:`torch.Tensor`): The simplified policy loss.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        # Get the policy logits and batch data
        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding'].contiguous().view(-1)
        target_policy = batch['target_policy'].contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = batch['child_sampled_actions'].contiguous().view(batch_size * num_unroll_steps, -1, action_space_size)

        # Flatten for vectorized computation
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        
        # Extract mean and standard deviation from logits
        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        dist = Independent(Normal(mu, sigma), 1)  # Create the normal distribution

        # Find the indices of the maximum values in the target policy
        target_best_action_idx = torch.argmax(target_policy, dim=1)

        # Select the best actions based on the indices
        target_best_action = target_sampled_actions[torch.arange(target_best_action_idx.size(0)), target_best_action_idx]

        # Clip the target actions to prevent numerical issues during arctanh
        target_best_action_clamped = torch.clamp(target_best_action, -1 + 1e-6, 1 - 1e-6)
        target_best_action_before_tanh = torch.arctanh(target_best_action_clamped)

        # Calculate the log probability of the best action
        log_prob_best_action = dist.log_prob(target_best_action_before_tanh)

        # Mask the log probability with the padding mask
        log_prob_best_action = log_prob_best_action * mask_batch

        # Return the negative log probability as the policy loss (we want to maximize log_prob)
        # policy_loss = -log_prob_best_action.mean()
        policy_loss = -log_prob_best_action

        policy_entropy = dist.entropy().mean()
        policy_entropy_loss = -policy_entropy * mask_batch
        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def _calculate_policy_loss_cont(self, outputs, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate the policy loss for continuous actions.

        Args:
            - outputs: Model outputs containing policy logits.
            - batch (:obj:`dict`): Batch data containing target policy, mask and sampled actions.
        Returns:
            - policy_loss (:obj:`torch.Tensor`): The calculated policy loss.
            - policy_entropy_loss (:obj:`torch.Tensor`): The entropy loss of the policy.
            - target_policy_entropy (:obj:`float`): The entropy of the target policy distribution.
            - target_sampled_actions (:obj:`torch.Tensor`): The actions sampled from the target policy.
            - mu (:obj:`torch.Tensor`): The mean of the normal distribution.
            - sigma (:obj:`torch.Tensor`): The standard deviation of the normal distribution.
        """
        batch_size, num_unroll_steps, action_space_size = outputs.logits_policy.shape[
            0], self.config.num_unroll_steps, self.config.action_space_size

        policy_logits_all = outputs.logits_policy
        mask_batch = batch['mask_padding']
        child_sampled_actions_batch = batch['child_sampled_actions']
        target_policy = batch['target_policy']

        # Flatten the unroll step dimension for easier vectorized operations
        policy_logits_all = policy_logits_all.view(batch_size * num_unroll_steps, -1)
        mask_batch = mask_batch.contiguous().view(-1)
        child_sampled_actions_batch = child_sampled_actions_batch.contiguous().view(batch_size * num_unroll_steps, -1,
                                                                                    action_space_size)

        mu, sigma = policy_logits_all[:, :action_space_size], policy_logits_all[:, action_space_size:]
        mu = mu.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        sigma = sigma.unsqueeze(1).expand(-1, child_sampled_actions_batch.shape[1], -1)
        dist = Independent(Normal(mu, sigma), 1)

        target_normalized_visit_count = target_policy.contiguous().view(batch_size * num_unroll_steps, -1)
        target_sampled_actions = child_sampled_actions_batch

        policy_entropy = dist.entropy().mean(dim=1)
        policy_entropy_loss = -policy_entropy * mask_batch

        # y = 1 - target_sampled_actions.pow(2)
        # target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -1 + 1e-6, 1 - 1e-6)
        # target_sampled_actions_before_tanh = torch.arctanh(target_sampled_actions_clamped)
        # log_prob = dist.log_prob(target_sampled_actions_before_tanh)
        # log_prob = log_prob - torch.log(y + 1e-6).sum(-1)
        # log_prob_sampled_actions = log_prob

        base_dist = Normal(mu, sigma)
        tanh_transform = TanhTransform()
        dist = TransformedDistribution(base_dist, [tanh_transform])
        dist = Independent(dist, 1)
        target_sampled_actions_clamped = torch.clamp(target_sampled_actions, -1 + 1e-6, 1 - 1e-6)
        # assert torch.all(target_sampled_actions_clamped < 1) and torch.all(target_sampled_actions_clamped > -1), "Actions are not properly clamped."
        log_prob = dist.log_prob(target_sampled_actions_clamped)
        log_prob_sampled_actions = log_prob

        # KL as projector
        # log_prob_sampled_actions = torch.clamp(log_prob_sampled_actions, max=0.0, min=-10.0) # TODO: clip the log_prob_sampled_actions to prevent numerical issues
        target_log_prob_sampled_actions = torch.log(target_normalized_visit_count + 1e-6)
        policy_loss = -torch.sum(
            torch.exp(target_log_prob_sampled_actions.detach()) * log_prob_sampled_actions, 1
        ) * mask_batch

        # Calculate the entropy of the target policy distribution
        non_masked_indices = torch.nonzero(mask_batch).squeeze(-1)
        if len(non_masked_indices) > 0:
            target_dist = Categorical(target_normalized_visit_count[non_masked_indices])
            target_policy_entropy = target_dist.entropy().mean().item()
        else:
            target_policy_entropy = 0.0

        return policy_loss, policy_entropy_loss, target_policy_entropy, target_sampled_actions, mu, sigma

    def compute_cross_entropy_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs is an object with logits attributes like 'rewards', 'policy', and 'value'.
        # labels is a target tensor for comparison. batch is a dictionary with a mask indicating valid timesteps.

        logits = getattr(outputs, f'logits_{element}')

        if torch.isnan(logits).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")
        
        if torch.isnan(labels).any():
            raise ValueError(f"NaN detected in labels_value for batch {batch} and element '{element}'")

        # Reshape your tensors
        logits = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[-1])  # Assume labels initially have shape [batch, time, dim]

        # Reshape your mask. True indicates valid data.
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)')

        # Compute cross-entropy loss
        loss = -(torch.log_softmax(logits, dim=1) * labels).sum(1)
        loss = (loss * mask_padding)

        if torch.isnan(loss).any():
            raise ValueError(f"NaN detected in outputs for batch {batch} and element '{element}'")

        if element == 'policy':
            # Compute policy entropy loss
            policy_entropy = self.compute_policy_entropy_loss(logits, mask_padding)
            # Combine losses with specified weight
            combined_loss = loss - self.policy_entropy_weight * policy_entropy
            return combined_loss, loss, policy_entropy

        return loss

    #@profile
    def compute_policy_entropy_loss(self, logits, mask):
        # Compute entropy of the policy
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(1)
        # Apply mask and return average entropy loss
        entropy_loss = (entropy * mask)
        return entropy_loss

    #@profile
    def compute_labels_world_model(self, obs_embeddings: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # Each sequence sample should have at most one 'done' flag
        mask_fill = torch.logical_not(mask_padding)

        # Prepare observation labels
        labels_observations = obs_embeddings.contiguous().view(rewards.shape[0], -1, self.projection_input_dim)[:, 1:]

        # Fill the masked areas of rewards
        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        # Fill the masked areas of ends
        # labels_endgs = ends.masked_fill(mask_fill, -100)

        # return labels_observations, labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)
        return labels_observations, labels_rewards.view(-1, self.support_size), None


    #@profile
    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute labels for value and policy predictions. """
        mask_fill = torch.logical_not(mask_padding)

        # Fill the masked areas of policy
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        # Fill the masked areas of value
        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)

        if self.continuous_action_space:
            return None, labels_value.reshape(-1, self.support_size)
        else:
            return labels_policy.reshape(-1, self.action_space_size), labels_value.reshape(-1, self.support_size)

    def clear_caches(self):
        """
        Clears the caches of the world model.
        """
        # self.past_kv_cache_init_infer.clear()
        for kv_cache_dict_env in self.past_kv_cache_init_infer_envs:
            kv_cache_dict_env.clear()
        self.past_kv_cache_recurrent_infer.clear()
        self.keys_values_wm_list.clear()
        print(f'Cleared {self.__class__.__name__} past_kv_cache.')

    def __repr__(self) -> str:
        return "transformer-based latent world_model of UniZero"
