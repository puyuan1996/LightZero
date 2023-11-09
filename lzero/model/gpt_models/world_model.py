import copy
from dataclasses import dataclass
from typing import Any, Tuple
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from einops import rearrange
from torch.distributions.categorical import Categorical

from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from .utils import init_weights, LossWithIntermediateLosses
from joblib import hash

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor

    logits_policy: torch.FloatTensor
    logits_value: torch.FloatTensor


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config

        self.transformer = Transformer(config)
        self.num_observations_tokens = 16
        # self.device = 'cpu'
        self.device = config.device
        self.support_size = config.support_size

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        value_policy_tokens_pattern = torch.zeros(config.tokens_per_block)
        value_policy_tokens_pattern[-2] = 1

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList(
                [nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        # self.head_rewards = Head(
        #     max_blocks=config.max_blocks,
        #     block_mask=act_tokens_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.embed_dim, config.embed_dim),
        #         nn.ReLU(),
        #         nn.Linear(config.embed_dim, 3)
        #     )
        # )
        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, self.support_size)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )
        self.head_observations_for_inference = Head(
            max_blocks=config.max_blocks,
            block_mask=obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )
        self.head_policy = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,  # TODO: value_policy_tokens_pattern
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)  # TODO(pu); action shape
            )
        )
        self.head_value = Head(
            max_blocks=config.max_blocks,
            block_mask=value_policy_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, self.support_size)  # TODO(pu): action shape
            )
        )

        self.apply(init_weights)

        # last_linear_layer_init_zero=True is beneficial for convergence speed.
        last_linear_layer_init_zero = True  # TODO
        if last_linear_layer_init_zero:
            # Locate the last linear layer and initialize its weights and biases to 0.
            for _, layer in enumerate(reversed(self.head_policy.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break
            for _, layer in enumerate(reversed(self.head_value.head_module)):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break

        self.past_keys_values_cache = {}

    def __repr__(self) -> str:
        return "world_model"

    def forward_recurrent(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None,
                          inference=True) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        # if prev_steps > 0:
        #     print('prev_steps > 0')

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values)

        logits_observations = self.head_observations_for_inference(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        # return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_policy, logits_value)

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None,
                inference=False) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size
        # if prev_steps > 0:
        #     print('prev_steps > 0')

        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(
            prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values)

        if inference:
            logits_observations = self.head_observations_for_inference(x, num_steps=num_steps, prev_steps=prev_steps)
        else:
            logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)
        # return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

        logits_policy = self.head_policy(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_value = self.head_value(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends, logits_policy, logits_value)

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)  # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)  # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(
            0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens  # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        outputs_wm = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return outputs_wm, self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        # self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        self.keys_values_wm = self.transformer.generate_empty_keys_values(n=n, max_tokens=self.config.max_tokens)
        outputs_wm = self.forward(obs_tokens, past_keys_values=self.keys_values_wm,
                                  inference=True)  # TODO: inference=False

        # return outputs_wm.output_sequence  # (B, K, E)
        return outputs_wm

    def forward_initial_inference(self, obs: torch.LongTensor, should_predict_next_obs: bool = True):

        if len(obs[0].shape) == 3:
            # obs is a 3-dimensional image
            pass
        elif len(obs[0].shape) == 1:
            # TODO()
            # obs is a 1-dimensional vector
            original_shape = list(obs.shape)
            desired_shape = original_shape + [64, 64]
            expanded_observations = obs.unsqueeze(-1).unsqueeze(-1)
            expanded_observations = expanded_observations.expand(*desired_shape)
            obs = expanded_observations

        outputs_wm, _ = self.reset_from_initial_observations(obs)

        return outputs_wm.output_sequence, outputs_wm.logits_observations, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value

    def forward_recurrent_inference(self, state_action_history, should_predict_next_obs: bool = True):
        # 已知 state_action_history[0] 是 (root_latent_state, root_action) 并且state_action_history[-1]是最后一步的latent state和action
        # 而且中间可能有很多个(root_latent_state, *） 由于transformer在计算时应该都是从(root_latent_state, *）开始 unroll的，因此我希望
        # 找到这样的一个(root_latent_state, *）的最后一个位置，然后从这个位置找到其在 past_keys_values_cache 中的对应的 value，然后从这个 value开始
        # 进行unroll，这样就可以保证在进行推断时，不会出现重复计算的情况。

        # 找到 state_action_history 中最后一个 (root_latent_state, *) 的位置
        root_latent_state, _ = state_action_history[0]
        last_root_position = max(
            i for i, (latent_state, _) in enumerate(state_action_history) if np.array_equal(latent_state, root_latent_state))

        # 从这个位置开始的 action_history
        action_history_from_last_root = state_action_history[last_root_position:]

        cache_key = hash(action_history_from_last_root)
        if cache_key in self.past_keys_values_cache:
            # 如果找到对应的缓存，那么直接使用缓存
            self.keys_values_wm = self.past_keys_values_cache[cache_key]
        # 如果没有找到对应的缓存，那么进行正常的计算

        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        # 提取出最后一个action
        action = state_action_history[-1][-1]

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.
            # obs is in token level
            outputs_wm = self.forward(token, past_keys_values=self.keys_values_wm,
                                      inference=False)  # TODO: inference=False
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                # reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(
                    -1)  # (B,)

                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)  # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)  # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        # return outputs_wm.output_sequence, outputs_wm.logits_observations, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value

        # 在计算结束后，更新缓存，注意需要深拷贝
        self.past_keys_values_cache[cache_key] = copy.deepcopy(self.keys_values_wm)

        return outputs_wm.output_sequence, self.obs_tokens, reward, outputs_wm.logits_policy, outputs_wm.logits_value

    def forward_recurrent_inference_single_step(self, action, should_predict_next_obs: bool = True):
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)
        # TODO: test reset
        # _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.
            # obs is in token level
            outputs_wm = self.forward(token, past_keys_values=self.keys_values_wm,
                                      inference=False)  # TODO: inference=False
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                # reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(
                    -1)  # (B,)

                reward = outputs_wm.logits_rewards  # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)  # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)  # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        # return outputs_wm.output_sequence, outputs_wm.logits_observations, outputs_wm.logits_rewards, outputs_wm.logits_policy, outputs_wm.logits_value

        return outputs_wm.output_sequence, self.obs_tokens, reward, outputs_wm.logits_policy, outputs_wm.logits_value

    def compute_loss(self, batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        if len(batch['observations'][0, 0].shape) == 3:
            # obs is a 3-dimensional image
            pass
        elif len(batch['observations'][0, 0].shape) == 1:
            # TODO()
            # obs is a 1-dimensional vector
            original_shape = list(batch['observations'].shape)
            desired_shape = original_shape + [64, 64]
            expanded_observations = batch['observations'].unsqueeze(-1).unsqueeze(-1)
            expanded_observations = expanded_observations.expand(*desired_shape)
            batch['observations'] = expanded_observations

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        # outputs = self(tokens)
        outputs = self.forward(tokens, inference=False)

        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'],
                                                                                           batch['ends'],
                                                                                           batch['mask_padding'])

        """
        >>> # Example of target with class probabilities
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5).softmax(dim=1)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
        """
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        # loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        # return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

        labels_policy, labels_value = self.compute_labels_world_model_value_policy(batch['target_value'],
                                                                                   batch['target_policy'],
                                                                                   batch['mask_padding'])

        # loss_policy = F.cross_entropy(rearrange(outputs.logits_policy, 'b t e -> (b t) e'), labels_policy)
        # loss_value = F.cross_entropy(rearrange(outputs.logits_value, 'b t e -> (b t) e'), labels_value)

        loss_rewards = self.compute_kl_loss(outputs, labels_rewards, batch, element='rewards')
        loss_policy = self.compute_kl_loss(outputs, labels_policy, batch, element='policy')
        loss_value = self.compute_kl_loss(outputs, labels_value, batch, element='value')

        # return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends, loss_value=loss_value, loss_policy=loss_policy)
        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_value=loss_value,
                                          loss_policy=loss_policy)

    def compute_kl_loss(self, outputs, labels, batch, element='rewards'):
        # Assume outputs.logits_rewards and labels are your predictions and targets
        # And mask_padding is a boolean tensor with True at positions to keep and False at positions to ignore

        if element == 'rewards':
            logits = outputs.logits_rewards
        elif element == 'policy':
            logits = outputs.logits_policy
        elif element == 'value':
            logits = outputs.logits_value

        # Reshape your tensors
        logits_rewards = rearrange(logits, 'b t e -> (b t) e')
        labels = labels.reshape(-1, labels.shape[
            -1])  # Assuming labels originally has shape [b, t, reward_dim]

        # Reshape your mask
        mask_padding = rearrange(batch['mask_padding'], 'b t -> (b t)').unsqueeze(-1)

        # Compute the loss
        loss_rewards = F.kl_div(torch.softmax(logits_rewards, dim=-1).log(), labels, reduction='none')

        # Apply the mask
        loss_rewards = loss_rewards.masked_select(mask_padding).mean()
        return loss_rewards

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor,
                                   mask_padding: torch.BoolTensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100),
                                        'b t k -> b (t k)')[:, 1:]

        # labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1} TODO(pu)

        mask_fill_rewards = mask_fill.unsqueeze(-1).expand_as(rewards)
        labels_rewards = rewards.masked_fill(mask_fill_rewards, -100)

        labels_ends = ends.masked_fill(mask_fill, -100)
        # return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
        return labels_observations.reshape(-1), labels_rewards.reshape(-1, self.support_size), labels_ends.reshape(-1)

    def compute_labels_world_model_value_policy(self, target_value: torch.Tensor, target_policy: torch.Tensor,
                                                mask_padding: torch.BoolTensor) -> Tuple[
        torch.Tensor, torch.Tensor]:

        mask_fill = torch.logical_not(mask_padding)
        mask_fill_policy = mask_fill.unsqueeze(-1).expand_as(target_policy)
        labels_policy = target_policy.masked_fill(mask_fill_policy, -100)

        mask_fill_value = mask_fill.unsqueeze(-1).expand_as(target_value)
        labels_value = target_value.masked_fill(mask_fill_value, -100)
        return labels_policy.reshape(-1, 2), labels_value.reshape(-1, self.support_size)  # TODO(pu)
        # return labels_policy.reshape(-1, ), labels_value.reshape(-1)
