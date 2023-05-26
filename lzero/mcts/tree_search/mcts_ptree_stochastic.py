from typing import TYPE_CHECKING, List, Any, Union
from easydict import EasyDict

import copy
import numpy as np
import torch

from lzero.mcts.ptree import MinMaxStatsList
from lzero.policy import InverseScalarTransform
import lzero.mcts.ptree.ptree_stochastic_mz as tree_muzero

if TYPE_CHECKING:
    import lzero.mcts.ptree.ptree_stochastic_mz as stochastic_mz_ptree


# ==============================================================
# Stochastic MuZero
# ==============================================================


class StochasticMuZeroMCTSPtree(object):
    """
    Overview:
        MCTSPtree for MuZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in python.
    Interfaces:
        __init__, search
    """

    # the default_config for MuZeroMCTSPtree.
    config = dict(
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_base=19652,
        # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_init=1.25,
        # (float) The maximum change in value allowed during the backup step of the search tree update.
        value_delta_max=0.01,
    )

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict = None) -> None:
        """
        Overview:
            Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
            in the default configuration, the user-provided value will override the default configuration. Otherwise,
            the default configuration will be used.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.inverse_scalar_transform_handle = InverseScalarTransform(
            self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
        )

    @classmethod
    def roots(cls: int, root_num: int, legal_actions: List[Any]) -> "stochastic_mz_ptree.Roots":
        """
        Overview:
            The initialization of CRoots with root num and legal action lists.
        Arguments:
            - root_num: the number of the current root.
            - legal_action_list: the vector of the legal action of this root.
        """
        import lzero.mcts.ptree.ptree_stochastic_mz as ptree
        return ptree.Roots(root_num, legal_actions)

    def search(
            self,
            roots: Any,
            model: torch.nn.Module,
            latent_state_roots: List[Any],
            to_play: Union[int, List[Any]] = -1
    ) -> None:
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
            Use the python ctree.
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - latent_state_roots (:obj:`list`): the hidden states of the roots
            - to_play (:obj:`list`): the to_play list used in two_player mode board games
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self._cfg.device
            pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor
            # the data storage of hidden states: storing the hidden states of all the ctree root nodes
            # latent_state_roots.shape  (2, 12, 3, 3)
            latent_state_batch_in_search_path = [latent_state_roots]

            # the index of each layer in the ctree
            current_latent_state_index = 0
            # minimax value storage
            min_max_stats_lst = MinMaxStatsList(num)

            for simulation_index in range(self._cfg.num_simulations):
                # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

                latent_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree_muzero.SearchResults(num=num)

                # latent_state_index_in_search_path: The first index of the latent state corresponding to the leaf node in latent_state_batch_in_search_path, that is, the search depth.
                # latent_state_index_in_batch: The second index of the latent state corresponding to the leaf node in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
                # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
                """
                MCTS stage 1: Selection
                    Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
                """
                # leaf_nodes, latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play = tree_muzero.batch_traverse(
                #     roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play)
                # )
                results, virtual_to_play = tree_muzero.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play)
                )
                leaf_nodes, latent_state_index_in_search_path, latent_state_index_in_batch, last_actions = results.nodes, results.latent_state_index_in_search_path, results.latent_state_index_in_batch, results.last_actions

                # obtain the states for leaf nodes
                for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
                    latent_states.append(
                        latent_state_batch_in_search_path[ix][iy])  # latent_state_batch_in_search_path[ix][iy] shape e.g. (64,4,4)

                latent_states = torch.from_numpy(np.asarray(latent_states)).to(device).float()
                # only for discrete action
                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).long()
                """
                   MCTS stage 2: Expansion
                       At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
                       Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
                   MCTS stage 3: Backup
                       At the end of the simulation, the statistics along the trajectory are updated.
                   """
                # network_output = model.recurrent_inference(latent_states, last_actions)

                is_child_chance_batch = [None] * num
                latent_state_batch = [None] * num
                value_batch = [None] * num
                reward_batch = [None] * num
                policy_logits_batch = [None] * num
                for i in range(num):
                    if leaf_nodes[i].is_chance:
                        # If leaf node is chance, then parent is not chance node.
                        # The parent is not a chance node, afterstate to latent state transition.
                        # The last action or outcome is a chance outcome.
                        network_output = model.recurrent_inference(latent_states[i].unsqueeze(0),
                                                                   last_actions[i].unsqueeze(0),
                                                                   afterstate=False)

                        # child_state = network_output.dynamics(parent.state, history.last_action_or_outcome())
                        # network_output = network_output.predictions(child_state)

                        # This child is a decision node.
                        is_child_chance_batch[i] = True

                        if not model.training:
                            # if not in training, obtain the scalars of the value/reward
                            network_output.value = self.inverse_scalar_transform_handle(
                                network_output.value).detach().cpu().numpy()
                            network_output.reward = self.inverse_scalar_transform_handle(
                                network_output.reward).detach().cpu().numpy()
                            network_output.latent_state = network_output.latent_state.detach().cpu().numpy()
                            network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                        latent_state_batch[i] = network_output.latent_state
                        value_batch[i] = network_output.value.reshape(-1).tolist()
                        reward_batch[i] = network_output.reward.reshape(-1).tolist()
                        policy_logits_batch[i] = network_output.policy_logits.tolist()
                    else:
                        # The parent is a decision node, latent state to afterstate transition.
                        # The last action or outcome is an action.

                        network_output = model.recurrent_inference(latent_states[i].unsqueeze(0),
                                                                   last_actions[i].unsqueeze(0),
                                                                   afterstate=True)

                        # child_state = network_output.afterstate_dynamics(parent.state, history.last_action_or_outcome())
                        # network_output = network_output.afterstate_predictions(child_state)

                        # The child is a chance node.
                        is_child_chance_batch[i] = False

                        if not model.training:
                            # if not in training, obtain the scalars of the value/reward
                            network_output.value = self.inverse_scalar_transform_handle(network_output.value
                                                                                        ).detach().cpu().numpy()
                            network_output.reward = self.inverse_scalar_transform_handle(network_output.reward
                                                                                         ).detach().cpu().numpy()
                            network_output.latent_state = network_output.latent_state.detach().cpu().numpy()
                            network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                        latent_state_batch[i] = network_output.latent_state
                        value_batch[i] = network_output.value.reshape(-1).tolist()
                        reward_batch[i] = network_output.reward.reshape(-1).tolist()
                        policy_logits_batch[i] = network_output.policy_logits.tolist()

                # latent_state_batch = torch.cat(latent_state_batch, dim=0)
                # value_batch = torch.cat(value_batch, dim=0)
                # reward_batch = torch.cat(reward_batch, dim=0)
                # policy_logits_batch = torch.cat(policy_logits_batch, dim=0)

                latent_state_batch = np.concatenate(latent_state_batch, axis=0)
                value_batch = np.concatenate(value_batch, axis=0)
                reward_batch = np.concatenate(reward_batch, axis=0)
                policy_logits_batch = np.concatenate(policy_logits_batch, axis=0)

                latent_state_batch_in_search_path.append(latent_state_batch)
                # increase the index of leaf node
                current_latent_state_index += 1

                # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
                # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
                # statistics.

                # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
                current_latent_state_index = simulation_index + 1
                tree_muzero.batch_backpropagate(
                    current_latent_state_index, discount_factor, reward_batch, value_batch, policy_logits_batch,
                    min_max_stats_lst, results, virtual_to_play, is_child_chance_batch
                )
