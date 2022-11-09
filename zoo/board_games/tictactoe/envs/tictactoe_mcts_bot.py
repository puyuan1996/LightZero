import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod
import time
import sys
from easydict import EasyDict
sys.path.append('/Users/yangzhenjie/code/jayyoung0802/LightZero')
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv


class MCTSNode(ABC):
    def __init__(self, state, parent=None):
        """
        Overview:
            Monte Carlo Tree Search Base Node
            https://github.com/int8/monte-carlo-tree-search
        Parameters:
            state: zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv
            parent: MCTSSearchNode
        """
        self.state = state
        self.parent = parent
        self.children = []
        self.parent_action = []
        self.best_action = -1

    @property
    @abstractmethod
    def untried_actions(self):
        """
        Returns:
            list of zoo.board_games.tictactoe.envs.tictactoe_env.TicTacToeEnv.legal_actions
        """
        pass

    @property
    @abstractmethod
    def q(self):
        pass

    @property
    @abstractmethod
    def n(self):
        pass

    @abstractmethod
    def expand(self):
        pass

    @abstractmethod
    def is_terminal_node(self):
        pass

    @abstractmethod
    def rollout(self):
        pass

    @abstractmethod
    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        self.best_action = self.parent_action[np.argmax(choices_weights)]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_actions):        
        return possible_actions[np.random.randint(len(possible_actions))]


class TwoPlayersMCTSNode(MCTSNode):

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self._untried_actions = None

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.legal_actions
        return self._untried_actions

    @property
    def q(self):
        print(self._results)
        print('parent.current_player={}'.format(self.parent.state.current_player))
        if self.parent.state.current_player==1:
            wins = self._results[1]
            loses = self._results[-1]
        
        if self.parent.state.current_player==2:
            wins = self._results[-1]
            loses = self._results[1]
        print("wins={}, loses={}".format(wins, loses))
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_simulator_env = self.state.simulate_action(action)
        child_node = TwoPlayersMCTSNode(
            next_simulator_env, parent=self
        )
        self.children.append(child_node)
        self.parent_action.append(action)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()[0]

    def rollout(self):
        print('simulation begin')
        current_rollout_state = self.state
        print(current_rollout_state.board)
        while not current_rollout_state.is_game_over()[0]:
            possible_actions = current_rollout_state.legal_actions
            action = self.rollout_policy(possible_actions)
            current_rollout_state = current_rollout_state.simulate_action(action)
            print('\n')
            print(current_rollout_state.board)
        print('simulation end \n')
        return current_rollout_state.is_game_over()[1]

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

class MCTSSearchNode(object):

    def __init__(self, node):
        """
        Overview:
            Monte Carlo Tree Search Node
        Arguments:
            node : TwoPlayersMCTSNode
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """
        Overview:

        Arguments:
            simulations_number : int
                number of simulations performed to get the best action
            total_simulation_seconds : float
                Amount of time the algorithm has to run. Specified in seconds
        Returns:
            Returns the best children node, and can get action from Node.best_action_index.
        -------
        """

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else :
            for i in range(0, simulations_number):
                print('****simlulation-{}****'.format(i))            
                v = self._tree_policy()
                reward = v.rollout()
                print('reward={}\n'.format(reward))
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        Overview:
            selects node to run rollout
        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node



class MCTSBot: 
    def __init__(self, name, num_simulation=10000):
        self.name = name
        self.num_simulation = num_simulation
    
    def get_actions(self, state, player_index, env_cfg):
        simulator_env = TicTacToeEnv(EasyDict(env_cfg))
        simulator_env.reset(start_player_index=player_index, init_state=state)
        legal_actions = simulator_env.legal_actions
        root = TwoPlayersMCTSNode(simulator_env)
        mcts = MCTSSearchNode(root)
        mcts.best_action(self.num_simulation)
        return root.best_action