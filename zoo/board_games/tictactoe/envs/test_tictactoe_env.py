import pytest
from easydict import EasyDict
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv


@pytest.mark.envtest
class TestTicTacToeEnv:

    def test_two_player_mode(self):
        cfg = EasyDict(battle_mode='two_player_mode', prob_random_agent=0, prob_expert_agent=0)
        env = TicTacToeEnv(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            # action = env.human_to_action()
            # action = env.random_action()
            action = env.expert_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 1 (human player) win')
                else:
                    print('draw')
                break
            """player 2"""
            action = env.expert_action()
            print('player 2 (computer player): ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                if reward > 0:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

    def test_one_player_mode(self):
        cfg = EasyDict(battle_mode='one_player_mode', prob_random_agent=0, prob_expert_agent=0)
        env = TicTacToeEnv(cfg)
        env.reset()
        print('init board state: ')
        env.render()
        while True:
            """player 1"""
            # action = env.human_to_action()
            action = env.random_action()
            print('player 1: ' + env.action_to_string(action))
            obs, reward, done, info = env.step(action)
            # reward is in the perspective of player1
            env.render()
            if done:
                if reward != 0 and info['next player to play'] == 2:
                    print('player 1 (human player) win')
                elif reward != 0 and info['next player to play'] == 1:
                    print('player 2 (computer player) win')
                else:
                    print('draw')
                break

test = TestTicTacToeEnv()
test.test_one_player_mode()
