from ding.utils import EasyTimer
from legal_actions import legal_actions
# from zoo.board_games.gomoku.envs.legal_actions import legal_actions
import numpy as np

timer = EasyTimer(cuda=True)

board = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
board_size = 3


def legal_actions_raw(board, board_size):
    legal_actions = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                legal_actions.append(i * board_size + j)
    return legal_actions


def legal_actions_np(board, board_size):
    zero_positions = np.argwhere(board == 0)
    legal_actions = [i * board_size + j for i, j in zero_positions]
    return legal_actions


def eval_raw_time(board, board_size, times=1000000):
    with timer:
        for i in range(times):
            legal_moves = legal_actions_raw(board, board_size)
    print(f"eval_raw_time: {timer.value}")


def eval_np_time(board, board_size, times=1000000):
    with timer:
        for i in range(times):
            legal_moves = legal_actions_np(board, board_size)
    print(f"eval_np_time: {timer.value}")


def eval_cython_time(board, board_size, times=1000000):
    with timer:
        for i in range(times):
            legal_moves = legal_actions(board, board_size)
    print(f"eval_cython_time: {timer.value}")


if __name__ == "__main__":
    print("###execute legal_actions 1000,000 times###")
    eval_raw_time(board, board_size, times=1000000)
    eval_np_time(board, board_size, times=1000000)
    eval_cython_time(board, board_size, times=1000000)
