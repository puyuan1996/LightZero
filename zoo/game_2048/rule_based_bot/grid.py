import copy
from enum import Enum
from typing import List, Tuple


class Direction(Enum):
    UP = "UP"
    RIGHT = "RIGHT"
    DOWN = "DOWN"
    LEFT = "LEFT"
    NONE = "NONE"


def diff(data1: List[List[int]], data2: List[List[int]]) -> bool:
    for i in range(len(data1)):
        for j in range(len(data1[i])):
            if data1[i][j] != data2[i][j]:
                return True
    return False


class Grid:
    def __init__(self, data: List[List[int]] = None):
        if data is None:
            self.data = [[0] * 4 for _ in range(4)]
        else:
            self.data = data

    def clone(self):
        return Grid(copy.deepcopy(self.data))

    def max(self) -> int:
        return max(max(row) for row in self.data)

    def vacant_points(self) -> List[Tuple[int, int]]:
        points = []
        for x, row in enumerate(self.data):
            for y, value in enumerate(row):
                if value == 0:
                    points.append((x, y))
        return points

    def move(self, d: Direction) -> bool:
        origin_data = copy.deepcopy(self.data)
        data = self.data

        if d == 'UP':
            for y in range(4):
                for x in range(3):
                    for nx in range(x + 1, 4):
                        if data[nx][y] > 0:
                            if data[x][y] <= 0:
                                data[x][y] = data[nx][y]
                                data[nx][y] = 0
                                x -= 1
                            elif data[x][y] == data[nx][y]:
                                data[x][y] += data[nx][y]
                                data[nx][y] = 0
                            break
        elif d == 'DOWN':
            for y in range(4):
                for x in range(3, 0, -1):
                    for nx in range(x - 1, -1, -1):
                        if data[nx][y] > 0:
                            if data[x][y] <= 0:
                                data[x][y] = data[nx][y]
                                data[nx][y] = 0
                                x += 1
                            elif data[x][y] == data[nx][y]:
                                data[x][y] += data[nx][y]
                                data[nx][y] = 0
                            break
        elif d == 'LEFT':
            for x in range(4):
                for y in range(3):
                    for ny in range(y + 1, 4):
                        if data[x][ny] > 0:
                            if data[x][y] <= 0:
                                data[x][y] = data[x][ny]
                                data[x][ny] = 0
                                y -= 1
                            elif data[x][y] == data[x][ny]:
                                data[x][y] += data[x][ny]
                                data[x][ny] = 0
                            break
        elif d == 'RIGHT':
            for x in range(4):
                for y in range(3, 0, -1):
                    for ny in range(y - 1, -1, -1):
                        if data[x][ny] > 0:
                            if data[x][y] <= 0:
                                data[x][y] = data[x][ny]
                                data[x][ny] = 0
                                y += 1
                            elif data[x][y] == data[x][ny]:
                                data[x][y] += data[x][ny]
                                data[x][ny] = 0
                            break

        return diff(data, origin_data)
