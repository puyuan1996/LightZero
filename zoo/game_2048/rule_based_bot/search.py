import copy


class AI:
    def __init__(self, grid, active=True):
        self.grid = grid
        self.active = active


directions = ["UP", "LEFT", "DOWN", "RIGHT"]

expect_map = {2: 0.9, 4: 0.1}

model1 = [[16, 15, 14, 13],
          [9, 10, 11, 12],
          [8, 7, 6, 5],
          [1, 2, 3, 4]]

model2 = [[16, 15, 12, 4],
          [14, 13, 11, 3],
          [10, 9, 8, 2],
          [7, 6, 5, 1]]

model3 = [[16, 15, 14, 4],
          [13, 12, 11, 3],
          [10, 9, 8, 2],
          [7, 6, 5, 1]]


def search(ai):
    best_dire = None
    best_score = -1
    depth = dept_select(ai)
    for dire in directions:
        new_grid = copy.deepcopy(ai.grid)
        if new_grid.move(dire):
            new_ai = AI(new_grid, active=False)
            new_score = expect_search(new_ai, depth)
            if new_score > best_score:
                best_dire = dire
                best_score = new_score
    return best_dire


def expect_search(ai, depth):
    if depth == 0:
        return float(score(ai))
    score_sum = 0
    if ai.active:
        for d in directions:
            new_grid = copy.deepcopy(ai.grid)
            if new_grid.move(d):
                new_ai = AI(new_grid, active=False)
                score_sum += expect_search(new_ai, depth - 1)
    else:
        points = ai.grid.vacant_points()
        for k, v in expect_map.items():
            for point in points:
                new_grid = copy.deepcopy(ai.grid)
                new_grid.data[point[0]][point[1]] = k
                new_ai = AI(new_grid, active=True)
                score_sum += expect_search(new_ai, depth - 1) * v
        score_sum /= float(len(points))
    return score_sum


def score(a):
    result = [0] * 24
    for x in range(4):
        for y in range(4):
            value = a.grid.data[x][y]
            if value != 0:
                model_score(0, x, y, value, model1, result)
                model_score(1, x, y, value, model2, result)
                model_score(2, x, y, value, model3, result)

    max_score = max(result)
    return max_score


def model_score(index, x, y, value, model, result):
    start = index * 8

    result[start] += value * model[x][y]
    result[start + 1] += value * model[x][3 - y]

    result[start + 2] += value * model[y][x]
    result[start + 3] += value * model[3 - y][x]

    result[start + 4] += value * model[3 - x][3 - y]
    result[start + 5] += value * model[3 - x][y]

    result[start + 6] += value * model[y][3 - x]
    result[start + 7] += value * model[3 - y][3 - x]


def dept_select(a):
    dept = 4
    max_value = a.grid.max()
    if max_value >= 2048:
        dept = 6
    elif max_value >= 1024:
        dept = 5

    return dept
