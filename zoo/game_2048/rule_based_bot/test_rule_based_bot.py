import gym

from grid import Grid
from search import AI, search

gym.register(
    id='MyCustomEnv-v0',
    entry_point='rule_based_env:Game2048Env',  # 使用 '文件名:环境类名' 的格式
)

env = gym.make("MyCustomEnv-v0")
observation = env.reset()
matrix_zt = observation['observation']

for i in range(2000):
    # 渲染环境
    env.render()

    # 使用随机动作执行操作
    # action selection
    action = env.action_space.sample()

    grid = Grid(matrix_zt)
    me = AI(grid)
    dir_zt = search(me)
    if (dir_zt == 'UP'):
        action = 0
    elif (dir_zt == 'DOWN'):
        action = 2
    elif (dir_zt == 'LEFT'):
        action = 3
    elif (dir_zt == 'RIGHT'):
        action = 1

    observation, reward, done, info = env.step(action)
    # if i > 32:
    #     print('zt')
    print('i: {}, action: {}, and reward: {}'.format(i, action, reward))
    matrix_zt = observation['observation']

    # 如果环境已结束（如，倒立摆已倒下），重置环境
    if done:
        print('done')
        print('total_step_number: {}'.format(i))
        break
        # observation = env.reset()
