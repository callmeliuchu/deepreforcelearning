import numpy as np
import pickle  # 用 pickle 替代 cPickle
import gym

# 超参数
H = 200  # 隐藏层神经元数量
batch_size = 10  # 每多少集进行一次参数更新
learning_rate = 1e-4
gamma = 0.99  # 折扣因子
decay_rate = 0.99  # RMSProp 衰减因子
resume = False  # 是否从以前的检查点恢复
render = False

# 模型初始化
D = 80 * 80  # 输入维度：80x80 网格
if resume:
    with open('save.p', 'rb') as f:
        model = pickle.load(f)
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" 初始化
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# 更新缓存，用于 RMSProp 平滑梯度
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid 函数

def prepro(I):
    """将 210x160x3 uint8 帧预处理为 6400 (80x80) 1D float 向量"""
    I = I[35:195]  # 裁剪
    I = I[::2, ::2, 0]  # 下采样因子为 2
    I[I == 144] = 0  # 删除背景类型 1
    I[I == 109] = 0  # 删除背景类型 2
    I[I != 0] = 1  # 其他设置为 1
    return I.astype(np.float32).ravel()

def discount_rewards(r):
    """对奖励进行折扣"""
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        if r[t] != 0: running_add = 0  # 如果是游戏边界（仅对 Pong 特有）
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # 返回采取动作 2 的概率和隐藏状态

def policy_backward(eph, epdlogp):
    """反向传播（eph 为隐藏状态数组）"""
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # Backprop ReLU
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

# 训练
env = gym.make("Pong-v0", render_mode="human")
observation, _ = env.reset()
prev_x = None  # 用于计算差分帧
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0

while True:
    if render: env.render()

    # 预处理观测，设定网络输入为差分图像
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # 前向传播并采样动作
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # 投掷骰子！

    # 记录中间变量（后续用于反向传播）
    xs.append(x)  # 观测值
    hs.append(h)  # 隐藏状态
    y = 1 if action == 2 else 0  # “伪标签”
    dlogps.append(y - aprob)  # 梯度鼓励采取的动作

    # 环境步进，获取新状态
    observation, reward, done, _, info = env.step(action)
    reward_sum += reward

    drs.append(reward)  # 记录奖励

    if done:  # 一个回合结束
        episode_number += 1

        # 堆叠输入、隐藏状态、动作梯度和奖励
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # 重置数组

        # 计算折扣奖励
        discounted_epr = discount_rewards(epr)
        # 标准化奖励
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # 使用优势调制梯度
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]  # 在 batch 中累积梯度

        # 每 batch_size 集执行 RMSProp 参数更新
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # 重置 batch 梯度缓存

        # 记录
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'resetting env. episode reward total was {reward_sum}. running mean: {running_reward}')
        if episode_number % 100 == 0:
            with open('save.p', 'wb') as f:
                pickle.dump(model, f)
        reward_sum = 0
        observation, _ = env.reset()
        prev_x = None

    if reward != 0:  # Pong 的奖励为 +1 或 -1（游戏结束时）
        print(f'ep {episode_number}: game finished, reward: {reward}' + ('' if reward == -1 else ' !!!!!!!!'))
