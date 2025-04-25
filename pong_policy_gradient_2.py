#!/usr/bin/env python
# coding: utf-8

# In[71]:


import gym
import numpy as np
np.bool8 = np.bool_


# In[72]:


# 创建环境
env = gym.make("Pong-v4")


# In[73]:


# env.reset()
# state,_ = env.reset()
# done = False
# count = 0
# while not done:
# #     env.render()
#     action = int(np.random.choice([2,3]))
#     next_state, reward, done, truncated, _ = env.step(action)
#     print(action,reward)
#     count += 1
# print(count)


# In[74]:


# env.close()


# In[75]:


from torch import nn
import torch


# In[76]:


def prepro(I):
    """将 210x160x3 uint8 帧预处理为 6400 (80x80) 1D float 向量"""
    I = I[35:195]  # 裁剪
    I = I[::2, ::2, 0]  # 下采样因子为 2
    I[I == 144] = 0  # 删除背景类型 1
    I[I == 109] = 0  # 删除背景类型 2
    I[I != 0] = 1  # 其他设置为 1
    return I.astype(np.float32).ravel()


# In[77]:


class PolicyNet(nn.Module):
    
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim,200)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(200,output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self,state):
        ### n
        x = self.linear1(state)
        x = self.relu(x)
        x = self.linear2(x) # n
        x = self.softmax(x) # n
        return x


# In[78]:


from torch.distributions import Categorical
import numpy as np
np.bool8 = np.bool_

from torch.optim import AdamW


# In[92]:


class Agent:
    
    def __init__(self):
        self.policy_net = PolicyNet(6400,2)
        self.optimizer = AdamW(self.policy_net.parameters(),lr=1e-3)
    
    def sample_action(self,state):
        probs = self.policy_net(state) # 4
        if np.random.uniform() < 0.2:
            action = np.random.randint(0,2)
            return action + 2, torch.log(probs[action]+1e-8)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item()+2,log_prob
    
    def update(self,rewards,log_probs):
        ### 一次游戏时间
        ret = []
        adding = 0
        for r in rewards[::-1]:
            if r != 0:
                adding = 0
            adding = adding * 0.99 + r
            ret.insert(0,adding)
        ret = torch.FloatTensor(ret)
        ret = ret - ret.mean()
        ret = ret / (ret.std()+1e-8)
        
        r_log_probs = []
        for r,log_prob in zip(ret,log_probs):
            r_log_probs.append(-r*log_prob)
        r_log_probs = torch.vstack(r_log_probs)
        
        loss = r_log_probs.sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss


# In[ ]:





# In[102]:


def train(agent,env):
    success_count = []
    max_size = 2000
    for epoch in range(20000):
        rewards = []
        log_probs = []
        terminated = False
        state,_ = env.reset()
        prev_x = None
        while not terminated:
            x = prepro(state)
            diff = np.zeros(6400) if prev_x is None else x - prev_x
            prev_x = x
            diff = torch.FloatTensor(diff)
            action, log_prob = agent.sample_action(diff)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            rewards.append(reward)
            log_probs.append(log_prob)
        
        loss = agent.update(rewards,log_probs) 
        
        
        if (epoch+1) % 10 == 0:
#             torch.save('pong.pt',agent.policy_net)
            torch.save(agent.policy_net,'pong.pt')
            print(f'epoch: {epoch}, loss: {loss}, rewards: {sum(rewards)}, count: {len(rewards)}')


# In[81]:


agent = Agent()


# In[100]:


# torch.save(agent.policy_net,'pong.pt')


# In[101]:


# torch.load('pong.pt')


# In[98]:


env = gym.make("Pong-v4")
train(agent,env)


# In[90]:


def sample_action(self,state):
    probs = self.policy_net(state) # 4
    if np.random.uniform() < 0.0:
        action = np.random.randint(0,2)
        return action + 2, torch.log(probs[action]+1e-8)
    dist = Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item()+2,log_prob

# 替换方法
import types
agent.sample_action = types.MethodType(sample_action, agent)


# In[ ]:


import time
def visualize_agent(env, agent, num_episodes=5):
    """
    渲染显示智能体的行动
    """
    env = gym.make('CliffWalking-v0', render_mode='human')  # 创建可视化环境
    
    for episode in range(num_episodes):
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        total_reward = 0
        steps = 0
        done = False
        
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            env.render()  # 渲染当前状态
            
            # 将状态转换为one-hot编码
            state_onehot = np.zeros(48)
            state_onehot[state] = 1
            
            # 使用训练好的策略选择动作
            with torch.no_grad():
                if np.random.random() < 0.0:
                    action = np.random.randint(0, 4)
                else:
                    state_tensor = torch.FloatTensor(state_onehot)
                    probs = agent.policy_net(state_tensor)
                    action = probs.argmax().item()  # 使用最可能的动作
            
            # 执行动作
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, _ = step_result
            else:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            # 添加小延迟使动作更容易观察
            time.sleep(0.5)
        
        print(f"Episode finished after {steps} steps. Total reward: {total_reward}")
    
    env.close()

# 在主程序最后添加：
if __name__ == "__main__":    
    # 训练完成后显示智能体行动
    print("\nVisualizing trained agent behavior...")
    env = gym.make('CliffWalking-v0',render_mode='human')
    visualize_agent(env, agent)


# In[ ]:


env.close()


# In[ ]:




