import math
import os.path

import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import random
from AdaptiveCN import Adaptive_Cognitive_Nevigation, Find_Threshold_Neighbors
import collections
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim).to(device)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)



class Data_L:
    def __init__(self, L_state, know, l_reward, next_L_state, done):
        self.L_state = L_state
        self.know = know
        self.l_reward = l_reward
        self.next_L_state = next_L_state
        self.done = done


class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        self.buffer.append(data)


    def get(self, batch_size,device):
        mini_batch = random.sample(self.buffer, batch_size)
        state = torch.tensor([data.L_state for data in mini_batch], dtype=torch.float).to(device)
        action = torch.tensor([data.know for data in mini_batch], dtype=torch.long).to(device)
        reward = torch.tensor([data.l_reward for data in mini_batch], dtype=torch.float).to(device)
        next_state = torch.tensor([data.next_L_state for data in mini_batch], dtype=torch.float).to(device)
        done = torch.tensor([data.done for data in mini_batch], dtype=torch.float).to(device)
        return state, action, reward, next_state, done


class PPO_Sim:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device


        self.tor_fc = torch.nn.Sequential(
            torch.nn.Linear(3,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1)
        )

        self.sig = torch.nn.Sigmoid()

        self.memory_counter = 0
        self.memory = Memory(capacity=2000)

        self.batch_size = batch_size


    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    def rasch_diff(self,state,p=0.5):
        state = state + np.log(p / (1 - p)) # Eq.(15), rasch model
        return state


    def get_torlerance(self,state,last_tor,last_prac_num,diff,threshhold):

        tol_model_path = "./Trained_tolerance_model.pt"
        if os.path.exists(tol_model_path):
            tol_model = torch.load(tol_model_path)
            torlerance = tol_model(state,last_tor,last_prac_num,diff)
        else: # auxiliary statistical method
            ability = float(last_tor -last_prac_num) / float(last_tor)
            delta = threshhold - state
            torlerance = 1 + (2 * (delta / ability) if ability != 0 else (last_tor+3))
            if torlerance > 20:
                torlerance = 20

        return round(torlerance)

    def filter_candidates(self, candidates, tensor):
        """
        判断 tensor 的每个值是否低于 tensor 最后一个元素的值，并筛选出符合条件的值。
        同时更新 candidates，去掉被筛选的下标。

        参数:
        - candidates: 一个包含下标或节点的列表
        - tensor: 一个大小为 (1, N) 的 Tensor

        返回:
        - 过滤后的 candidates 和对应的 tensor
        """
        # 提取最后一个值
        threshold = tensor[0, -1].item()  # 将最后一个值转换为 Python 数值

        # 使用布尔掩码判断每个元素是否低于阈值
        mask = tensor[0] >= threshold  # Tensor中的值 >= 最后一个值保留 (>= 的原因是"低于"除去)


        # 使用布尔掩码对 tensor 和 candidates 同时筛选
        filtered_tensor = tensor[0][mask].unsqueeze(0)  # 保留符合条件的元素，并重新调整维度为(1, N)
        filtered_candidates = [candidates[i] for i in range(len(candidates)) if mask[i]]

        return filtered_candidates, filtered_tensor


    def take_action(self, state, know, Know_G, threshold, last_tor=3,last_prac_num=2):
        probs = self.actor(state.to(device))
        mastery = float(state[0][ know])
        #找到与该节点相似的所有知识概念，并且掌握程度都是小于阈值
        candidates = Find_Threshold_Neighbors(Know_G, know, state, threshold)
        candidates_probs = []
        for i in (range(len(candidates))):
            candidates_probs.append(i)
        candidate_probs = probs.gather(dim=1, index=torch.Tensor(candidates_probs).long().view(1, -1).to(device))
        action = int(random.choices(candidates, weights=candidate_probs.cpu().view(-1).detach().numpy())[0])
        next_mastery = float(state[0][action])
        diff = self.rasch_diff(next_mastery)
        if next_mastery > threshold:
            torlerance = 1
        else:
            torlerance = self.get_torlerance(next_mastery,last_tor,last_prac_num,diff,threshold)

        # #筛选掉比当前节点小的节点
        # candidates, candidate_probs = self.filter_candidates(candidates, candidate_probs)

        # actions = []
        # torlerances = []
        # # 将权重转为 numpy 数组
        # weights = candidate_probs.cpu().view(-1).detach().numpy()
        # while candidates:
        #     action = int(random.choices(candidates, weights=weights)[0])
        #     actions.append(action)
        #     next_mastery = float(state[0][action])
        #     diff = self.rasch_diff(next_mastery)
        #     if next_mastery > threshhold:
        #         torlerance = 1
        #     else:
        #         torlerance = self.get_torlerance(next_mastery,last_tor,last_prac_num,diff,threshhold)
        #     torlerances.append(torlerance)
        #
        #     # 移除选中的元素及其对应的权重
        #     index = candidates.index(chosen)  # 找到选中元素的索引
        #     candidates.pop(index)  # 移除该元素
        #     weights = list(weights)  # 转为列表，便于删除操作
        #     weights.pop(index)  # 移除对应的权重

        return action, torlerance


    def learn(self):
        states, actions, rewards, next_states, dones = self.memory.get(self.batch_size,self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(2, actions.unsqueeze(1).unsqueeze(-1))).detach()

        for _ in range(self.epochs):

            log_probs = torch.log(self.actor(states).gather(2, actions.unsqueeze(1).unsqueeze(-1)))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

