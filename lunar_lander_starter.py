import gymnasium as gym
import torch
import torch.nn.functional as F
# ... (other imports stay the same)
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # TODO: Experiment with network depth. 
        # For LunarLander, 2-3 hidden layers with 64-128 units are standard.
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        # ... (hyperparameters stay the same)
        self.learn_rate = 0.001
        self.gamma = 0.99
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = 128
        self.memory = deque(maxlen=20000)
        self.network_update_rate = 0.001
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_reduce = 0.99
        self.loss = 0
        
        # TODO: Initialize TWO networks for Fixed Q-Targets
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learn_rate)
        self.criterion = nn.MSELoss()

    # ε-greedy 选择动作
    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))  # 随机动作
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.qnetwork_local(state)
        return q_values.argmax().item()
    
    # 添加 step 方法存储经历
    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.learn()  # 每一步尝试学习

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        # TODO: 1. Sample a batch of experiences (s, a, r, s', done)
        experiences = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        
        # TODO: 2. Convert to PyTorch Tensors
        # states, actions, rewards, next_states, dones = ...
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # TODO: 3. Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # TODO: 4. Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # TODO: 5. Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # TODO: 6. Compute loss, backpropagate, and update local network
        loss = self.criterion(Q_expected, Q_targets)
        self.loss = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO: 7. Soft Update the Target Network (important for stability)
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.network_update_rate*local_param.data + (1.0-self.network_update_rate)*target_param.data)
        
        return loss.item()



num_episodes = 1000
reward_history = []
loss_history = []

agent = DQNAgent(state_size, action_size)

for episode in range(num_episodes):
    state, Info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, Info = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        final_loss = agent.loss
    reward_history.append(total_reward)
    if episode >750:
         agent.epsilon_min = 0.01
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_reduce)
    
    if (episode+1) % 10 == 0:
        print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, Loss: {final_loss:.2f}")
