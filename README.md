# 月球着陆器 DQN 强化学习
本项目使用深度 Q 网络（Deep Q-Network, DQN）训练智能体在 OpenAI Gym 的 **LunarLander-v3** 环境中完成着陆任务。实现了经验回放（Experience Replay）、固定 Q 目标（Fixed Q-Targets）和软更新（Soft Update）等关键技术，是强化学习入门与算法实践的典型项目。
## 项目结构
├── lunar_lander_starter.py # DQN 训练主程序

└── README.md # 项目说明

## 环境说明

- **LunarLander-v3**：经典强化学习环境，智能体需要控制月球着陆器安全降落在指定平台上。
- **状态空间**：8 维连续状态（位置、速度、角度、角速度、左右腿触地状态）
- **动作空间**：4 个离散动作（无操作、左引擎、主引擎、右引擎）
- **奖励**：成功着陆获得正奖励，坠毁或超出边界获得负奖励

## 算法原理

### DQN 核心机制

1. **经验回放**：将每次交互的 (state, action, reward, next_state, done) 存入回放缓冲区，训练时随机采样小批量，打破数据相关性。
2. **固定 Q 目标**：使用一个目标网络（target network）计算 Q 值目标，其参数每隔一段时间软更新，减少训练震荡。
3. **ε-greedy 探索策略**：以 ε 概率随机选择动作，以 1-ε 概率选择当前 Q 值最大的动作，随着训练进行 ε 逐渐衰减。

### 网络结构

- 输入：8 维状态
- 全连接层：64 → 128 → 64 → 4（输出各动作的 Q 值）
- 激活函数：ReLU

## 依赖安装

```bash
pip install gymnasium torch numpy
```
使用方法
训练智能体
bash
python lunar_lander_starter.py
训练过程将输出每 10 个 episode 的平均奖励和损失，共训练 1000 个 episode。

主要超参数
参数	值	说明
学习率	0.001	Adam 优化器学习率
折扣因子 γ	0.99	未来奖励的折扣系数
批大小	128	每次训练采样的经验数量
经验池大小	20000	最大存储经验数
初始 ε	1.0	初始探索率
最终 ε	0.1 / 0.01	探索率下限
ε 衰减率	0.99	每个 episode 后 ε 乘以该系数
软更新率 τ	0.001	目标网络更新速率
训练结果
训练 1000 个 episode 后，智能体通常能够学会稳定着陆，获得 200+ 的平均奖励。训练过程中奖励曲线应呈现上升趋势，损失逐渐收敛。

典型输出示例
text
Episode 10, Reward: -120.45, Epsilon: 0.904, Loss: 0.32
Episode 20, Reward: -80.12, Epsilon: 0.818, Loss: 0.28
...
Episode 900, Reward: 220.34, Epsilon: 0.010, Loss: 0.05
自定义与改进方向
调整网络结构：修改 QNetwork 类的层数和神经元数量，尝试更深的网络或 Dropout。

调整超参数：改变学习率、批大小、ε 衰减率等，观察对收敛速度和最终性能的影响。

添加优先级经验回放：实现 PER（Prioritized Experience Replay）提高样本效率。

保存与加载模型：使用 torch.save(agent.qnetwork_local.state_dict(), "model.pth") 保存训练好的模型。

参考文献
Mnih, V. et al. (2015) ‘Human-level control through deep reinforcement learning’, Nature, 518(7540), pp. 529–533. DOI: 10.1038/nature14236.

OpenAI Gym: LunarLander-v3

作者
[李梓赫]
