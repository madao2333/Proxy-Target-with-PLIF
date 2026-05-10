# 本科毕业论文大纲：PLIF 在脉冲强化学习中的任务相关优势分析

## 摘要

本文围绕“可学习时间常数 PLIF 是否能提升 SNN 强化学习性能”展开。现有实验结果显示，PLIF 在 Walker2d-v4 上优势最明显，5 个 seed 全部优于 LIF，平均最大评估回报提升约 48.14%；在 InvertedDoublePendulum-v4 上基本持平，在 Ant-v4、HalfCheetah-v4、Hopper-v4 上暂未表现出稳定优势。

因此论文主线建议写成：PLIF 的优势不是所有任务上无条件成立，而是在需要更强时序保持和动态适应能力的任务中更明显。后续通过 tau 演化、固定 tau 对照实验和 STBP 梯度分析解释这一现象。

## 1. 绪论

### 1.1 研究背景

- 强化学习在连续控制任务中的应用。
- 脉冲神经网络 SNN 的低功耗和生物可解释性潜力。
- SNN 在连续控制任务中面临的训练困难：脉冲离散性、梯度传播困难、动作输出连续性要求。

### 1.2 问题提出

- 传统 LIF 神经元使用固定膜电位衰减系数 tau。
- 固定 tau 难以适配不同环境中的时序依赖。
- PLIF 将 tau 设为可学习参数，可能提升 SNN actor 对任务动态的适应能力。

### 1.3 研究目标

- 在 Proxy Target + TD3 框架下比较 LIF 与 PLIF 的性能。
- 分析 PLIF 中 tau 在不同任务、不同网络层中的学习趋势。
- 通过固定 tau LIF 消融实验判断 PLIF 优势是否来自更合适的时间常数。
- 通过 STBP 梯度打印分析 PLIF 对梯度传播和训练稳定性的影响。

### 1.4 主要贡献

- 在多个 MuJoCo 连续控制环境中完成 LIF、PLIF、ANN、CLIF 对比实验。
- 发现 PLIF 在 Walker2d-v4 上取得稳定优势，但在其他任务上呈现任务相关差异。
- 从 tau 演化和梯度传播角度解释 PLIF 的作用机制。
- 给出 PLIF 在 SNN 强化学习中的适用边界，而不是简单声称全面优于 LIF。

## 2. 相关理论与方法基础

### 2.1 强化学习与连续控制

- 马尔可夫决策过程 MDP。
- Actor-Critic 框架。
- TD3 算法的核心思想：双 critic、延迟策略更新、目标动作平滑。

### 2.2 脉冲神经网络

- SNN 与 ANN 的差异。
- 脉冲发放、膜电位、阈值和重置机制。
- LIF 神经元模型。

### 2.3 PLIF 神经元

- PLIF 与 LIF 的主要差异：tau 可学习。
- 当前代码中 PLIF tau 初始化与 LIF 固定值 0.75 对齐。
- tau 对膜电位保持、历史信息记忆和脉冲发放节奏的影响。

### 2.4 代理梯度与 STBP

- 脉冲发放函数不可导的问题。
- surrogate gradient 的基本思想。
- STBP 如何在时间维度和网络层维度上传播梯度。

### 2.5 Proxy Target 框架

- Proxy Target 用于缓解 SNN 离散脉冲输出与连续动作控制之间的差距。
- 说明 actor、critic、proxy target 的更新关系。
- 本文实验基于 Proxy Target + TD3 框架展开。

## 3. PLIF 脉冲 Actor 设计

### 3.1 整体网络结构

- Population Spike Encoder：将连续状态编码为脉冲序列。
- Spike MLP：使用 LIF 或 PLIF 神经元进行时序脉冲计算。
- Population Decoder：将输出脉冲活动解码为连续动作。

### 3.2 LIF 与 PLIF 的计算差异

- LIF 使用固定 tau：

```text
volt = volt * fixed_tau * (1 - spike) + current
```

- PLIF 使用可学习 tau：

```text
volt = volt * learnable_tau * (1 - spike) + current
```

- 论文中需要说明：当前代码打印的 tau 更接近膜电位衰减/保留系数，而不是 TD3 目标网络软更新中的 tau。

### 3.3 PLIF tau 参数化

- PLIFNode 中使用可学习参数 w。
- 通过 sigmoid 将 w 映射到 tau 范围。
- 每个隐藏层和输出 population 层各有一个 PLIFNode。

### 3.4 训练流程

- 环境交互并存入 replay buffer。
- 更新 proxy target。
- 更新 critic。
- 延迟更新 spiking actor。
- 对 PLIF actor，tau 参数与普通 actor 权重使用不同学习率。

## 4. 实验设置

### 4.1 实验环境

- Ant-v4。
- HalfCheetah-v4。
- Hopper-v4。
- InvertedDoublePendulum-v4。
- Walker2d-v4。

### 4.2 对比方法

- ANN：非脉冲 TD3 actor baseline。
- LIF：固定 tau 脉冲 actor。
- PLIF：可学习 tau 脉冲 actor。
- CLIF：带电流衰减的脉冲 actor。

重点比较 LIF 与 PLIF，同时使用 ANN 和 CLIF 作为补充参照。

### 4.3 评价指标

- 最大评估回报 max evaluation reward。
- 平均学习曲线。
- seed 级别胜负统计。
- PLIF tau 随训练变化曲线。
- STBP 梯度范数和 tau 参数梯度。

### 4.4 当前已有实验结果

- LIF 与 PLIF 在 5 个环境、5 个 seed 上已有对齐实验。
- figures 中已有 LIF/PLIF 最大回报图、学习曲线、PLIF tau 曲线和 Walker2d seed 曲线。
- logs 中还包含 ANN、CLIF 等更多结果，后续论文表格应补全这些实验。

### 4.5 待补实验

- 固定 tau LIF sweep：
  - tau = 0.50
  - tau = 0.60
  - tau = 0.70
  - tau = 0.75
  - tau = 0.80
  - tau = 0.90
  - tau = 0.95

- STBP 梯度打印：
  - actor 普通权重梯度范数。
  - PLIF tau 参数梯度。
  - 不同层梯度范数。
  - 梯度随训练步骤变化曲线。

## 5. 实验结果与分析

### 5.1 总体性能对比

- 汇总 ANN、CLIF、LIF、PLIF 在所有环境上的最大评估回报。
- 不只使用 figures 中已有的 LIF/PLIF 表格，应从 logs 和 results 中补全所有实验。

### 5.2 LIF 与 PLIF 对比

当前 LIF 与 PLIF 最大评估回报均值对比：

| 环境 | LIF 均值 | PLIF 均值 | PLIF - LIF | 相对变化 | PLIF 胜出 seed 数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ant-v4 | 5477.796 | 4614.055 | -863.741 | -15.77% | 2/5 |
| HalfCheetah-v4 | 9685.713 | 9467.573 | -218.140 | -2.25% | 1/5 |
| Hopper-v4 | 3498.091 | 3411.046 | -87.045 | -2.49% | 2/5 |
| InvertedDoublePendulum-v4 | 9346.853 | 9347.697 | 0.844 | 0.01% | 4/5 |
| Walker2d-v4 | 3223.649 | 4775.394 | 1551.745 | 48.14% | 5/5 |

分析重点：

- Walker2d-v4 是 PLIF 的主要正例，优势稳定且幅度大。
- InvertedDoublePendulum-v4 接近饱和，PLIF 与 LIF 基本持平。
- Ant-v4、HalfCheetah-v4、Hopper-v4 上 PLIF 没有稳定超过 LIF，说明可学习 tau 不一定自动带来性能提升。

### 5.3 学习曲线分析

- 比较不同算法的收敛速度。
- 观察 PLIF 是否在训练后期超过 LIF。
- 重点展示 Walker2d-v4 每个 seed 的曲线，说明优势不是个别 seed 偶然造成。

### 5.4 PLIF tau 演化分析

- 对比不同环境中 hidden h0、hidden h1、output 层 tau 的变化。
- 分析 tau 是否偏离 LIF 固定值 0.75。
- Walker2d-v4 中 hidden tau 明显升高，可能表示模型需要更强的历史状态保持。
- HalfCheetah-v4 中输出层 tau 可能降低，说明不同层对时间尺度的需求不同。

### 5.5 固定 tau 消融实验分析

- 将固定 tau LIF 的不同取值与 PLIF 对比。
- 如果某个固定 tau 接近 PLIF 性能，说明 PLIF 可能主要起到自动选择 tau 的作用。
- 如果 PLIF 仍优于所有固定 tau，说明按层学习和训练过程中的动态调整也很重要。

### 5.6 STBP 梯度分析

- 打印和绘制 actor 权重梯度范数。
- 打印和绘制 PLIF tau 参数梯度。
- 比较不同任务中梯度是否更稳定。
- 分析 PLIF 是否改善梯度消失、梯度过小或脉冲发放过稀疏的问题。

## 6. 讨论

### 6.1 为什么 Walker2d-v4 更能体现 PLIF 优势

- Walker2d-v4 需要更复杂的双腿协调。
- 控制过程具有较强时序依赖。
- 固定 tau 可能难以同时适应不同阶段的动态需求。
- PLIF 可以在隐藏层中学习更长的膜电位保持时间。

### 6.2 为什么部分任务 PLIF 不占优

- 任务可能较容易或回报接近饱和。
- tau 学习可能增加优化难度。
- 不同环境对时序记忆的需求不同。
- 当前超参数可能更适合 LIF 或特定任务。

### 6.3 PLIF 的适用边界

- PLIF 不应被表述为全面替代 LIF。
- 更合理的结论是：PLIF 为 SNN actor 提供了任务自适应的时间尺度建模能力。
- 在时序依赖更强的连续控制任务上，PLIF 更有潜力。

### 6.4 实验局限

- 实验环境数量有限。
- 每个环境 seed 数量有限。
- 固定 tau 消融和 STBP 梯度实验还需要补充。
- 暂未分析脉冲率、能耗和推理延迟。

## 7. 结论与展望

### 7.1 结论

- PLIF 在 Walker2d-v4 上取得稳定且显著的性能提升。
- PLIF 在 InvertedDoublePendulum-v4 上与 LIF 基本持平。
- PLIF 在 Ant-v4、HalfCheetah-v4、Hopper-v4 上暂未显示稳定优势。
- 可学习 tau 的价值主要体现在任务相关的时间尺度自适应，而不是所有任务上的无条件性能提升。

### 7.2 展望

- 扩展更多连续控制环境。
- 系统研究 tau 初始化和 tau 学习率。
- 加入固定 tau sweep 完整消融。
- 加入 STBP 梯度可视化。
- 进一步分析脉冲率和能耗优势。

## 图表安排

| 编号 | 内容 |
| --- | --- |
| 表 1 | 实验环境与超参数设置 |
| 表 2 | ANN、CLIF、LIF、PLIF 最大评估回报汇总 |
| 表 3 | LIF vs PLIF 的均值、最优值和 seed 胜负统计 |
| 图 1 | Proxy Target + SNN Actor 框架图 |
| 图 2 | LIF 与 PLIF 最大评估回报柱状图 |
| 图 3 | 五个环境的 LIF/PLIF 学习曲线 |
| 图 4 | Walker2d-v4 每个 seed 的学习曲线 |
| 图 5 | PLIF 各层 tau 随训练变化曲线 |
| 图 6 | 固定 tau LIF sweep 与 PLIF 对比 |
| 图 7 | STBP 梯度范数与 tau 梯度变化曲线 |

## 写作注意事项

- 不要把论文结论写成“PLIF 全面优于 LIF”。
- 推荐表述为：“PLIF 通过可学习 tau 提供了更强的任务自适应能力，并在 Walker2d-v4 这类时序依赖更强的任务中表现出明显优势。”
- 需要区分两个 tau：
  - PLIF 神经元中的 tau：膜电位衰减/保留系数。
  - TD3 中的 tau：目标网络软更新系数。
- 后续实验表格应优先从 logs 和 results 自动生成，避免手动整理出错。
