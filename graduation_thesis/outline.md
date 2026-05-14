# 本科毕业论文大纲：PLIF 与 LIF 脉冲强化学习的训练机制对比分析

## 摘要

本文围绕“可学习时间常数 PLIF 是否改变 Proxy Target 脉冲强化学习的训练机制”展开。当前实验结果显示，PT-PLIF 与 PT-LIF 在总体回报上并未表现出稳定、普适的性能优势差距；这也是合理的，因为连续控制强化学习的性能上限不可能仅通过替换神经元模型直接突破。

因此论文主线建议从“证明 PLIF 全面优于 LIF”调整为“解释脉冲 actor 为什么会学习不动，以及 PLIF/LIF 在 Proxy Target 框架下为什么表现接近”。论文首先对失败组进行 STBP 梯度、放电率、电流、膜电位和参数摘要的深入分析，并与成功组进行对比；然后说明通过调整 `proxy_lr` 与 `policy_freq` 可以缓解这种训练失败；最后结合 LIF 的梯度和参数轨迹，论证 PLIF 与 LIF 在该框架下具有较强的实验等价性和相近的性能表现。

## 1. 绪论

### 1.1 研究背景

- 强化学习在连续控制任务中的应用。
- 脉冲神经网络 SNN 的低功耗和生物可解释性潜力。
- SNN 在连续控制任务中面临的训练困难：脉冲离散性、梯度传播困难、动作输出连续性要求。

### 1.2 问题提出

- 传统 LIF 神经元使用固定膜电位衰减系数 tau。
- 固定 tau 难以适配不同环境中的时序依赖。
- PLIF 将 tau 设为可学习参数，可能提升 SNN actor 对任务动态的适应能力。
- 但现有实验表明，仅将 LIF 替换为 PLIF 并不会稳定突破强化学习任务的性能上限。
- 更关键的问题是：在 Proxy Target + STBP 训练中，SNN actor 何时会进入“放电仍然存在但梯度已经消失”的学习停滞状态。

### 1.3 研究目标

- 在 Proxy Target + TD3 框架下比较 LIF 与 PLIF 的性能接近性和训练稳定性。
- 分析失败组中 STBP 梯度塌缩、放电率饱和、电流增大和参数正偏之间的关系。
- 分析成功组中 PLIF tau、权重、偏置、电流和膜电位的更新方向。
- 通过 `proxy_lr` 与 `policy_freq` 的调参对照，记录哪些设置能够缓解训练失败。
- 通过补充 LIF 的梯度和参数分析，论证 LIF 与 PLIF 在实验结果和训练机制上的接近性。

### 1.4 主要贡献

- 在多个 MuJoCo 连续控制环境中完成 LIF、PLIF、ANN、CLIF 对比实验，并指出 PT-PLIF 相比 PT-LIF 并不存在稳定的全面优势。
- 对 Hopper-v4 中“学习不动”的失败组进行细粒度诊断，发现其表现为高放电/饱和放电状态下的 STBP 梯度塌缩，而不是简单的脉冲沉默。
- 将失败组与成功组对比，说明成功训练依赖于更健康的电流尺度、膜电位分布、非零梯度和 PLIF 时间常数演化。
- 提出 `proxy_lr` 与 `policy_freq` 是影响该失败模式的重要超参数，并通过调参前后的实验记录说明有效配置。
- 将论文结论收束为：PLIF 提供了可学习时间尺度，但在当前 Proxy Target 强化学习框架中更适合作为机制分析对象，而不是被表述为全面优于 LIF 的替代模型。

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
- 在 proxy target 更新阶段，SNN actor 的输出作为监督信号，proxy target ANN 通过 MSE 拟合该动作输出。
- 对 actor 输出使用 `torch.no_grad()` 或 `detach()`，使 proxy loss 只反向传播到 proxy target ANN，避免对 SNN actor 进行无效的 STBP 反向计算。
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
- 更新 proxy target：
  - 使用当前 SNN actor 生成动作标签。
  - 将 actor 输出从计算图中分离，作为固定监督目标。
  - 使用 MSE 损失只更新 proxy target ANN。
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
- STBP 梯度范数、signed gradient mean 和 tau 参数梯度。
- 各层放电率、current mean/current abs mean、volt mean/volt std。
- SNN actor 参数摘要：权重均值、标准差、范数、偏置均值和范数。

### 4.4 当前已有实验结果

- LIF 与 PLIF 在 5 个环境、5 个 seed 上已有对齐实验。
- figures 中已有 LIF/PLIF 最大回报图、学习曲线和 PLIF tau 曲线。
- logs 中还包含 ANN、CLIF 等更多结果，后续论文表格应补全这些实验。
- Hopper-v4 上已有 PLIF 失败组与调参成功组的 STBP trace 和参数摘要，可作为训练失败机制分析的主要案例。
- LIF 失败组已有 STBP trace 和参数摘要，可用于后续论证 LIF/PLIF 失败模式和结果表现的接近性。

### 4.5 待补实验

- `proxy_lr` 与 `policy_freq` 调参对照：
  - 保留默认失败设置作为失败基线。
  - 记录跑完后表现有效的调参设置，例如降低 `proxy_lr`、增大 `policy_freq` 或二者组合。
  - 对比失败组与有效组的 reward、STBP 梯度、放电率、current/volt 和参数摘要。
  - 不强行拆解每个超参数的独立作用，结论写成“该配置能够缓解失败模式”。

- 成功组更新机制补充：
  - 记录 W/b 梯度 signed mean。
  - 分析 hidden/output 层 W mean、W std、bias mean 与 current mean 的关系。
  - 分析 PLIF tau 增大后，膜电位泄露项与 current 项在 `v_{t+1}` 中的相对贡献。

- LIF 与 PLIF 等价性分析：
  - 补充 LIF 成功组和失败组的 STBP 梯度、放电率、current、volt 和参数摘要。
  - 对比 LIF 与 PLIF 在相同失败模式下是否都表现出高放电、current 正偏和梯度塌缩。
  - 对比 LIF 与 PLIF 成功组是否都形成较健康的放电率和非零梯度。

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
  - actor 普通权重和偏置梯度的 signed mean。
  - PLIF tau 参数梯度。
  - 不同层梯度范数。
  - 梯度随训练步骤变化曲线。

## 5. 实验结果与分析

### 5.1 总体性能对比

- 汇总 ANN、CLIF、LIF、PLIF 在所有环境上的最大评估回报。
- 不只使用 figures 中已有的 LIF/PLIF 表格，应从 logs 和 results 中补全所有实验。

### 5.2 LIF 与 PLIF 总体性能对比

当前写作重点不应放在“PLIF 是否显著超过 LIF”，而应放在二者整体表现接近这一事实：

- PT-PLIF 与 PT-LIF 在多个连续控制任务上的最终回报差距并不稳定。
- 个别环境或 seed 上 PLIF 可能更好，但该现象不足以支持“替换神经元即可突破强化学习上限”的强结论。
- 这说明 LIF 与 PLIF 的差异主要体现在训练动力学、时间常数适应和失败模式上，而不是简单的最终性能支配关系。
- 论文表格应使用最新 logs/results 自动生成，并报告均值、标准差、最优值和 seed 胜负，而不是只挑选优势任务。

### 5.3 学习曲线分析

- 比较不同算法的收敛速度。
- 观察 LIF 与 PLIF 是否都存在“早期学习失败”和“后期成功收敛”的分化。
- 对成功组与失败组分别展示学习曲线，说明相同算法在不同超参数下可能进入完全不同的训练状态。
- 弱化“PLIF 后期超过 LIF”的叙事，改为分析二者是否具有相近的收敛上限和失败概率。

### 5.4 PLIF tau 演化分析

- 对比不同环境中 hidden h0、hidden h1、output 层 tau 的变化。
- 分析 tau 是否偏离 LIF 固定值 0.75。
- 对成功组重点分析 hidden tau 被推高的过程：更大的 tau 使未放电神经元保留更多膜电位历史，减少对瞬时 current 的依赖。
- 对失败组说明 tau 基本停留在初始化附近，因为 STBP 梯度早期塌缩后 PLIF 参数失去学习信号。
- 区分 hidden 层和 output 层：hidden 层可能学习更长的时间保持，output 层更直接服务于动作输出。

### 5.5 失败组机制分析：STBP 梯度塌缩

- 使用 PLIF Hopper-v4 失败组作为主要案例。
- 证明失败组并不是脉冲沉默，而是高放电/饱和放电状态下的梯度死亡。
- 分析链条：
  - 默认 `proxy_lr=1e-3, policy_freq=2` 下 actor 更新较密。
  - 早期参数形成正向偏置结构：正 W mean、正 bias、高 pre-spike rate。
  - current mean/current abs mean 被推高，膜电位偏离 surrogate gradient 的有效窗口。
  - 后续 STBP 梯度、W/b 参数梯度和 PLIF tau 梯度趋近于 0。
  - actor 参数冻结，episode reward 长期停留在低水平。
- 使用参数摘要佐证：失败组后期权重范数不一定最大，但 W mean 与 bias 的正偏会在密集 pre-spike 下产生稳定正向 current。

### 5.6 成功组机制分析：可学习但不饱和的更新状态

- 使用 PLIF Hopper-v4 成功组作为对照案例。
- 分析成功组中的关键特征：
  - hidden 层 pre/post spike 更稀疏，避免整体高放电锁死。
  - W std 和 W l2 norm 增大，但 W mean 接近 0 或为负，说明权重正负分化而不是整体正偏。
  - hidden0 bias 可被推到负值，从而降低 baseline excitability。
  - output 层保留较强正向驱动，用于动作输出。
  - current_grad、param_weight_grad、param_bias_grad 和 PLIF tau 梯度持续非零。
- 结合新加入的 signed mean trace，直接分析 W/b 梯度方向，而不只依赖参数摘要反推。
- 分析 `v_{t+1} = tau * v_t * (1 - spike) + current` 中泄露项与 current 项的相对贡献，说明成功组 hidden 层形成膜电位记忆。

### 5.7 `proxy_lr` 与 `policy_freq` 调参对照分析

- 以失败组和成功组的差异为动机，比较默认失败配置与已验证有效配置。
- 可讨论两种合理解释：
  - 降低 `proxy_lr` 可能减弱 proxy target 对 actor 更新方向的扰动。
  - 增大 `policy_freq` 可能降低 actor 更新频率，避免早期过快进入饱和放电状态。
- 根据实际跑完的实验结果说明哪组设置有效，不要求严格区分单个超参数的因果贡献。
- 对每组报告 reward、STBP 梯度、放电率、current/volt 和参数摘要。
- 结论应谨慎表述为“某组调参设置缓解了失败模式”，而不是简单声称某个超参数普遍最优。

### 5.8 LIF 与 PLIF 等价性分析

- 将 LIF 成功组/失败组的 STBP trace 与 PLIF 对齐分析。
- 如果 LIF 失败组也出现高放电、current 正偏、output spike 固定、梯度塌缩，则说明该失败模式来自 Proxy Target + STBP 训练动力学，而不是 PLIF 独有。
- 如果 LIF 成功组也能通过 W/b 调整形成可学习的电流和膜电位分布，则说明固定 tau 与可学习 tau 在当前任务上可能具有相近表达能力。
- 结合最终回报接近的实验结果，论证 LIF 与 PLIF 在该框架下具有实验等价性：PLIF 提供额外自由度，但不必然带来更高上限。

### 5.9 固定 tau 消融实验分析

- 将固定 tau LIF 的不同取值与 PLIF 对比。
- 如果某个固定 tau 接近 PLIF 性能，说明 PLIF 可能主要起到自动选择 tau 的作用。
- 如果 PLIF 与多个固定 tau 结果都接近，说明当前任务和 Proxy Target 框架对 tau 自适应的需求有限。
- 如果 PLIF 仍优于所有固定 tau，说明按层学习和训练过程中的动态调整也很重要。

### 5.10 训练效率与梯度隔离分析

- 说明 proxy target 更新中的 actor 输出只承担监督标签作用，不需要对 actor 反向传播。
- 分析若不使用 `detach` 或 `torch.no_grad()`，每个 proxy iteration 都会额外触发一次 SNN actor 的 STBP 反向传播。
- 结合 `proxy_iters` 和 `policy_freq` 说明额外开销会随 proxy target 更新次数放大。
- 强调该设计不改变 proxy target 的优化目标，只减少无效梯度计算和显存占用。

## 6. 讨论

### 6.1 为什么 PLIF 不应被写成全面优于 LIF

- 强化学习性能上限由算法、环境、探索、critic 学习、actor 表达能力和超参数共同决定。
- 单纯替换 LIF 为 PLIF 只增加神经元时间常数自由度，不足以保证突破任务上限。
- 在 Proxy Target 框架下，LIF 与 PLIF 共享相同的 encoder、Spike MLP、decoder、critic 和 proxy target 更新机制，因此二者可能表现接近。
- PLIF 的价值更适合表述为提供可分析的时间尺度自适应机制，而不是保证更高回报。

### 6.2 失败模式的理论解释

- surrogate gradient 的有效窗口只覆盖阈值附近膜电位。
- 高放电率不一定带来有效梯度；如果膜电位长期远高于阈值，前向仍然放电，但反向可能进入代理梯度死区。
- 频繁 spike 会通过 `(1 - spike)` reset 项切断膜电位时间路径。
- 因此失败组表现为“高放电但不可学习”，而不是“无放电导致无梯度”。

### 6.3 LIF 与 PLIF 的适用边界

- PLIF 不应被表述为全面替代 LIF。
- 更合理的结论是：PLIF 为 SNN actor 提供了任务自适应的时间尺度建模能力，但其收益依赖任务、超参数和训练稳定性。
- 当 LIF 固定 tau 已能支持足够的膜电位记忆时，PLIF 与 LIF 的最终回报可能接近。
- 当训练失败由 current 正偏、放电饱和和梯度塌缩主导时，PLIF 的额外 tau 参数也可能因无梯度而无法发挥作用。

### 6.4 实验局限

- 实验环境数量有限。
- 每个环境 seed 数量有限。
- `proxy_lr` 与 `policy_freq` 的调参对照结果还需要整理。
- 成功组 W/b signed gradient mean 和 LIF 对照 trace 还需要补充。
- 固定 tau 消融和 STBP 梯度可视化还需要补充。
- 暂未分析脉冲率、能耗和推理延迟。

## 7. 结论与展望

### 7.1 结论

- PT-PLIF 与 PT-LIF 在当前实验中整体表现接近，不能写成 PLIF 稳定全面优于 LIF。
- SNN actor 学习失败的关键现象是 STBP 梯度塌缩：前向仍然高放电，但反向梯度和参数更新已经消失。
- 失败组与成功组的差异可以从 current、volt、放电率、W/b 参数摘要和 PLIF tau 梯度中得到解释。
- 调整 `proxy_lr` 与 `policy_freq` 可以缓解失败模式；具体有效设置以最终跑完的实验记录为准。
- PLIF 的可学习 tau 提供了有价值的机制分析窗口；LIF 与 PLIF 的实验等价性需要通过 LIF 梯度和参数轨迹进一步佐证。

### 7.2 展望

- 扩展更多连续控制环境。
- 系统研究 `proxy_lr`、`policy_freq`、tau 初始化和 tau 学习率。
- 整理 `proxy_lr` 与 `policy_freq` 调参对照结果。
- 加入固定 tau sweep 完整消融。
- 加入 STBP 梯度、signed mean、current/volt 和放电率可视化。
- 补充 LIF 成功组和失败组的梯度/参数分析，完善 LIF-PLIF 等价性论证。
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
| 图 4 | 代表性环境中 LIF/PLIF 的成功组与失败组学习曲线 |
| 图 5 | PLIF 各层 tau 随训练变化曲线 |
| 图 6 | PLIF 失败组与成功组的 STBP 梯度、放电率、current/volt 对比 |
| 图 7 | W/b 参数摘要与 signed gradient mean 变化曲线 |
| 图 8 | `proxy_lr` 与 `policy_freq` 调参对照结果 |
| 图 9 | 固定 tau LIF sweep 与 PLIF 对比 |
| 图 10 | LIF 与 PLIF 失败/成功机制对照图 |

## 写作注意事项

- 不要把论文结论写成“PLIF 全面优于 LIF”。
- 推荐表述为：“PLIF 通过可学习 tau 提供了额外的时间尺度自适应能力，但在当前 Proxy Target 强化学习框架下，PT-PLIF 与 PT-LIF 的最终性能整体接近；本文重点在于解释 SNN actor 的训练失败机制及其缓解方式。”
- 需要区分两个 tau：
  - PLIF 神经元中的 tau：膜电位衰减/保留系数。
  - TD3 中的 tau：目标网络软更新系数。
- 后续实验表格应优先从 logs 和 results 自动生成，避免手动整理出错。
- 对 `proxy_lr` 和 `policy_freq` 的结论必须基于最终跑完的实验记录，避免过度因果化。
