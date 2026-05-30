# 毕业论文答辩 PPT 大纲

论文题目：基于类脑神经网络的强化学习算法研究

建议形式：LaTeX Beamer / PowerPoint  
建议时长：8--10 分钟  
建议页数：14--16 页  
核心叙事：PLIF 看似应比 LIF 更灵活 -> 但 PT-PLIF 未稳定优于 PT-LIF -> 因此本文转向训练机制诊断 -> 解释失败、时间常数演化和 LIF/PLIF 机制近似等价

答辩主线：
- 不是证明“PLIF 全面优于 LIF”。
- 而是在 Proxy Target + TD3 训练闭环中，分析可学习时间常数是否真正改变脉冲 actor 的训练机制。
- 重点回答：为什么会“前向仍放电、反向已失效”；PLIF 学到了什么；成功训练时 LIF 与 PLIF 是否本质不同。

## 1. 封面

标题：基于类脑神经网络的强化学习算法研究

页面内容：
- 论文题目
- 学生姓名、学号、专业
- 指导教师
- 学院与答辩日期

讲述重点：
- 一句话定位：本文研究 Proxy Target + TD3 框架下 LIF 与 PLIF 脉冲 actor 的训练机制。

## 2. 研究对象：Proxy Target 脉冲 Actor

页面内容：
- 连续控制强化学习中，actor 需要从连续状态输出连续动作。
- Proxy Target 框架使脉冲 actor 能够接入 TD3 训练流程。
- 本文关注的核心 actor 类型：LIF 与 PLIF。

建议画法：
- 状态输入 -> Population Spike Encoder -> Spike MLP -> Population Decoder -> 连续动作
- 在 Spike MLP 处标注：LIF / PLIF 是本文核心对照。
- 在旁边简化标出：Critic 提供策略梯度，Proxy Target 提供连续目标动作。

讲述重点：
- SNN 的低功耗潜力是研究动机，但本文不做硬件能耗评估。
- 本文真正关注的是：在这个训练闭环中，脉冲 actor 如何学习、何时失效、LIF 与 PLIF 是否形成不同机制。

## 3. 关键矛盾：PLIF 看起来更灵活，但并未稳定更强

页面内容：
- LIF 使用固定膜电位保留系数。
- PLIF 将时间常数设为可学习参数。
- 直觉预期：PLIF 具有更强时间尺度自适应能力，可能提升脉冲 actor 训练表现。
- 实验观察：PT-PLIF 相比 PT-LIF 未形成稳定、普适的回报优势。

建议公式：

```latex
v_t^l = \tau_l v_{t-1}^l(1-s_{t-1}^l) + I_t^l
```

```latex
\tau_l =
\begin{cases}
0.75, & \text{LIF}\\
\sigma(w_l), & \text{PLIF}
\end{cases}
```

讲述重点：
- 本文的问题不是“PLIF 是否一定更强”，而是“PLIF 的额外自由度在强化学习训练闭环中是否真的发挥了不同作用”。
- 这个矛盾是后续机制分析的出发点。

## 4. 本文核心问题与贡献

页面内容：
- 问题一：PT-PLIF 是否稳定优于 PT-LIF？
- 问题二：低回报停滞来自脉冲沉默，还是有效梯度塌缩？
- 问题三：PLIF 时间常数在成功训练中是否呈现可解释的层间调节？
- 问题四：成功训练时，LIF 与 PLIF 是否形成显著不同的内部机制？

主要贡献：
- 修正实验叙事：不把 PLIF 描述为必然提升回报的替代模型。
- 诊断失败机制：揭示“前向仍放电、反向已失效”的停滞模式。
- 分析时间常数：说明 PLIF 不是所有层同步增强记忆，而是呈现层间分化。
- 比较成功机制：发现 LIF 与 PLIF 在多项机制指标上高度一致。

讲述重点：
- 这一页要明确告诉老师：本文的价值在机制解释，而不是算法排行榜。

## 5. 方法框架：Proxy Target + TD3

页面内容：
- TD3 critic 提供策略梯度信号。
- Proxy Target 使用连续 ANN 目标网络拟合在线 SNN actor 输出。
- 在线 SNN actor 仍由 critic 的策略损失更新。
- LIF 与 PLIF 在相同网络结构、训练流程和评价协议下比较。

建议画法：
- Replay Buffer -> Critic 更新
- SNN Actor -> 当前动作 -> Critic -> Actor loss
- SNN Actor 输出 -> Proxy Target 拟合
- Proxy Target -> 下一状态动作 -> TD target

讲述重点：
- Proxy Target 不是本文要比较的变量，而是共同训练框架。
- 本文把 PLIF 放入完整强化学习闭环中考察，而不是孤立讨论神经元模型。

## 6. 脉冲 Actor 结构与 LIF/PLIF 差异

页面内容：
- Population Spike Encoder：连续状态到脉冲序列。
- Spike MLP：两层隐藏层与动作输出 population 层。
- Population Decoder：平均输出脉冲活动并解码为连续动作。
- PLIF 在每个隐藏层和动作输出层各有一个可学习时间常数。

建议突出：
- 默认仿真时间步：5。
- LIF 固定 $\tau=0.75$。
- PLIF 初始化与 LIF 对齐，但训练中可按层演化。

讲述重点：
- LIF/PLIF 的结构差异很小，但影响前向膜电位保留和反向时间梯度路径。
- 如果 STBP 梯度塌缩，PLIF 的时间常数也无法继续学习。

## 7. 实验设计：从总体结果到机制诊断

页面内容：
- 环境：Ant-v4、HalfCheetah-v4、Hopper-v4、InvertedDoublePendulum-v4、Walker2d-v4。
- 对照对象：ANN、LIF、PLIF、CLIF。
- 核心比较：PT-LIF 与 PT-PLIF。
- 性能指标：最大评估回报、学习曲线。
- 机制指标：STBP 梯度、放电率、电流、膜电位、有效保留项、时间常数、动作梯度范数。

讲述重点：
- ANN 和 CLIF 主要提供总体参照。
- Hopper-v4 作为低回报停滞的代表案例，用于深入机制分析。
- 本文不把最终回报作为唯一结论依据。

## 8. 总体结果：PT-PLIF 未稳定优于 PT-LIF

建议图：
- `../../figures/algorithm_max_eval.pdf`

页面内容：
- 展示五个 MuJoCo 环境中不同 actor 的最大评估回报。
- 强调 PLIF 没有形成跨环境、跨随机种子的稳定优势。
- 由此引出：仅看回报无法解释训练过程。

讲述重点：
- 最大回报回答“有没有稳定更强”，答案是不明显。
- 因此后续要转向学习曲线和内部机制指标。

## 9. 学习曲线：定位训练停滞现象

建议图：
- `../../figures/algorithm_learning_curves.pdf`

页面内容：
- 展示不同方法在五个环境中的平均学习曲线。
- 指出不同环境和随机种子下存在训练波动与低回报停滞。
- Hopper-v4 中的 PLIF 失败现象适合作为机制分析案例。

讲述重点：
- 学习曲线把问题从“结果高低”推进到“训练过程如何失败”。
- 后续以 Hopper-v4 代表案例解释停滞机制。

## 10. Hopper-v4 失败现象与缓解

建议图：
- `../../figures/plif_hopper_seed10991_repeat_learning_curves.pdf`

页面内容：
- 默认失败组：`policy_freq=2`，长期停留低回报。
- 调整组：`policy_freq=4`，进入高回报区间。
- 同一随机种子 10991 下重复运行，失败和缓解现象均可复现。

关键数据：
- 默认失败组最大评估回报约为 239.393。
- 调整组最大评估回报约为 3563.550。

讲述重点：
- 调整 actor 更新间隔能够缓解代表性失败现象。
- 这页不把 `policy_freq=4` 宣称为普适最优，而是把它作为机制对照入口。

## 11. STBP 梯度塌缩：反向学习通道失效

建议图：
- `../../figures/plif_stbp_gradient_collapse.pdf`

页面内容：
- 对比默认失败组和 actor 更新间隔调整组的分层 STBP 梯度。
- 失败组后期电流梯度、权重梯度、PLIF 时间常数梯度接近零。
- 成功组保持非零梯度。

建议公式：

```latex
\frac{\partial s_t}{\partial v_t}
\approx
\mathbb{I}(|v_t - V_{\mathrm{th}}| < \Delta)
```

讲述重点：
- 低回报停滞不是单纯输出动作失败，而是反向学习通道失效。
- PLIF 时间常数本身也依赖梯度信号；梯度塌缩后，额外自由度无法继续发挥作用。

## 12. 放电状态诊断：不是简单脉冲沉默

建议图：
- `../../figures/plif_stbp_spike_rates.pdf`

页面内容：
- 失败组后期仍保持较高放电率。
- 成功组隐藏层放电更稀疏，但仍保持有效梯度。
- 高放电率不等价于有效学习。

讲述重点：
- 本文识别的是“前向仍放电、反向已失效”的停滞模式。
- 频繁放电会通过重置项削弱时间递推路径。

## 13. PLIF 时间常数的层间演化

建议图：
- `../../figures/plif_tau_curves.pdf`

页面内容：
- 隐藏层时间常数通常上升。
- 动作输出层时间常数变化更保守，甚至可能下降。
- PLIF 并不是在所有层同步增强记忆。

建议公式：

```latex
R_l = \mathbb{E}[\tau_l(1-s_{t-1}^l)]
```

讲述重点：
- 隐藏层更倾向于保留亚阈值历史信息。
- 输出层更直接服务于动作 population 的即时读出。
- 这说明 PLIF 学到的是层间分化的时间尺度，而不是简单增加所有层记忆。

## 14. LIF 与 PLIF 成功机制的近似等价性

建议图：
- `../../figures/lif_plif_success_mechanism_paired.pdf`

页面内容：
- 比较成功训练状态下 LIF 与 PLIF 的配对机制指标。
- 指标包括放电率、电流幅值、膜电位尺度、代理梯度窗口占比、有效保留项、非零梯度占比、动作梯度范数等。
- 多数指标呈高度一致的配对结构。

讲述重点：
- PLIF 的有效保留项整体高于 LIF，说明额外时间常数确有调节作用。
- 但这种调节发生在 LIF 已有训练机制附近，并没有形成完全不同的学习路径。

## 15. 主要结论与研究价值

页面内容：
- PT-PLIF 与 PT-LIF 在当前实验范围内整体表现接近。
- Hopper-v4 失败组揭示“前向仍放电、反向已失效”的训练停滞模式。
- 调整 actor 更新频率能够在代表性案例中缓解失败模式。
- PLIF 时间常数呈现层间自适应。
- LIF 与 PLIF 在成功训练状态下具有机制近似等价性。

理论价值：
- 明确可学习时间常数、放电重置和代理梯度传播之间的耦合关系。
- 将 LIF 与 PLIF 的比较从最终回报优劣推进到训练机制解释层面。

应用价值：
- 提供一组可操作的 SNN actor 训练状态诊断指标。
- 为 actor 更新频率、神经元时间常数和相关参数调节提供依据。

讲述重点：
- PLIF 的价值不是“必然提高回报”，而是提供可解释的时间尺度调节机制。

## 16. 不足与展望

页面内容：
- 实验任务和随机种子数量仍可扩展。
- Hopper-v4 失败机制还需要在更多环境中验证。
- 超参数交互关系尚未完整网格化分析。
- 尚未直接评估硬件能耗和神经形态部署收益。

未来工作：
- 扩展任务、随机种子和训练阶段分析。
- 建立更系统的 Proxy Target + SNN actor 训练稳定性判据。
- 进一步比较 ANN、LIF、PLIF 在相同性能下的放电率、延迟和理论能耗。

讲述重点：
- 主动说明本文不做硬件能耗实测，避免答辩问题偏到部署评估。

## 17. 致谢 / Q&A

页面内容：
- 感谢指导教师、课题组和答辩专家。
- Q&A。

讲述重点：
- 简短收束，不再展开技术内容。

## Beamer 章节建议

可在 LaTeX Beamer 中组织为以下 sections：

```latex
\section{研究问题与核心矛盾}
\section{方法框架与实验设计}
\section{实验结果与机制诊断}
\section{结论与展望}
```

推荐 frame 顺序：

```latex
\begin{frame}{研究对象：Proxy Target 脉冲 Actor}
\end{frame}

\begin{frame}{关键矛盾：PLIF 并未稳定更强}
\end{frame}

\begin{frame}{核心问题与贡献}
\end{frame}

\begin{frame}{Proxy Target + TD3 框架}
\end{frame}

\begin{frame}{LIF 与 PLIF 脉冲 Actor}
\end{frame}

\begin{frame}{实验设计}
\end{frame}

\begin{frame}{总体实验结果}
\end{frame}

\begin{frame}{学习曲线与失败现象}
\end{frame}

\begin{frame}{Hopper-v4 失败现象与缓解}
\end{frame}

\begin{frame}{STBP 梯度塌缩诊断}
\end{frame}

\begin{frame}{放电状态诊断}
\end{frame}

\begin{frame}{PLIF 时间常数演化}
\end{frame}

\begin{frame}{LIF 与 PLIF 机制近似等价}
\end{frame}

\begin{frame}{结论与价值}
\end{frame}
```
