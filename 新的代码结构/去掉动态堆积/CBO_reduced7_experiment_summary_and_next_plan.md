# CBO reduced7 当前实验阶段总结与后续实验计划

生成时间：2026-06-08

## 1. 当前阶段总判断

当前实验已经从“方法结构探索”进入到“主线控制空间收敛”阶段。

目前最重要的结论是：

```text
reduced7 是当前最值得继续作为主线候选的控制空间。
```

原因是：

1. `reduced6` 的固定能耗权重确实存在解释风险；
2. `alpha_direct` 虽然可解释性强，但在 balanced 和 RT-heavy 静态场景下没有稳定优于 reduced6；
3. `reduced9` 虽然表达能力更强，但 9 维搜索空间更大，当前结果不够稳定；
4. `reduced7` 只在 reduced6 基础上增加一个全局能耗缩放参数 `W_Energy_Scale`，既解决了“能耗权重固定”的质疑，又没有显著增加 BO 搜索难度。

因此当前主线建议为：

```text
主线候选：reduced7
高维消融：reduced9
解释性消融：alpha_direct
旧主线对照：reduced6
```

---

## 2. 参数选择范围目前是否都能解释？

基本可以解释，但要区分“已经可以写清楚”和“还需要最终多 seed 确认”。

### 2.1 已经比较稳的解释

#### 1）为什么时延按任务类型设置参数？

RT、Batch、AI 三类任务对时延和 deadline 的敏感性不同：

```text
RT：强时延敏感；
Batch：时延容忍度高；
AI：介于 RT 与 Batch 之间，且受算力和加速器影响。
```

所以 reduced6 / reduced7 / reduced9 中保留：

```text
W_RT_Latency
W_Batch_Latency
W_AI_Latency
```

这是合理的任务类型差异建模。

#### 2）为什么 reduced6 的能耗固定权重有局限？

reduced6 中能耗权重固定，本质是一个低维化假设：

```text
任务差异主要由时延权重表示；
能耗先作为系统级统一偏好固定；
这样可以减少 BO 维度，提高收敛稳定性。
```

这个假设可以解释，但会被质疑：

```text
既然时延可以按任务类型学习，为什么能耗完全固定？
```

因此后续引入 reduced7 / reduced9 是合理的。

#### 3）为什么选择 reduced7？

reduced7 的核心思想是：

```text
物理能耗差异由 E_ij 表达；
BO/CBO 只学习系统层面的整体节能偏好 W_Energy_Scale。
```

公式可理解为：

```text
Score_k(j)
= W_k_Latency * L_kj
+ W_Energy_Scale * E_kj
+ W_Queue * Q_j
+ W_Risk * R_kj
+ Cloud_Gate / Cloud_Penalty
```

其中：

```text
k ∈ {RT, Batch, AI}
```

`E_kj` 已经包含任务工作量、节点功耗、传输能耗、节点类型差异等信息，因此 `W_Energy_Scale` 不是说所有任务能耗一样，而是表示系统当前整体有多重视节能。

这就是 reduced7 的“最小充分改动”逻辑：

```text
reduced6：能耗固定，解释风险较高；
reduced7：增加一个全局能耗缩放，只增加 1 维；
reduced9：三类任务能耗权重都学习，维度更高，稳定性风险更大。
```

#### 4）为什么 reduced9 暂不作为主线？

reduced9 引入：

```text
W_RT_Energy
W_Batch_Energy
W_AI_Energy
```

理论上更灵活，但会带来：

```text
1. 维度从 7 增加到 9；
2. 500 轮 BO 下更难稳定收敛；
3. 参数解释更复杂；
4. 当前结果没有稳定超过 reduced7。
```

所以 reduced9 更适合作为高维能耗参数化消融，而不是当前主线。

---

## 3. 当前实验结果总结

### 3.1 reduced7 相比 reduced6 的结果

在统一 common reference 的 balanced 与 RT-heavy 静态实验中，reduced7 整体表现优于或接近 reduced6。

典型现象：

```text
balanced λ=1.8:
reduced7_cbo 优于 reduced6_cbo

balanced λ=2.6:
reduced7_cbo 优于 reduced6_cbo，并且接近或优于 reduced6_bo

RT-heavy λ=2.6:
reduced7_bo / reduced7_cbo 明显优于 reduced6_bo / reduced6_cbo
```

这说明：

```text
固定能耗权重确实可能限制 reduced6；
引入全局能耗缩放是有价值的。
```

### 3.2 alpha_direct 的结果

alpha_direct 的设计是：

```text
w_delay = alpha
w_energy = 1 - alpha
```

它的优点是 delay-energy tradeoff 解释非常清楚。

但当前结果显示：

```text
balanced 静态场景：alpha_direct 没有稳定超过 reduced6；
RT-heavy 静态场景：alpha_direct 仍然没有稳定超过 reduced7 / reduced6 主线；
risk / no-risk 经常完全一致，说明 risk penalty 在这些静态场景中没有真正触发。
```

因此 alpha_direct 当前定位为：

```text
可解释重参数化消融；
不作为主线。
```

### 3.3 reduced9 的结果

reduced9 有时接近或略优，但不稳定。

典型现象：

```text
RT-heavy λ=1.8:
reduced9_bo 表现很好；

RT-heavy λ=2.6:
reduced9_bo 明显回落，不如 reduced7_bo / reduced7_cbo。
```

说明：

```text
任务级能耗权重可能有潜在价值；
但高维搜索在 500 轮下不够稳定；
当前不建议直接作为主线。
```

---

## 4. W_Energy_Scale 参数范围检验结果

已经完成三类范围中的两类和 refined：

```text
narrow:  0.8 - 2.0
mid:     0.5 - 3.0
refined: 0.5 - 2.0
```

wide：

```text
0.2 - 4.0
```

未完整跑完，但目前优先级不高。

### 4.1 narrow 的问题

narrow 的下界是 0.8。

实验中多个场景的 `W_Energy_Scale` 接近 0.8 下界，例如：

```text
balanced λ=1.8 CBO: W_Energy_Scale ≈ 0.82
balanced λ=2.6 BO:  W_Energy_Scale ≈ 0.84
RT-heavy λ=2.6 CBO: W_Energy_Scale ≈ 0.87
```

这说明：

```text
0.8 的下界偏高，可能限制了优化器降低能耗项影响。
```

因此 narrow 不建议作为最终主范围。

### 4.2 mid 与 refined 的比较

从 CBO 主方法看，四个场景平均 final rolling50 约为：

```text
narrow:  1.1651
mid:     1.1599
refined: 1.1600
```

所以：

```text
mid 和 refined 非常接近；
mid 略优一点；
refined 更窄、更容易解释。
```

典型现象：

```text
balanced λ=2.6:
refined CBO 最好，final rolling50 ≈ 1.175342；

RT-heavy λ=2.6:
mid CBO 最好，final rolling50 ≈ 1.111972；
refined CBO 退到 ≈ 1.137581。
```

因此当前不能一锤定音地说 refined 更好。

### 4.3 当前范围选择建议

当前最合理的状态是：

```text
临时主范围：W_Energy_Scale = 0.5 - 3.0
候选收窄范围：W_Energy_Scale = 0.5 - 2.0
淘汰范围：W_Energy_Scale = 0.8 - 2.0
```

如果后续 multi-seed 结果显示 `0.5–2.0` 与 `0.5–3.0` 基本相当，则优先选择 refined：

```text
W_Energy_Scale = 0.5 - 2.0
```

因为它更窄，BO 搜索更集中，也更容易在论文中解释。

---

## 5. 当前实验可靠性评价

当前实验已经可以支撑“方法方向判断”，但还不能作为最终论文结论。

### 5.1 已经可靠的部分

```text
1. common reference 机制已经解决跨目录 normalized score 不可比的问题；
2. reduced7 相比 reduced6 的优势已经在 balanced 和 RT-heavy 中初步体现；
3. narrow / mid / refined 的参数范围行为已经能解释；
4. reduced9 不稳定、alpha_direct 暂不作为主线的结论比较清楚。
```

### 5.2 仍需补强的部分

```text
1. 当前大多数结果还是 seed=43；
2. reduced7 的最终 W_Energy_Scale 范围需要 multi-seed 确认；
3. fixed_tuned 需要针对 reduced7 新空间重新调；
4. 任务强度 λ 的来源需要负载标定；
5. CBO context 的优势仍需在动态场景中验证；
6. risk 模块需要 stress / tight deadline / burst 场景验证。
```

---

## 6. 后续实验路线

### 阶段 A：完成 reduced7 参数范围最终确认

下一步建议只比较：

```text
mid:     W_Energy_Scale = 0.5 - 3.0
refined: W_Energy_Scale = 0.5 - 2.0
```

场景：

```text
balanced:
λ = 1.8, 2.6
task mix = 30/40/30

RT-heavy:
λ = 1.8, 2.6
task mix = 60/30/10
```

方法：

```text
reduced7-bo-greedy
reduced7-cbo
```

seed：

```text
42, 43, 44
```

目标：

```text
判断 0.5–2.0 是否能替代 0.5–3.0；
如果二者接近，选择 0.5–2.0；
如果 0.5–3.0 多 seed 稳定更好，则保留 0.5–3.0。
```

### 阶段 B：任务强度标定实验

是的，后面应该开始测试任务强度。

但不是直接大规模跑方法对比，而是先做负载标定，解释：

```text
λ=1.0 / 1.8 / 2.6 / 3.0 为什么代表低 / 中 / 高 / 过载？
```

建议使用 fixed_mid 或 fixed_tuned 作为标定策略，跑：

```text
task mix = 30/40/30
λ = 1.0, 1.4, 1.8, 2.2, 2.6, 3.0
```

统计：

```text
平均到达任务数
完成率
unfinished_rate
backlog_growth_rate
平均时延
RT violation rate
energy_per_arrival
节点利用率 / queue pressure
```

最终形成一个负载标定表：

| λ | 到达量 | 完成率 | unfinished_rate | backlog_growth | 平均时延 | RT violation | 场景解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1.0 | ... | ... | ... | ... | ... | ... | low load |
| 1.8 | ... | ... | ... | ... | ... | ... | medium load |
| 2.6 | ... | ... | ... | ... | ... | ... | high load |
| 3.0 | ... | ... | ... | ... | ... | ... | overload / stress |

这一步很重要，因为它能解决“λ 来源不明”的问题。

### 阶段 C：reduced7 主线多 seed 验证

当范围确定后，正式验证 reduced7 主线。

建议方法：

```text
reduced6-bo-greedy
reduced6-cbo
reduced7-bo-greedy
reduced7-cbo
fixed_tuned
direct greedy
queue-aware greedy
```

场景：

```text
balanced
RT-heavy
AI-heavy
```

强度：

```text
λ=1.8, 2.6
必要时加 λ=3.0 stress
```

seed：

```text
42, 43, 44
```

输出：

```text
mean ± std
final rolling50
best rolling50
rebound
Avg Delay
Avg RT Delay
energy_per_arrival
unfinished_rate
backlog_growth_rate
RT violation rate
```

### 阶段 D：reduced7 fixed_tuned 重调

如果 reduced7 被确定为主线，则必须重新调 fixed_tuned：

```text
reduced7_fixed_tuned
```

否则会被质疑：

```text
BO/CBO 是 reduced7 搜索空间；
fixed_tuned 却还是 reduced6 / old bounds 口径；
比较不公平。
```

fixed_tuned 应该基于：

```text
相同 objective
相同 common reference
相同 W_Energy_Scale bounds
相同任务场景
```

重新搜索或网格调参。

### 阶段 E：动态场景与 CBO context 验证

目前静态场景下 CBO 不一定稳定优于 BO-greedy。

所以 CBO 的真正优势应放到动态场景验证：

```text
任务比例切换；
λ 强度变化；
资源扰动；
链路扰动；
AI accelerator 可用性变化；
阶段 reference 切换。
```

重点比较：

```text
reduced7-bo-greedy
reduced7-cbo
reduced7-cbo-no-context
reduced7-cbo-prev-unfinished
```

看：

```text
切换后恢复速度；
rolling50 反弹；
phase reference 是否正确切换；
CBO 是否比 BO 更快适应新 phase。
```

---

## 7. 论文叙事建议

当前可以形成的论文实验叙事是：

```text
1. 首先提出 reduced6 低维控制空间；
2. 发现固定能耗权重存在解释局限；
3. 引入 reduced7 全局能耗缩放和 reduced9 任务级能耗权重；
4. 统一 reference 后比较发现 reduced7 更稳定，reduced9 高维但不稳定；
5. 对 W_Energy_Scale 做范围敏感性分析，排除下界过高的 narrow；
6. 在 mid/refined 中通过 multi-seed 进一步确定最终范围；
7. 用任务强度标定解释 λ 设置；
8. 最后用 reduced7 主线做多 seed、多任务结构、动态场景验证。
```

---

## 8. 当前最短结论

当前可以总结为：

```text
所有主要参数范围已经具备解释框架，但 reduced7 的 W_Energy_Scale 最终范围还需要在 0.5–3.0 与 0.5–2.0 之间做 multi-seed 确认。narrow=0.8–2.0 因下界偏高基本淘汰。下一阶段应先做 reduced7 范围 multi-seed 验证，然后进行任务强度 λ 的负载标定实验，用完成率、unfinished_rate、backlog_growth_rate 和时延等指标解释 λ=1.0/1.8/2.6/3.0 的来源。
```
