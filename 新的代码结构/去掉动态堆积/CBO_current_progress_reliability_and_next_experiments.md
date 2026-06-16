# CBO / reduced7 当前进度、实验结果、可靠性与后续计划

生成时间：2026-06-08

---

## 0. 当前一句话结论

当前实验已经从“模型与参数合理性解释”推进到“主线方法验证”阶段。

目前较稳的判断是：

```text
reduced7 可以作为后续主线控制空间；
W_Energy_Scale 主范围建议采用 0.5–2.0；
λ=1.0–3.0 已经能解释为从低负载到高压负载，但 λ=3.0 还不是严重过载；
后续应先跑 reduced7 主线静态对比与动态场景验证，107 场景可以作为后续正式扩展实验。
```

---

## 1. 当前主线方法状态

### 1.1 reduced6 的问题

reduced6 的控制参数主要是：

```text
W_RT_Latency
W_Batch_Latency
W_AI_Latency
W_Queue
W_Risk_Scale
Cloud_Gate
```

能耗权重固定。这可以解释为早期低维化设计，但存在一个明显问题：

```text
既然不同任务类型的时延权重可以学习，为什么能耗权重完全固定？
```

因此 reduced6 适合作为旧主线和消融对照，但不再建议作为最终主线。

---

### 1.2 alpha_direct 的定位

alpha_direct 的优点是解释性强：

```text
w_delay = alpha
w_energy = 1 - alpha
```

它很适合解释 delay-energy tradeoff，但当前 balanced / RT-heavy 静态实验中没有稳定优于 reduced7 / reduced6 主线。因此当前定位为：

```text
可解释 tradeoff 参数化消融
```

而不是主方法。

---

### 1.3 reduced9 的定位

reduced9 允许任务级能耗权重：

```text
W_RT_Energy
W_Batch_Energy
W_AI_Energy
```

理论上更灵活，但维度更高，500 轮 BO 下结果不够稳定。它适合作为：

```text
高维能耗参数化消融
```

而不是当前主线。

---

### 1.4 当前建议主线：reduced7

reduced7 的控制参数为：

```text
W_RT_Latency
W_Batch_Latency
W_AI_Latency
W_Queue
W_Risk_Scale
Cloud_Gate
W_Energy_Scale
```

它的核心逻辑是：

```text
reduced6 固定能耗权重，有解释风险；
reduced7 只增加一个全局能耗缩放参数；
reduced9 虽更灵活，但搜索维度更高、稳定性不足。
```

因此 reduced7 是一个“最小充分改动”：

```text
既承认能耗权重固定不合理；
又避免直接进入 9 维高维搜索。
```

---

## 2. 参数范围与解释性问题总结

### 2.1 为什么时延权重按任务类型分开？

RT / Batch / AI 对时延和 deadline 的敏感性不同：

```text
RT：强时延 / deadline 敏感；
Batch：时延容忍度较高；
AI：受计算量、加速器和能耗影响，介于二者之间。
```

因此保留：

```text
W_RT_Latency
W_Batch_Latency
W_AI_Latency
```

是合理的任务类型差异建模。

---

### 2.2 为什么能耗只用一个全局 W_Energy_Scale？

reduced7 不是认为所有任务能耗一样，而是认为：

```text
不同任务、不同节点的物理能耗差异已经体现在 E_ij 中；
W_Energy_Scale 学的是系统整体节能偏好。
```

因此 reduced7 的 score 可理解为：

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

这比 reduced6 更灵活，又比 reduced9 更稳定。

---

### 2.3 W_Energy_Scale 范围检验

已经检验过：

```text
narrow:  0.8–2.0
mid:     0.5–3.0
refined: 0.5–2.0
```

结论：

```text
narrow=0.8–2.0 下界偏高，基本淘汰；
mid=0.5–3.0 与 refined=0.5–2.0 多 seed 结果接近；
从 CBO 主方法平均表现和搜索空间紧凑性看，推荐 refined=0.5–2.0。
```

multi-seed 中 CBO 主方法大致表现：

```text
CBO mid 平均 final_roll50 ≈ 1.1601
CBO refined 平均 final_roll50 ≈ 1.1578
```

需要注意：

```text
refined 不是所有场景都更好；
例如 RT-heavy λ=2.6 中 mid 的 final rolling50 更稳；
但 refined 整体更紧凑，更适合作为主范围。
```

当前建议：

```text
W_Energy_Scale = 0.5–2.0
```

---

### 2.4 其他 6 个参数是否还要继续扫范围？

当前不建议继续逐个扫。原因是：

1. 参数已经能导出并审计；
2. 多个参数出现边界命中，不一定说明范围错误，也可能说明 BO 正在利用这些参数调节不同场景；
3. 当前主线不应继续发散到大范围调参。

全参数审计显示：

```text
W_Queue 经常贴近下界；
W_Risk_Scale 经常贴近上界；
部分 latency 权重贴近上界；
Cloud_Gate 有时贴低、有时贴高；
W_Energy_Scale 可审计，且 refined=0.5–2.0 可作为主范围。
```

其中最需要注意的是：

```text
selected_risk_penalty 基本为 0。
```

这说明 balanced / RT-heavy 静态场景没有真正激活 risk penalty，所以：

```text
W_Risk_Scale 不能说已经完全验证；
后续要放到 tight deadline / burst / dynamic stress 场景中验证。
```

---

## 3. common reference 与 normalized score 的可靠性

之前跨目录比较 normalized score 的问题在于：

```text
不同目录可能使用不同 reference；
不同 reference 下 normalized score 不能直接比较。
```

现在已经通过 common reference / scenario reference 机制解决：

```text
同一场景、同一 task mix、同一 λ 下，不同方法使用相同 reference；
因此 normalized_tradeoff_score 可以公平比较。
```

当前 reliable 的说法是：

```text
normalized score 的跨方法比较必须建立在相同场景 reference 上；
跨不同 task mix / λ 时可用于归一化汇总，但最好同时报告 raw 指标。
```

---

## 4. λ 任务强度标定总结

### 4.1 λ 的含义

λ 不应解释为绝对负载等级，而应解释为：

```text
任务到达强度缩放因子。
```

窗口到达任务数近似为：

```text
Expected arrivals ≈ λ × T
```

当前 T 约为 240，因此：

```text
λ=1.0 时，每窗口约 240 个任务；
λ=3.0 时，每窗口约 720 个任务。
```

实际标定结果也基本吻合。

---

### 4.2 balanced 场景标定结果

在 balanced 30/40/30 下，使用 reduced7_fixed_mid 标定：

| λ | window arrivals | Avg Util | Backlog | unfinished_rate | Avg Delay | SLA Success | 解释 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1.0 | ≈237.44 | ≈0.148 | ≈12.84 | ≈0.054 | ≈13.23 | ≈1.000 | 低负载 |
| 1.4 | ≈338.16 | ≈0.211 | ≈19.62 | ≈0.058 | ≈13.87 | ≈0.9998 | 低-中负载 |
| 1.8 | ≈434.48 | ≈0.263 | ≈25.24 | ≈0.058 | ≈14.15 | ≈0.9997 | 中负载 |
| 2.2 | ≈523.78 | ≈0.295 | ≈31.34 | ≈0.060 | ≈14.29 | ≈0.9996 | 中高负载 |
| 2.6 | ≈621.76 | ≈0.315 | ≈37.00 | ≈0.060 | ≈14.39 | ≈0.9997 | 高负载 |
| 3.0 | ≈718.32 | ≈0.336 | ≈42.54 | ≈0.059 | ≈14.34 | ≈0.9994 | 高压 / stress |

结论：

```text
λ 能有效提高系统到达压力；
但 λ=3.0 在 balanced 场景下还不是严重崩溃式过载。
```

因此建议命名：

```text
λ=1.0：低负载
λ=1.4：低-中负载
λ=1.8：中负载
λ=2.2：中高负载
λ=2.6：高负载
λ=3.0：高压 / stress
```

---

## 5. task mix 与负载解释

不能只用 λ 解释负载，因为任务比例会改变压力类型。

实验场景应定义为：

```text
scenario = (λ, task mix, window length)
```

其中：

```text
λ 控制单位时间到达强度；
task mix 控制任务结构；
window length 控制评价时间尺度。
```

不同 task mix 的含义：

```text
balanced 30/40/30：综合负载；
RT-heavy 60/30/10：deadline / 实时任务高压；
AI-heavy 10/20/70：计算 / 能耗高压；
Batch-heavy 10/70/20：吞吐 / 批任务压力。
```

因此不能说：

```text
所有 λ=2.6 都是同一种高负载。
```

更准确的说法是：

```text
在相同 task mix 内，λ 越大，输入到达压力越高；
不同 task mix 下，同一 λ 表示不同类型的系统压力。
```

---

## 6. 窗口长度解释

### 6.1 为什么固定窗口长度？

固定窗口长度 T 是为了保证不同 λ 下具有相同观测时间尺度。

如果让窗口长度随 λ 变化，例如：

```text
T = C / λ
```

那么每个窗口任务数可能差不多，但高 λ 不再表示“单位时间到达压力更大”，而只是改变了观测时间尺度。

因此主实验中应固定 T。

---

### 6.2 为什么选择 T=240？

可解释为统计稳定性和仿真成本之间的折中：

```text
T=240 时，λ=1.0 下约有 240 个任务；
即使某类任务比例只有 10%，也约有 24 个任务样本；
这能较稳定估计不同任务类型的延迟、能耗和完成情况。
```

相比：

```text
T=40：随机波动较大；
T=400：更平滑，但单轮仿真成本更高，并可能掩盖短期队列变化。
```

因此 T=240 是折中选择。

如果后续需要进一步增强解释性，可以补一个小型窗口敏感性实验：

```text
T = 200, 240, 400
λ = 1.8, 2.6
method = fixed_mid
task mix = 30/40/30
```

但当前不是第一优先级。

---

## 7. 当前已经解决的问题

### 已基本解决

```text
1. reduced7 为什么合理；
2. 为什么不用 alpha_direct 作为主线；
3. 为什么 reduced9 暂时作为消融；
4. W_Energy_Scale 范围为什么取 0.5–2.0；
5. normalized score 为什么必须统一 reference；
6. λ 是什么，以及如何标定低/中/高负载；
7. 为什么 task mix 要和 λ 一起解释负载；
8. 为什么 fixed window length 比随 λ 变化更合理。
```

### 仍需注意

```text
1. W_Risk_Scale 在当前静态实验中未被充分激活；
2. λ=3.0 不是严重过载，只能叫高压 / stress；
3. fixed_tuned 的来源还不够强，如果老师追问，后续需补 candidate search 或 fixed oracle；
4. CBO 的真正优势需要 dynamic 场景验证，而不是只靠静态场景；
5. 107 场景对比可以作为后续正式扩展，但不建议直接先做 200 候选 fixed oracle。
```

---

## 8. 后续实验建议

### 8.1 是否现在跑 107 场景对比？

建议：

```text
不要马上直接上 107 全场景。
```

更合理顺序是：

```text
1. 先跑 4 个 task mix × 3 个 λ × 主要方法 × seed=43；
2. 看 reduced7 主线是否在 balanced / RT-heavy / AI-heavy / Batch-heavy 下表现稳定；
3. 再补 seed=42/44；
4. 然后再扩展到 107 场景。
```

原因：

```text
107 场景太大；
如果当前主线方法、baseline 或 reference 口径还有问题，会浪费大量算力；
先用代表性场景验证方向更稳。
```

---

### 8.2 推荐下一步静态主线实验

任务结构：

```text
balanced:    30,40,30
RT-heavy:    60,30,10
AI-heavy:    10,20,70
Batch-heavy: 10,70,20
```

λ：

```text
1.8, 2.6, 3.0
```

方法：

```text
direct_greedy_cost
direct_queue_aware_greedy
reduced7_fixed_mid
reduced7_fixed_tuned
reduced7_bo_greedy
reduced7_cbo
```

先跑：

```text
seed = 43
```

如果结果合理，再补：

```text
seed = 42, 44
```

---

### 8.3 动态场景实验

动态场景用于验证 CBO 的 context / history 价值。

推荐动态 schedule：

```text
1.8:30,40,30:100;
2.6:60,30,10:100;
2.6:10,20,70:100;
1.8:30,40,30:100;
3.0:60,30,10:100
```

方法：

```text
reduced7_bo_greedy
reduced7_cbo
reduced7_cbo_no_context / pressure_only / prev_unfinished（如果支持）
```

重点看：

```text
phase 切换后的恢复速度；
rolling50 反弹；
backlog / unfinished；
CBO 是否比 BO-greedy 更快适应新 phase。
```

---

### 8.4 fixed baseline / 7D 搜索

当前代码不支持 `--reduced7-fixed-theta` 或 `--fixed-theta-file`，因此现在不建议卡在 7D fixed candidate search。

如果后面确实需要补 fixed_tuned 来源，建议最小改动是新增：

```text
--reduced7-fixed-theta "wrt,wbatch,wai,wqueue,wrisk,cloud,wenergy"
```

然后做：

```text
100 或 200 个 Sobol / Latin Hypercube 候选；
在少量训练场景上选 global fixed_tuned；
再拿到测试场景评估。
```

注意：

```text
200 候选 × 每场景选最优 = fixed oracle；
它不能作为公平 baseline，只能作为诊断上界。
```

---

## 9. 当前最合理实验路线图

```text
阶段 1：参数与负载合理性验证
已基本完成：
- reduced7 参数范围
- W_Energy_Scale 多 seed
- 全参数边界审计
- λ 标定

阶段 2：静态主线验证
下一步：
- 4 task mix × 3 λ × 主要方法 × seed43
- 再补 seed42/44

阶段 3：动态验证
- 任务比例和 λ 阶段切换
- 验证 CBO context / history

阶段 4：扩展验证
- 107 场景对比
- fixed candidate scan / oracle diagnostic
- window length sensitivity（可选）
- risk stress / tight deadline（可选）
```

---

## 10. 当前最短回答

```text
当前问题基本解释清楚了：
reduced7 主线合理；
W_Energy_Scale=0.5–2.0 有实验依据；
λ 是到达强度缩放，并已用固定策略标定；
task mix 必须和 λ 一起解释负载；
T=240 是固定评价窗口，在统计稳定性和仿真成本间折中。

但当前还不是最终论文主实验闭环。
下一步不建议立刻跑 107 场景，也不建议现在做 7D 网格搜索。
建议先跑 4 个代表性 task mix × 3 个 λ × 主要方法的 reduced7 静态主线实验，然后跑动态场景验证 CBO context。
107 场景作为后续扩展实验。
```
