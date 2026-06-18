# CBO 多种子稳健性实验结论 V1.0

## 1. 本轮实验检验的问题

本轮实验主要检验的是：

1. 在相同静态困难场景下，CBO 是否能够在训练后期稳定超过 BO。
2. 之前单 seed 结果中 CBO 后期优于 BO，是否只是随机性导致。
3. 固定前 5 轮初始探索点后，BO 与 CBO 是否具有公平的冷启动对比基础。
4. 在不启用 Trust Region 的情况下，CBO 的情景信息是否能在后期带来收益。

本轮实验不用于证明：

1. CBO 从训练一开始就优于 BO。
2. CBO 在所有指标上都严格优于 BO。
3. CBO 在所有场景、所有 seed 下都必胜。
4. `recent_confidence` 不同窗口大小的影响。

## 2. 当前实验配置

结果目录：

`D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v8_fixed5_multiseed`

代码目录：

`D:\CBOv2\新的代码结构\去掉动态堆积`

方法：

- BO：`reduced7_bo_greedy`
- CBO：`reduced7_cbo_lite_pressure_taskmix_counts`

核心设置：

- 静态困难场景 top5
- seed：`40, 41, 42, 43, 44`
- 每个 seed 跑 5 个场景
- 每个场景同时比较 BO 与 CBO
- 总计：`25` 个场景任务，`50` 个方法结果
- 总轮数：`500`
- 情景窗口：`240`
- backlog growth penalty：已关闭
- Trust Region：关闭
- 初始探索：BO 与 CBO 使用相同 fixed5 初始点
- BO/CBO 历史窗口：本轮结果显示均为 `effective_history_mode = recent`，`effective_recent_window = 80`

## 3. 本轮使用的主评估函数

主比较口径使用：

`normalized_tradeoff_score`

这是当前 BO/CBO 训练与优化的主目标函数，综合考虑服务质量、能耗、未完成率、积压等归一化指标。

同时使用以下原始指标做 sanity check：

- `Eval_Cost`
- 平均时延
- RT 平均时延
- Batch 平均时延
- AI 平均时延
- `energy_per_arrival`
- `unfinished_rate`
- backlog
- violation rate
- SLA success rate

因此，主结论以 `normalized_tradeoff_score` 为准，但不会只依赖该归一化分数。

## 4. 完整性检查

本轮结果完整：

- `pressure_scan_summary_all.csv`：`25 / 25`
- BO/CBO round summary：`50 / 50`
- Python 进程：已结束
- stderr 中存在 `Mean of empty slice` 和绘图 legend 警告，但不是实验失败

前 5 轮冷启动检查：

- BO 与 CBO 的控制向量完全一致
- BO 与 CBO 的 deployed theta 完全一致
- BO 与 CBO 的 `Eval_Cost` 完全一致
- BO 与 CBO 的时延、能耗、积压等原始指标完全一致

因此，前 5 轮 fixed5 初始探索是公平的。

注意：前 5 轮的 `normalized_tradeoff_score` 不适合作为冷启动优劣判断，因为 CBO-first 归一化基准仍在建立中；冷启动公平性应该看原始指标和控制向量。

## 5. 主要结果

按 `normalized_tradeoff_score`，数值越低越好。

### 5.1 分阶段结果

| 阶段 | CBO 胜场 | BO 胜场 | 平均 CBO-BO | 相对变化 |
|---|---:|---:|---:|---:|
| first50 | 1/25 | 24/25 | +0.104138 | +8.13% |
| 51-100 | 2/25 | 23/25 | +0.105309 | +8.50% |
| 101-200 | 10/25 | 15/25 | +0.028733 | +2.37% |
| 201-300 | 14/25 | 11/25 | +0.004190 | +0.36% |
| 301-400 | 19/25 | 6/25 | -0.004695 | -0.39% |
| last100 | 18/25 | 7/25 | -0.006122 | -0.50% |
| last50 | 18/25 | 7/25 | -0.004914 | -0.40% |
| all500 | 4/25 | 21/25 | +0.024352 | +2.00% |

解释：

- CBO 在前期明显不如 BO。
- 约从 200 轮以后，CBO 逐渐追平。
- 约从 300 轮以后，CBO 开始稳定略优。
- 看全 500 轮均值时，CBO 会被前期学习成本拖累。

因此，本轮实验支持的结论是：

**CBO 在训练后期能够稳定略微超过 BO，但不能证明 CBO 全程优于 BO。**

### 5.2 last50 场景级结果

| 场景 | CBO 胜场 | 平均 CBO-BO | 相对变化 |
|---|---:|---:|---:|
| `lambda2p6_rt20_batch10_ai70` | 4/5 | -0.010143 | -0.81% |
| `lambda3p0_rt10_batch30_ai60` | 4/5 | -0.009920 | -0.82% |
| `lambda3p0_rt10_batch40_ai50` | 4/5 | -0.002561 | -0.20% |
| `lambda3p0_rt30_batch60_ai10` | 4/5 | -0.001583 | -0.13% |
| `lambda2p6_rt70_batch20_ai10` | 2/5 | -0.000364 | -0.03% |

解释：

- 其中 4 个场景中，CBO 在 5 个 seed 里赢了 4 个。
- `lambda2p6_rt70_batch20_ai10` 基本打平，优势很弱，稳定性不足。
- 当前结果最能说明 CBO 后期优势的场景是：
  - `lambda2p6_rt20_batch10_ai70`
  - `lambda3p0_rt10_batch30_ai60`

## 6. 指标层面的观察

last50 中，CBO 的指标表现如下：

| 指标 | CBO 胜场 | 平均 CBO-BO | 结论 |
|---|---:|---:|---|
| `normalized_tradeoff_score` | 18/25 | -0.004914 | CBO 后期更优 |
| `Eval_Cost` | 14/25 | -0.007353 | CBO 略优 |
| 平均时延 | 17/25 | -0.135397 | CBO 较好 |
| RT 时延 | 15/25 | -0.006658 | CBO 略好 |
| Batch 时延 | 12/25 | -0.043456 | 基本接近 |
| AI 时延 | 18/25 | -0.323508 | CBO 较好 |
| `energy_per_arrival` | 14/25 | -3.850408 | CBO 略好 |
| `unfinished_rate` | 15/25 | -0.000489 | CBO 略好 |
| backlog | 15/25 | -0.332000 | CBO 略好 |
| violation rate | 10/25 | +0.000013 | 无优势 |
| SLA success rate | 10/25 | -0.000013 | 无优势 |

解释：

- CBO 的后期收益主要体现在综合 score、平均时延、AI 时延、积压和未完成率上。
- 违约率和 SLA 本身已经非常接近饱和，因此没有明显优势。
- 能耗指标有轻微改善，但不是最主要收益来源。

## 7. 当前可以写入论文/报告的谨慎结论

推荐表述：

> 在固定初始探索点和多随机种子设置下，CBO 在训练早期存在额外学习成本，但随着历史样本积累，情景信息开始发挥作用。在 500 轮训练的后期窗口中，CBO 相比 BO 表现出更稳定的综合目标改善。具体而言，在 last50 窗口内，CBO 在 25 个 seed-scene 配对中赢得 18 个，平均 normalized tradeoff score 相对 BO 降低约 0.40%。这说明当前 CBO 方法并非依靠单次随机结果取胜，而是在后期具有一定稳定优势。

不推荐表述：

> CBO 全程优于 BO。

也不推荐表述：

> CBO 显著大幅优于 BO。

更准确的说法是：

> CBO 后期稳定略优，但优势幅度较小，且前期存在学习成本。

## 8. 当前结果的限制

1. 本轮 CBO 与 BO 的有效历史模式都是 `recent`，窗口都是 `80`。
2. 本轮尚未检验 `CBO recent_confidence window = 80/120/160/240` 的问题。
3. 本轮没有启用 Trust Region。
4. 后期优势幅度较小，平均约 `0.4%`，需要更多 seed 或更多场景进一步增强统计说服力。
5. `all500` 口径下 CBO 不优，因为 CBO 前期学习成本明显。
6. 当前结果更适合支持“后期稳定超过”，不适合支持“整体训练全过程超过”。

## 9. 后续建议

下一步建议重新运行或补充以下实验：

1. CBO-only recent window sweep：
   - CBO recent window = `80`
   - CBO recent window = `120`
   - CBO recent window = `160`
   - CBO recent window = `240`
   - BO 保持 recent window = `80`

2. 检查 CBO 记忆长度是否不足：
   - 如果 120/160 明显优于 80，说明 CBO 当前记忆窗口偏短。
   - 如果 240 不如 120/160，说明过长历史可能引入旧样本干扰。

3. 如需写论文级结论，建议增加：
   - 更多 seed
   - 置信区间
   - paired t-test 或 Wilcoxon signed-rank test
   - 分场景箱线图
   - 分阶段收敛曲线

## 10. V1.0 总结

本轮 V8 fixed5 multiseed 实验已经证明：

**在固定初始探索、公平冷启动、多 seed、静态困难场景下，CBO 虽然前期不如 BO，但在训练后期能够较稳定地超过 BO。**

当前证据强度：

- 支持“CBO 后期稳定略优”
- 支持“不是单 seed 偶然现象”
- 支持“fixed5 冷启动公平”
- 不支持“CBO 全程优于 BO”
- 不支持“CBO 大幅优于 BO”

