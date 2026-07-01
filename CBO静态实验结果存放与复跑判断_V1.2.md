# CBO 静态实验结果存放与复跑判断 V1.2

更新时间：2026-06-28

## 1. 主索引文档

旧版完整索引仍保存在：

```text
D:\CBOv2\CBO静态108_ABCD服务器结果存放文档_V1.1.md
```

本文件补充已经完成的 V17 full108 数据，并说明 V1.9 动态分层复用修改后是否需要重跑静态实验。

## 2. 服务器静态原始结果位置

### 基线、直接调度和固定权重

```text
/home/ecs-user/CBO/result/static108_v11_sigma_calibrated_s43
```

包含 108 场景的：

```text
direct_greedy_cost
direct_queue_aware_greedy
reduced7_fixed_mid
reduced7_fixed_tuned
reduced7_bo_greedy
reduced7_cbo_lite_pressure_taskmix_counts
```

### BO 与旧 CBO 多种子结果

```text
/home/ecs-user/CBO/result/static108_v15_adaptive_core_s43
/home/ecs-user/CBO/result/static108_v15_adaptive_core_s44
/home/ecs-user/CBO/result/static108_v15_adaptive_core_s45
```

每个目录均有 108 场景的 `reduced7_bo_greedy` 和当时完整 CBO。

### BO adaptive 与旧 CBO 消融

```text
/home/ecs-user/CBO/result/static108_v16_bc_s43
```

包含 108 场景的：

```text
reduced7_bo_adaptive
reduced7_cbo_lite_pressure_taskmix_counts
```

### V17 新内部情景结构 selected12

```text
/home/ecs-user/CBO/result/static_v17_internal_context_selected12_newmods_s43
```

包含 D4、D6C、D4C 三种方法，每种 12 个代表场景。

### V17 新内部情景结构 full108，带 adaptive exploration

```text
/home/ecs-user/CBO/result/static108_v17_internal_context_newmods_s43
```

已完成 108 个场景、324 个轮次汇总文件：

```text
reduced7_cbo_lite_internal4
reduced7_cbo_lite_internal6_context
reduced7_cbo_lite_internal4_context
```

### V17 新内部情景结构 full108，不带 adaptive exploration

```text
/home/ecs-user/CBO/result/static108_v17_cmods_noadaptive_s43
```

已完成 108 个场景、432 个轮次汇总文件。该目录用于拆分内部情景结构与 adaptive exploration 的贡献。

## 3. 本地结果

本地 `D:\CBOv2\results` 主要保存早期 top5/window240 实验，不是服务器 full108 原始数据。代表目录包括：

```text
D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v8_fixed5
D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v8_fixed5_multiseed
D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v8_fixed5_cbo_window_sweep
D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v10_internal6_externalgate_s43
```

下载后的 full108 图表位于 `D:\CBOv2\图片` 下，但图表不能替代服务器上的逐轮 CSV 原始数据。

## 4. 是否需要重跑静态实验

当前不需要重跑全部静态实验。

V1.9 的 `phase_hierarchical_reuse`、阶段 archive、exact/similar/novel 关系判断和渐退复用，针对动态阶段切换设计。静态实验没有外部阶段切换，因此这些修改不使 V15-V17 的静态 BO、CBO、internal4、recent_context 和 adaptive exploration 结果失效。

以下基线不需要重跑：

```text
直接调度
fixed-mid / fixed-tuned
BO recent80
BO adaptive
旧 CBO
V17 D4 / D6C / D4C 结构消融
```

只有在论文把 V1.9 动态分层复用方法声明为“静态与动态统一的最终方法”时，才建议先补 12 个代表静态场景、3 个种子的兼容性检查。结果若与 V17 D4C 接近，再停止；出现明显差异后才有必要补 full108。V1.9 本身更适合作为动态扩展，不建议为了形式统一重跑所有旧基线。

## 5. 为什么内部情景在静态实验中只表现为后期小幅改善

1. 静态场景中外部任务强度和任务比例固定，普通 BO 的平稳黑箱假设本来就较合适，跨阶段负迁移问题几乎不存在。
2. BO 的 GP 输入是 7 维控制参数；internal4 CBO 的输入为 `theta(7) + context(4) = 11` 维。相同观测轮数下，CBO 前期更稀疏，通常需要更多样本才能学到可靠相关性。
3. 前 5 轮固定初始点相同。之后 BO 可以把最近 80 条都视为同一函数的观测；CBO 还要区分 backlog、利用率和未完成率状态，等价于把有限历史拆成多个局部状态邻域。
4. 内部情景是由泊松到达波动和上一轮调度共同产生的内生状态，不是独立外部变量。前期策略快速变化时，同一 context 邻域中的样本少，内部情景容易增加方差而不是立即提供收益。
5. 到后期积累了足够多相似内部状态后，CBO 才能减少“同一组 theta 在不同积压状态下被直接平均”的偏差，因此预测和选点可能略优于 BO。
6. 静态最优权重对内部状态的敏感度本身可能不高；若不同内部状态下的最优 theta 相近，那么即使 context 提高预测精度，最终调度评分也只会小幅改善。

现有 full108 多种子结果支持的是“前期付出样本复杂度，后期基本追平并出现很小改善”，而不是“静态 CBO 明显优于 BO”。selected12 困难场景出现过更明显的后期收益，但不能代替全 108 场景结论。

## 6. 下一步最有价值的验证

不先重跑 108 场景，先用 12 个代表场景、seed 43/44/45 做机制验证，同时记录：

```text
各分段 GP prediction MAE / RMSE
实际进入 GP 的样本数
selected_context_count 与 context similarity
同一 theta 邻域在不同 context 下的反馈方差
001-050、051-100、101-200、201-300、301-400、401-500
能耗、时延、unfinished rate、normalized_tradeoff_score
```

判定逻辑：

- 若 CBO 后期预测误差下降且评分同步下降，说明内部情景有效，但需要积累局部样本。
- 若预测误差下降而评分不变，说明静态最优策略对内部状态不敏感，小收益是问题本身决定的。
- 若预测误差也不下降，说明 internal4 特征、相似度或样本筛选仍需修改。
- 若只在高压场景有效，应把内部情景定位为高压状态稳健机制，而不是所有静态场景的统一增益来源。

