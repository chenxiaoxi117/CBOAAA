# CBO V10 参数范围与后验不确定性校准修改说明 V1.1

## 1. 修改目的

本次修改解决两个独立问题：

1. 依据 107/108 静态场景扫描中的参数贴边情况，调整 Reduced7 的推荐搜索范围。
2. 修正 CBO GP 后验标准差长期偏小、候选评分过度自信的问题。

GP 输入仍为 `theta(7) + internal_pressure6(6)`，维度、训练样本选择、外部门控和 internal_pressure6 均未改变；没有新增风险模型。

## 2. 推荐 Reduced7 搜索范围

| 参数 | 旧范围 | 新范围 | 调整依据 |
|---|---:|---:|---|
| `W_RT_Latency` / `W_Batch_Latency` / `W_AI_Latency` | `[0.1, 5.0]` | `[0.1, 7.0]` | 多个负载下延迟权重频繁命中上界，需要继续验证高权重区间 |
| `W_Queue` | `[0.0, 5.0]` | `[0.0, 3.0]` | BO/CBO 大量集中在低值侧，缩小无效高值空间 |
| `W_Risk_Scale` | `[0.0, 5.0]` | `[0.0, 8.0]` | 约半数结果命中旧上界，旧范围明显截断 |
| `Cloud_Gate` | `[0.05, 0.95]` | `[0.01, 0.95]` | 低边界命中较多，需要允许更积极的云端候选 |
| `W_Energy_Scale` | V10 脚本为 `[0.5, 3.0]` | `[0.25, 2.0]` | 结果主要集中在低值侧，旧上界利用率低 |

这些范围只用于 Reduced7。传统 full/reduced6 BO 的搜索范围保持原样，避免无关方法的实验口径变化。

## 3. 后验不确定性校准

### Ask 阶段

CBO 对最终选中候选保存：

- `predicted_cost = -posterior_mu`
- `raw_sigma = GP posterior sigma`
- 当轮使用的 `sigma_scale`
- `sigma_calibrated`

样本少于 10 条时：

```text
sigma_scale = 4.0
```

样本达到 10 条后：

```text
estimated_scale = RMSE(prediction_error) / RMS(raw_sigma)
history_weight = calibration_buffer_size / 50
sigma_scale = (1-history_weight)*4.0 + history_weight*estimated_scale
sigma_scale = clip(sigma_scale, 1.0, 6.0)
sigma_calibrated = max(raw_sigma*sigma_scale, 0.03)
```

候选评分使用 `sigma_calibrated`，不再直接使用 `raw_sigma`：

```text
score = posterior_mu + beta_eff * sigma_calibrated
```

### Tell 阶段

窗口结束后计算：

```text
prediction_error = actual_cost - predicted_cost
raw_surprise = prediction_error / raw_sigma
calibrated_surprise = prediction_error / sigma_calibrated
```

`(prediction_error, raw_sigma)` 写入 calibration buffer，最多保留最近 50 条。TR 残差诊断如启用，也统一使用 `calibrated_surprise`。

## 4. 诊断字段

逐轮结果新增：

- `predicted_cost`
- `actual_cost`
- `prediction_error`
- `raw_sigma`
- `sigma_scale`
- `sigma_calibrated`
- `raw_surprise`
- `calibrated_surprise`
- `calibration_buffer_size`

另保留 `sigma_scale_estimated`、`sigma_scale_history_weight`、`sigma_floor`，便于判断缩放是否长期卡在 `[1, 6]` 边界。

## 5. 推荐验证顺序

1. 先运行 V10 top5，确认 buffer 从 0 增长至 50，前 10 条 `sigma_scale=4.0`。
2. 对比校准前后 `abs(raw_surprise)` 与 `abs(calibrated_surprise)`，检查过度自信是否缓解。
3. 再运行完整 108 场景 seed 43，检查新范围的上下边界命中率。
4. 最后使用至少 3 个种子复验 BO/CBO 实际指标和后期 rolling50，不用单一种子下结论。

本机制只修正“不确定性量级和候选评分”，不保证单轮 cost 必然下降；有效性应以多种子实际指标、后期稳定性和校准误差共同判断。
