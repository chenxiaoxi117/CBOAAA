# CBO sigma 校准与采集解耦修改说明 V1.2

## 修改目的

保留 calibration buffer、残差统计和校准诊断，但不再默认让 `sigma_calibrated` 替代 GP 的 `raw_sigma` 参与候选评分。

GP 输入、训练数据、external gate、internal_pressure6 和风险模型均未改变。

## 新参数

```text
cbo_sigma_calibration_use_in_acq = false | soft | true
cbo_sigma_calibration_eta = 0.25
```

默认值：

```text
cbo_sigma_calibration_use_in_acq = false
cbo_sigma_calibration_eta = 0.25
```

## 三种采集模式

### false：默认诊断模式

```text
sigma_acq = raw_sigma
score = mu + beta_eff * sigma_acq
```

calibration buffer 继续更新，仍输出 `sigma_scale`、`sigma_calibrated` 和 `calibrated_surprise`，但校准结果不进入采集函数。当前主方法为 fixed-beta greedy-mean，因此最终仍按 `argmax(mu)` 选点，与 `calibration=off` 保持同一部署行为；`score` 和 raw sigma 仅保留为候选诊断。

### soft：部分校准模式

```text
sigma_acq = raw_sigma + eta * (sigma_calibrated - raw_sigma)
score = mu + beta_eff * sigma_acq
```

默认 `eta=0.25`。`eta=0` 等价于 raw sigma，`eta=1` 等价于完全校准。

### true：完全校准消融

```text
sigma_acq = sigma_calibrated
score = mu + beta_eff * sigma_acq
```

该模式保留用于复现 sigma calibration 全量参与采集的负向消融结果，不再作为默认主配置。

## 新增诊断字段

```text
sigma_acq
sigma_calibration_use_in_acq
sigma_calibration_eta
sigma_acq_formula
```

原有字段全部保留：

```text
predicted_cost
actual_cost
prediction_error
raw_sigma
sigma_scale
sigma_calibrated
raw_surprise
calibrated_surprise
calibration_buffer_size
```

## 推荐命令

主实验：

```text
--cbo-sigma-calibration on
--cbo-sigma-calibration-use-in-acq false
--cbo-sigma-calibration-eta 0.25
```

soft 消融：

```text
--cbo-sigma-calibration on
--cbo-sigma-calibration-use-in-acq soft
--cbo-sigma-calibration-eta 0.25
```

完全校准消融：

```text
--cbo-sigma-calibration on
--cbo-sigma-calibration-use-in-acq true
```

彻底关闭 calibration buffer：

```text
--cbo-sigma-calibration off
```

注意：`calibration=off` 与 `calibration=on,use_in_acq=false` 不完全相同。前者不维护校准 buffer；后者维护诊断，并在采集时使用 raw sigma。
