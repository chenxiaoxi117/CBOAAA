# CBO V1.3 CBO-First Reference Run Commands

## 1. Main Top5 BO+CBO Run

This is the recommended V1.3 rerun for the five old CBO-losing static scenes.

Key differences from `window240_nogrowth_top5_static_bo_cbo_v5`:

```text
--cbo-shared-reference-policy cbo_first
--cbo-shared-reference-warmup-rounds 5
--cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts
```

Meaning:

```text
Total BO budget remains 500 windows.
CBO runs first.
CBO uses the first 5 windows inside those 500 windows to define the reference.
BO then uses the CBO-derived shared reference.
The same scenario signature reuses the cached reference.
A new scenario signature triggers a new CBO warm-up reference.
```

Run in PowerShell:

```powershell
$ErrorActionPreference = "Continue"

$ROOT = "D:\CBOv2\新的代码结构\去掉动态堆积"
$OUT  = "D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v1_3_cbo_first"
$KEY  = "reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts"

New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT\logs" | Out-Null

$cases = @(
    @{ lam = "3.0"; probs = "10,40,50"; name = "lambda3p0_rt10_batch40_ai50" },
    @{ lam = "3.0"; probs = "30,60,10"; name = "lambda3p0_rt30_batch60_ai10" },
    @{ lam = "3.0"; probs = "10,30,60"; name = "lambda3p0_rt10_batch30_ai60" },
    @{ lam = "2.6"; probs = "20,10,70"; name = "lambda2p6_rt20_batch10_ai70" },
    @{ lam = "2.6"; probs = "70,20,10"; name = "lambda2p6_rt70_batch20_ai10" }
)

Set-Location $ROOT

Write-Host "============================================================"
Write-Host "Start BO+CBO top5 rerun | V1.3 CBO-first shared reference"
Write-Host "ROOT = $ROOT"
Write-Host "OUT  = $OUT"
Write-Host "KEY  = $KEY"
Write-Host "============================================================"

foreach ($case in $cases) {
    $name = $case.name
    $lam = $case.lam
    $probs = $case.probs

    $caseOut = "$OUT\$name"
    $stdout = "$OUT\logs\$name.stdout.log"
    $stderr = "$OUT\logs\$name.stderr.log"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running $name"
    Write-Host "lambda = $lam"
    Write-Host "task_probs = $probs"
    Write-Host "stdout = $stdout"
    Write-Host "stderr = $stderr"
    Write-Host "============================================================"

    $args = @(
        "-m", "new_tr_split",
        "--mode", "pressure_scan",
        "--selected-keys", $KEY,
        "--bo-iterations", "500",
        "--bo-interval", "240",
        "--session-duration", "120000",
        "--fixed-rng",
        "--fixed-seed", "43",
        "--reduced7-energy-scale-bounds", "0.5,3.0",
        "--feedback-score", "task_effective_backlog_violation",
        "--bo-history-mode", "recent",
        "--bo-recent-window", "80",
        "--cbo-objective-mode", "normalized_tradeoff",
        "--cbo-reference-mode", "calibrate",
        "--cbo-shared-reference-policy", "cbo_first",
        "--cbo-shared-reference-warmup-rounds", "5",
        "--cbo-reference-source-method-key", "reduced7_cbo_lite_pressure_taskmix_counts",
        "--cbo-backlog-growth-penalty-weight", "0",
        "--scheduler-score-norm-mode", "candidate_minmax_deadline",
        "--task-adaptation",
        "--lambda-values", $lam,
        "--task-probs", $probs,
        "--output-root", $caseOut
    )

    & python @args > $stdout 2> $stderr

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[FAILED] $name"
        Write-Host "ExitCode = $LASTEXITCODE"
        Write-Host "===== stderr tail ====="
        Get-Content $stderr -Tail 120
        Write-Host "===== stdout tail ====="
        Get-Content $stdout -Tail 80
        throw "Case failed: $name"
    }

    Write-Host "[OK] $name"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "All V1.3 BO+CBO tests finished."
Write-Host "Result root:"
Write-Host $OUT
Write-Host "============================================================"

$count = (Get-ChildItem $OUT -Recurse -Filter "*round_summary_轮次汇总.csv").Count
Write-Host "round_summary count = $count"
```

Expected final count:

```text
round_summary count = 10
```

## 2. Quick Smoke Test

Use this only to check whether the reference mechanism works. It is not a formal result.

```powershell
Set-Location "D:\CBOv2\新的代码结构\去掉动态堆积"

python -m new_tr_split `
  --mode pressure_scan `
  --selected-keys reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts `
  --bo-iterations 6 `
  --bo-interval 240 `
  --session-duration 3000 `
  --fixed-rng `
  --fixed-seed 43 `
  --reduced7-energy-scale-bounds "0.5,3.0" `
  --feedback-score task_effective_backlog_violation `
  --bo-history-mode recent `
  --bo-recent-window 80 `
  --cbo-objective-mode normalized_tradeoff `
  --cbo-reference-mode calibrate `
  --cbo-shared-reference-policy cbo_first `
  --cbo-shared-reference-warmup-rounds 5 `
  --cbo-reference-source-method-key reduced7_cbo_lite_pressure_taskmix_counts `
  --cbo-backlog-growth-penalty-weight 0 `
  --scheduler-score-norm-mode candidate_minmax_deadline `
  --task-adaptation `
  --lambda-values 2.6 `
  --task-probs 70,20,10 `
  --output-root "D:\CBOv2\results\quick_cbo_first_reference_smoke"
```

Expected log signals:

```text
[ScenarioReference] policy=cbo_first source=reduced7_cbo_lite_pressure_taskmix_counts
[ScenarioReference] CBO-first cached reference signature=...
[ScenarioReference] CBO-first published 1 shared reference(s) ...
[SCENARIO-REF] method=reduced7_bo_greedy ... cache_entries=1
```

In BO round summary, check:

```text
cbo_reference_status = cache_frozen
cbo_reference_available = 1
cbo_reference_frozen = 1
delay_ref / energy_norm_ref / backlog_ref are non-empty
```

## 3. Notes

Do not add `reduced7_fixed_mid` for this V1.3 run. That belongs to the old fixed-probe reference protocol.

Do not use `reduced7_cbo` as the method name here. Use the explicit current CBO key:

```text
reduced7_cbo_lite_pressure_taskmix_counts
```

The current dynamic backlog growth penalty is off:

```text
--cbo-backlog-growth-penalty-weight 0
```

The scenario statistics window is already aligned to:

```text
SCENARIO_WINDOW = 240
BO_INTERVAL = 240
```
