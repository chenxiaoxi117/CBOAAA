$ErrorActionPreference = "Stop"

$ROOT = "D:\CBOv2\新的代码结构\去掉动态堆积"
$OUT  = "D:\CBOv2\results\window240_top5_static_cbo_only"
$KEY  = "reduced7_cbo_lite_pressure_taskmix_counts"

New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path "$OUT\logs" | Out-Null

Set-Location $ROOT

$cases = @(
    @{ lam = "3.0"; probs = "10,40,50"; name = "lambda3p0_rt10_batch40_ai50" },
    @{ lam = "3.0"; probs = "30,60,10"; name = "lambda3p0_rt30_batch60_ai10" },
    @{ lam = "3.0"; probs = "10,30,60"; name = "lambda3p0_rt10_batch30_ai60" },
    @{ lam = "2.6"; probs = "20,10,70"; name = "lambda2p6_rt20_batch10_ai70" },
    @{ lam = "2.6"; probs = "70,20,10"; name = "lambda2p6_rt70_batch20_ai10" }
)

Write-Host "============================================================"
Write-Host "Start CBO-only window240 top5 static rerun"
Write-Host "ROOT = $ROOT"
Write-Host "OUT  = $OUT"
Write-Host "KEY  = $KEY"
Write-Host "============================================================"

foreach ($case in $cases) {
    $name = $case.name
    $lam = $case.lam
    $probs = $case.probs
    $log = "$OUT\logs\$name.log"
    $caseOut = "$OUT\$name"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Running $name"
    Write-Host "lambda = $lam"
    Write-Host "task_probs = $probs"
    Write-Host "log = $log"
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
        "--cbo-reference-calibrationive_backlog_violation",
        "--bo-history-mode", "recent",
        "--bo-recent-window", "80",
        "--cbo-objective-mode-rounds", "30",
        "--scheduler-score-norm-mode", "candidate_minmax_deadline",
        "--task-adaptation",
        "--lambda-values", $lam,
        "--task-probs", $probs,
        "--output-root", $caseOut
    )

    & python @args *> $log

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[FAILED] $name"
        Write-Host "Last log lines:"
        Get-Content $log -Tail 80
        throw "Case failed: $name"
    }

    Write-Host "[OK] $name"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "All CBO-only tests finished."
Write-Host "Result root:"
Write-Host $OUT
Write-Host "============================================================"

$count = (Get-ChildItem $OUT -Recurse -Filter "*round_summary_轮次汇总.csv").Count
Write-Host "round_summary count = $count"
