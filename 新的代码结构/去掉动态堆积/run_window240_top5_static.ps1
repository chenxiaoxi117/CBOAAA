$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputRoot = "D:\CBOv2\results\window240_top5_static_rerun"
$SelectedKeys = "direct_greedy_cost,direct_queue_aware_greedy,reduced7_fixed_mid,reduced7_fixed_tuned,reduced7_bo_greedy,reduced7_cbo"
$CommonArgs = @(
    "-m", "new_tr_split",
    "--mode", "pressure_scan",
    "--selected-keys", $SelectedKeys,
    "--bo-iterations", "500",
    "--bo-interval", "240",
    "--session-duration", "120000",
    "--fixed-rng",
    "--fixed-seed", "43",
    "--reduced7-energy-scale-bounds", "0.5,3.0",
    "--feedback-score", "task_effective_backlog_violation",
    "--bo-history-mode", "recent",
    "--bo-recent-window", "80"
)

$Scenarios = @(
    @{ Name = "lambda3p0_rt10_batch40_ai50"; Lambda = "3.0"; Probs = "10,40,50" },
    @{ Name = "lambda3p0_rt30_batch60_ai10"; Lambda = "3.0"; Probs = "30,60,10" },
    @{ Name = "lambda3p0_rt10_batch30_ai60"; Lambda = "3.0"; Probs = "10,30,60" },
    @{ Name = "lambda2p6_rt20_batch10_ai70"; Lambda = "2.6"; Probs = "20,10,70" },
    @{ Name = "lambda2p6_rt70_batch20_ai10"; Lambda = "2.6"; Probs = "70,20,10" }
)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
Set-Location $ProjectRoot

$statusPath = Join-Path $OutputRoot "run_status.csv"
"scenario,lambda,task_probs,status,start_time,end_time,exit_code,output_dir" | Set-Content -LiteralPath $statusPath -Encoding UTF8

foreach ($s in $Scenarios) {
    $sceneOut = Join-Path $OutputRoot $s.Name
    New-Item -ItemType Directory -Force -Path $sceneOut | Out-Null
    $start = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$($s.Name),$($s.Lambda),$($s.Probs),running,$start,,,$sceneOut" | Add-Content -LiteralPath $statusPath -Encoding UTF8

    $logPath = Join-Path $sceneOut "run.log"
    $args = $CommonArgs + @(
        "--lambda-values", $s.Lambda,
        "--task-probs", $s.Probs,
        "--output-root", $sceneOut
    )

    & python @args *>&1 | Tee-Object -FilePath $logPath
    $exit = $LASTEXITCODE
    $end = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $status = if ($exit -eq 0) { "done" } else { "failed" }
    "$($s.Name),$($s.Lambda),$($s.Probs),$status,$start,$end,$exit,$sceneOut" | Add-Content -LiteralPath $statusPath -Encoding UTF8
    if ($exit -ne 0) {
        throw "Scenario $($s.Name) failed with exit code $exit"
    }
}
