cd D:\CBOv2

$ErrorActionPreference = "Continue"

# ============================================================
# Denoise warm-start validation
# Method: reduced6_cbo_lite_pressure_only
# Denoise config: R050_M2_CW03
# Goal:
#   P1 RT60 -> RT50: can denoise keep early gain and reduce late regression?
#   P2 RT60 -> AI70: can denoise reduce negative transfer?
# ============================================================

$METHOD = "reduced6_cbo_lite_pressure_only"

$SOURCE_HISTORY = "D:\CBOv2\results\transfer_cbo_pressure\source_lam2p6_RT60_Batch30_AI10"

$SOURCE_WARM_FILE = Get-ChildItem $SOURCE_HISTORY -Recurse -Filter "bo_warm_history.csv" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($null -eq $SOURCE_WARM_FILE) {
    Write-Host "[ERROR] Cannot find bo_warm_history.csv under:"
    Write-Host $SOURCE_HISTORY
    exit 1
}

Write-Host "[OK] Using source warm history:"
Write-Host $SOURCE_WARM_FILE.FullName

$ROOT = "D:\CBOv2\results\transfer_cbo_pressure_denoise_R050_M2_CW03"
$LOGDIR = Join-Path $ROOT "logs"
$TARGET_ROOT = Join-Path $ROOT "targets"

New-Item -ItemType Directory -Force $ROOT | Out-Null
New-Item -ItemType Directory -Force $LOGDIR | Out-Null
New-Item -ItemType Directory -Force $TARGET_ROOT | Out-Null

$SEEDS = @(43, 44, 45)

$FAILED = @()
$MANIFEST = @()

function Has-RoundSummary {
    param(
        [string]$RunDir
    )

    if (-not (Test-Path $RunDir)) {
        return $false
    }

    $f = Get-ChildItem $RunDir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name.ToLower().Contains("round_summary") -and
            $_.FullName.ToLower() -notmatch "_short_export"
        } |
        Select-Object -First 1

    return ($null -ne $f)
}

function Run-CBOScenario {
    param(
        [string]$RunName,
        [string]$Lambda,
        [string]$TaskProbs,
        [int]$Seed,
        [string]$OutputRoot,
        [bool]$Warm,
        [string]$WarmHistoryDir,
        [string]$WarmLabel
    )

    $log = Join-Path $LOGDIR "$RunName.log"

    if (Has-RoundSummary -RunDir $OutputRoot) {
        Write-Host "[SKIP] Existing round_summary found for $RunName"
        $script:MANIFEST += [pscustomobject]@{
            run_name = $RunName
            lambda = $Lambda
            task_probs = $TaskProbs
            seed = $Seed
            warm = $Warm
            warm_history = $WarmHistoryDir
            output_root = $OutputRoot
            status = "skipped_existing"
            log = $log
        }
        return
    }

    Write-Host "============================================================"
    Write-Host "START $RunName"
    Write-Host "lambda=$Lambda task_probs=$TaskProbs seed=$Seed warm=$Warm"
    Write-Host "out=$OutputRoot"
    Write-Host "log=$log"
    Write-Host "============================================================"

    $argsList = @(
        "-W", "ignore::RuntimeWarning",
        "-m", "new_tr_split",
        "--mode", "scenario",
        "--fixed-rng",
        "--fixed-seed", "$Seed",
        "--lambda-schedule", "0:120000:$Lambda",
        "--session-duration", "120000",
        "--task-probs", "$TaskProbs",
        "--bo-iterations", "500",
        "--bo-interval", "240",
        "--selected-keys", "$METHOD",
        "--feedback-score", "window_original",
        "--cbo-reference-mode", "off",
        "--cbo-objective-mode", "eval_cost",

        # Denoise: R050_M2_CW03
        "--cbo-history-denoise-mode", "local_median",
        "--cbo-history-denoise-k", "7",
        "--cbo-history-denoise-radius", "0.50",
        "--cbo-history-denoise-min-neighbors", "2",
        "--cbo-history-denoise-context-weight", "0.3",
        "--cbo-history-denoise-theta-weight", "1.0",
        "--cbo-history-denoise-stat", "median",
        "--cbo-history-denoise-apply-to", "all",

        "--output-root", "$OutputRoot",
        "--export-short-names"
    )

    if ($Warm) {
        if ([string]::IsNullOrWhiteSpace($WarmHistoryDir) -or -not (Test-Path $WarmHistoryDir)) {
            Write-Host "[ERROR] Warm run requested but warm history dir missing:"
            Write-Host $WarmHistoryDir
            $script:FAILED += "$RunName missing warm history: $WarmHistoryDir"
            return
        }

        $argsList += @(
            "--cbo-warm-start-history", "$WarmHistoryDir",
            "--cbo-warm-start-mode", "similar_topk",
            "--cbo-warm-start-topk", "100",
            "--cbo-warm-start-max-rows", "300",
            "--cbo-warm-start-label", "$WarmLabel"
        )
    }

    & python @argsList 2>&1 | Tee-Object -FilePath $log

    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "[FAILED] $RunName exit=$exitCode"
        $script:FAILED += "$RunName exit=$exitCode log=$log"
        $status = "failed_exit_$exitCode"
    } else {
        Write-Host "[DONE] $RunName"
        $status = "done"
    }

    $script:MANIFEST += [pscustomobject]@{
        run_name = $RunName
        lambda = $Lambda
        task_probs = $TaskProbs
        seed = $Seed
        warm = $Warm
        warm_history = $WarmHistoryDir
        output_root = $OutputRoot
        status = $status
        log = $log
    }

    Write-Host ""
}

$EXPERIMENTS = @(
    @{
        Pair = "P1_RT_sim_RT60_to_RT50"
        TargetName = "lam2p6_RT50_Batch40_AI10"
        Lambda = "2.6"
        TaskProbs = "50,40,10"
        SourceDir = $SOURCE_HISTORY
        WarmLabel = "source_RT60_Batch30_AI10"
    },
    @{
        Pair = "P2_RT_to_AI_neg_RT60_to_AI70"
        TargetName = "lam3p0_RT10_Batch20_AI70"
        Lambda = "3.0"
        TaskProbs = "10,20,70"
        SourceDir = $SOURCE_HISTORY
        WarmLabel = "source_RT60_Batch30_AI10"
    }
)

foreach ($seed in $SEEDS) {
    foreach ($exp in $EXPERIMENTS) {
        $pair = $exp.Pair
        $targetName = $exp.TargetName
        $lambda = $exp.Lambda
        $taskProbs = $exp.TaskProbs
        $sourceDir = $exp.SourceDir
        $warmLabel = $exp.WarmLabel

        $coldOut = Join-Path $TARGET_ROOT "seed${seed}\${pair}\cold_${targetName}"
        $warmOut = Join-Path $TARGET_ROOT "seed${seed}\${pair}\warm_${targetName}"

        Run-CBOScenario `
            -RunName "seed${seed}_${pair}_cold_denoise" `
            -Lambda $lambda `
            -TaskProbs $taskProbs `
            -Seed $seed `
            -OutputRoot $coldOut `
            -Warm $false `
            -WarmHistoryDir "" `
            -WarmLabel ""

        Run-CBOScenario `
            -RunName "seed${seed}_${pair}_warm_denoise" `
            -Lambda $lambda `
            -TaskProbs $taskProbs `
            -Seed $seed `
            -OutputRoot $warmOut `
            -Warm $true `
            -WarmHistoryDir $sourceDir `
            -WarmLabel $warmLabel
    }
}

$manifestPath = Join-Path $ROOT "runs_manifest.csv"
$MANIFEST | Export-Csv -NoTypeInformation -Encoding UTF8 $manifestPath

$failedPath = Join-Path $ROOT "failed_runs.txt"
if ($FAILED.Count -gt 0) {
    $FAILED | Out-File -FilePath $failedPath -Encoding utf8
    Write-Host "[WARNING] Some runs failed. See:"
    Write-Host $failedPath
} else {
    "All runs completed or skipped successfully." | Out-File -FilePath $failedPath -Encoding utf8
    Write-Host "[OK] All runs completed or skipped successfully."
}

Write-Host "============================================================"
Write-Host "Denoise warm-start validation completed."
Write-Host "ROOT:"
Write-Host $ROOT
Write-Host "Manifest:"
Write-Host $manifestPath
Write-Host "Logs:"
Write-Host $LOGDIR
Write-Host "Failed:"
Write-Host $failedPath
Write-Host "============================================================"