cd D:\CBOv2

$ErrorActionPreference = "Continue"

# ============================================================
# Overnight warm-start transfer validation
# Method: reduced6_cbo_lite_pressure_only
# Goal:
#   1) similar source -> target positive transfer
#   2) dissimilar source -> target negative transfer
#   3) validate on multiple scene families and seeds
# ============================================================

$ROOT = "D:\CBOv2\results\transfer_cbo_pressure_extended_overnight"
$LOGDIR = Join-Path $ROOT "logs"
$SOURCE_ROOT = Join-Path $ROOT "sources"
$TARGET_ROOT = Join-Path $ROOT "targets"

New-Item -ItemType Directory -Force $ROOT | Out-Null
New-Item -ItemType Directory -Force $LOGDIR | Out-Null
New-Item -ItemType Directory -Force $SOURCE_ROOT | Out-Null
New-Item -ItemType Directory -Force $TARGET_ROOT | Out-Null

$METHOD = "reduced6_cbo_lite_pressure_only"

# target seeds: seed42 already ran in your earlier experiment, so run new seeds.
$TARGET_SEEDS = @(43, 44, 45)

# If you want even longer tomorrow, change this to @(43,44,45,46)
# But tonight 43/44/45 should already be long enough.

$FAILED = @()
$MANIFEST = @()

function Find-WarmHistoryDir {
    param(
        [string[]]$CandidateDirs
    )

    foreach ($d in $CandidateDirs) {
        if (Test-Path $d) {
            $f = Get-ChildItem $d -Recurse -Filter "bo_warm_history.csv" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -ne $f) {
                return $d
            }
        }
    }
    return $null
}

function Has-RoundSummary {
    param(
        [string]$RunDir
    )

    if (-not (Test-Path $RunDir)) {
        return $false
    }

    $f = Get-ChildItem $RunDir -Recurse -Filter "*.csv" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name.ToLower().Contains("round_summary") -and $_.FullName.ToLower() -notmatch "_short_export" } |
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
        "--output-root", "$OutputRoot",
        "--export-short-names"
    )

    if ($Warm) {
        if ([string]::IsNullOrWhiteSpace($WarmHistoryDir) -or -not (Test-Path $WarmHistoryDir)) {
            Write-Host "[ERROR] Warm run requested but warm history dir missing: $WarmHistoryDir"
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

function Ensure-Source {
    param(
        [string]$SourceName,
        [string]$Lambda,
        [string]$TaskProbs,
        [string[]]$KnownCandidateDirs
    )

    $existing = Find-WarmHistoryDir -CandidateDirs $KnownCandidateDirs
    if ($null -ne $existing) {
        Write-Host "[OK] Reusing existing source for $SourceName : $existing"
        return $existing
    }

    $out = Join-Path $SOURCE_ROOT $SourceName

    Run-CBOScenario `
        -RunName "source_$SourceName" `
        -Lambda $Lambda `
        -TaskProbs $TaskProbs `
        -Seed 42 `
        -OutputRoot $out `
        -Warm $false `
        -WarmHistoryDir "" `
        -WarmLabel ""

    $created = Find-WarmHistoryDir -CandidateDirs @($out)
    if ($null -eq $created) {
        Write-Host "[ERROR] Source did not produce bo_warm_history.csv: $SourceName"
        $script:FAILED += "source_$SourceName missing bo_warm_history.csv"
        return ""
    }

    Write-Host "[OK] Created source for $SourceName : $created"
    return $created
}

# ============================================================
# Phase 1: prepare source histories
# ============================================================

# Source A: RT-heavy / RT-Batch pressure.
# Prefer your already completed source.
$SRC_RT60 = Ensure-Source `
    -SourceName "source_lam2p6_RT60_Batch30_AI10" `
    -Lambda "2.6" `
    -TaskProbs "60,30,10" `
    -KnownCandidateDirs @(
        "D:\CBOv2\results\transfer_cbo_pressure\source_lam2p6_RT60_Batch30_AI10",
        "D:\CBOv2\results\transfer_cbo_pressure_extended_overnight\sources\source_lam2p6_RT60_Batch30_AI10"
    )

# Source B: AI-heavy.
# If your previous dissimilar cold run produced bo_warm_history.csv, reuse it.
# Otherwise run a new AI-heavy source.
$SRC_AI70 = Ensure-Source `
    -SourceName "source_lam3p0_RT10_Batch20_AI70" `
    -Lambda "3.0" `
    -TaskProbs "10,20,70" `
    -KnownCandidateDirs @(
        "D:\CBOv2\results\transfer_cbo_pressure\target_dissimilar_cold_lam3p0_RT10_Batch20_AI70",
        "D:\CBOv2\results\transfer_cbo_pressure_extended_overnight\sources\source_lam3p0_RT10_Batch20_AI70"
    )

# Source C: Batch-heavy.
$SRC_BATCH60 = Ensure-Source `
    -SourceName "source_lam1p8_RT30_Batch60_AI10" `
    -Lambda "1.8" `
    -TaskProbs "30,60,10" `
    -KnownCandidateDirs @(
        "D:\CBOv2\results\transfer_cbo_pressure_extended_overnight\sources\source_lam1p8_RT30_Batch60_AI10"
    )

Write-Host "============================================================"
Write-Host "Source histories:"
Write-Host "SRC_RT60    = $SRC_RT60"
Write-Host "SRC_AI70    = $SRC_AI70"
Write-Host "SRC_BATCH60 = $SRC_BATCH60"
Write-Host "============================================================"

# ============================================================
# Phase 2: target transfer experiments
# Each pair has cold + warm for seeds 43/44/45.
# ============================================================

$EXPERIMENTS = @(
    # Positive transfer expected:
    # RT60 -> RT50, same lambda, similar RT-Batch pressure.
    @{
        Pair = "P1_RT_sim_RT60_to_RT50"
        TargetName = "lam2p6_RT50_Batch40_AI10"
        Lambda = "2.6"
        TaskProbs = "50,40,10"
        SourceDir = $SRC_RT60
        WarmLabel = "source_RT60_Batch30_AI10"
    },

    # Negative transfer expected:
    # RT60 -> AI70, very different workload.
    @{
        Pair = "P2_RT_to_AI_neg_RT60_to_AI70"
        TargetName = "lam3p0_RT10_Batch20_AI70"
        Lambda = "3.0"
        TaskProbs = "10,20,70"
        SourceDir = $SRC_RT60
        WarmLabel = "source_RT60_Batch30_AI10"
    },

    # Positive transfer expected:
    # AI70 -> AI60, AI-heavy to AI-heavy.
    @{
        Pair = "P3_AI_sim_AI70_to_AI60"
        TargetName = "lam3p0_RT10_Batch30_AI60"
        Lambda = "3.0"
        TaskProbs = "10,30,60"
        SourceDir = $SRC_AI70
        WarmLabel = "source_AI70"
    },

    # Positive or mild transfer expected:
    # Batch-heavy -> Batch-heavy / mixed batch.
    @{
        Pair = "P4_Batch_sim_Batch60_to_Batch60"
        TargetName = "lam2p6_RT20_Batch60_AI20"
        Lambda = "2.6"
        TaskProbs = "20,60,20"
        SourceDir = $SRC_BATCH60
        WarmLabel = "source_Batch60"
    }
)

foreach ($seed in $TARGET_SEEDS) {
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
            -RunName "seed${seed}_${pair}_cold" `
            -Lambda $lambda `
            -TaskProbs $taskProbs `
            -Seed $seed `
            -OutputRoot $coldOut `
            -Warm $false `
            -WarmHistoryDir "" `
            -WarmLabel ""

        Run-CBOScenario `
            -RunName "seed${seed}_${pair}_warm" `
            -Lambda $lambda `
            -TaskProbs $taskProbs `
            -Seed $seed `
            -OutputRoot $warmOut `
            -Warm $true `
            -WarmHistoryDir $sourceDir `
            -WarmLabel $warmLabel
    }
}

# ============================================================
# Save manifest / failed list
# ============================================================

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
Write-Host "Overnight warm-start extended experiment completed."
Write-Host "ROOT:"
Write-Host $ROOT
Write-Host "Manifest:"
Write-Host $manifestPath
Write-Host "Logs:"
Write-Host $LOGDIR
Write-Host "Failed:"
Write-Host $failedPath
Write-Host "============================================================"