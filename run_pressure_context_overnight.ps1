cd D:\CBOv2

$ErrorActionPreference = "Continue"

$ROOT = "D:\CBOv2\results\pressure_context_overnight_validation"
$LOGDIR = "$ROOT\logs"
$TARGET_ROOT = "$ROOT\targets"

New-Item -ItemType Directory -Force $ROOT | Out-Null
New-Item -ItemType Directory -Force $LOGDIR | Out-Null
New-Item -ItemType Directory -Force $TARGET_ROOT | Out-Null

$FAILED = @()
$MANIFEST = @()

function Has-RoundSummary {
    param([string]$RunDir)

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

function Run-One {
    param(
        [string]$SceneKey,
        [string]$SceneName,
        [string]$Lambda,
        [string]$TaskProbs,
        [int]$Seed,
        [string]$MethodName,
        [string]$SelectedKey
    )

    $safeRunName = "seed${Seed}_${SceneKey}_${MethodName}"
    $out = "$TARGET_ROOT\seed${Seed}\${SceneKey}\${MethodName}_${SceneName}"
    $log = "$LOGDIR\$safeRunName.log"

    if (Has-RoundSummary -RunDir $out) {
        Write-Host "[SKIP] Existing round_summary found: $safeRunName"
        $script:MANIFEST += [pscustomobject]@{
            run_name = $safeRunName
            scene = $SceneKey
            scene_name = $SceneName
            lambda = $Lambda
            task_probs = $TaskProbs
            seed = $Seed
            method_name = $MethodName
            selected_key = $SelectedKey
            output_root = $out
            status = "skipped_existing"
            log = $log
        }
        return
    }

    Write-Host "============================================================"
    Write-Host "START $safeRunName"
    Write-Host "scene=$SceneName lambda=$Lambda task_probs=$TaskProbs seed=$Seed"
    Write-Host "method=$MethodName selected_key=$SelectedKey"
    Write-Host "out=$out"
    Write-Host "log=$log"
    Write-Host "============================================================"

    $argsList = @(
        "-W", "ignore::RuntimeWarning",
        "-W", "ignore::UserWarning",
        "-m", "new_tr_split",
        "--mode", "scenario",
        "--fixed-rng",
        "--fixed-seed", "$Seed",
        "--lambda-schedule", "0:120000:$Lambda",
        "--session-duration", "120000",
        "--task-probs", "$TaskProbs",
        "--bo-iterations", "500",
        "--bo-interval", "240",
        "--selected-keys", "$SelectedKey",
        "--feedback-score", "window_original",
        "--cbo-reference-mode", "off",
        "--cbo-objective-mode", "eval_cost",
        "--output-root", "$out",
        "--export-short-names"
    )

    & python @argsList 2>&1 | Tee-Object -FilePath $log

    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "[FAILED] $safeRunName exit=$exitCode"
        $script:FAILED += "$safeRunName exit=$exitCode log=$log"
        $status = "failed_exit_$exitCode"
    } else {
        Write-Host "[DONE] $safeRunName"
        $status = "done"
    }

    $script:MANIFEST += [pscustomobject]@{
        run_name = $safeRunName
        scene = $SceneKey
        scene_name = $SceneName
        lambda = $Lambda
        task_probs = $TaskProbs
        seed = $Seed
        method_name = $MethodName
        selected_key = $SelectedKey
        output_root = $out
        status = $status
        log = $log
    }

    Write-Host ""
}

$SCENES = @(
    @{
        Key = "P1_RT50_Batch40_AI10"
        Name = "lam2p6_RT50_Batch40_AI10"
        Lambda = "2.6"
        TaskProbs = "50,40,10"
        Seeds = @(43,44,45,46,47)
        Include5D = $true
    },
    @{
        Key = "P0_RT60_Batch30_AI10"
        Name = "lam2p6_RT60_Batch30_AI10"
        Lambda = "2.6"
        TaskProbs = "60,30,10"
        Seeds = @(43,44,45)
        Include5D = $false
    },
    @{
        Key = "P2_AI70_RT10_Batch20"
        Name = "lam3p0_RT10_Batch20_AI70"
        Lambda = "3.0"
        TaskProbs = "10,20,70"
        Seeds = @(43,44,45)
        Include5D = $false
    }
)

$METHODS_BASE = @(
    @{
        Name = "fixed_tuned"
        Key = "reduced6_fixed_tuned"
    },
    @{
        Name = "cbo4d_pressure_only"
        Key = "reduced6_cbo_lite_pressure_only"
    },
    @{
        Name = "cbo6d_pressure_transition"
        Key = "reduced6_cbo_lite_pressure_transition"
    }
)

$METHOD_5D = @{
    Name = "cbo5d_prev_unfinished"
    Key = "reduced6_cbo_lite_pressure_prev_unfinished"
}

foreach ($scene in $SCENES) {
    $methods = @()
    $methods += $METHODS_BASE
    if ($scene.Include5D) {
        $methods += $METHOD_5D
    }

    foreach ($seed in $scene.Seeds) {
        foreach ($method in $methods) {
            Run-One `
                -SceneKey $scene.Key `
                -SceneName $scene.Name `
                -Lambda $scene.Lambda `
                -TaskProbs $scene.TaskProbs `
                -Seed $seed `
                -MethodName $method.Name `
                -SelectedKey $method.Key
        }
    }
}

$manifestPath = "$ROOT\runs_manifest.csv"
$MANIFEST | Export-Csv -NoTypeInformation -Encoding UTF8 $manifestPath

$failedPath = "$ROOT\failed_runs.txt"
if ($FAILED.Count -gt 0) {
    $FAILED | Out-File -FilePath $failedPath -Encoding utf8
    Write-Host "[WARNING] Some runs failed. See:"
    Write-Host $failedPath
} else {
    "All runs completed or skipped successfully." | Out-File -FilePath $failedPath -Encoding utf8
    Write-Host "[OK] All runs completed or skipped successfully."
}

Write-Host "============================================================"
Write-Host "Pressure context overnight validation completed."
Write-Host "ROOT:"
Write-Host $ROOT
Write-Host "Manifest:"
Write-Host $manifestPath
Write-Host "Logs:"
Write-Host $LOGDIR
Write-Host "Failed:"
Write-Host $failedPath
Write-Host "============================================================"