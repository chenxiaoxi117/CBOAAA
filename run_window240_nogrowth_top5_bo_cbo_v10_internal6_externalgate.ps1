param(
    [int]$Seed = 43
)

$ErrorActionPreference = 'Continue'

# Keep this script ASCII-only so Windows PowerShell 5.1 will not corrupt
# the Chinese project path when reading a UTF-8 file without BOM.
$ProjectSubPath = -join @(
    [char]0x65B0, [char]0x7684, [char]0x4EE3, [char]0x7801, [char]0x7ED3, [char]0x6784,
    [char]0x005C,
    [char]0x53BB, [char]0x6389, [char]0x52A8, [char]0x6001, [char]0x5806, [char]0x79EF
)

$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'

$ROOT = Join-Path 'D:\CBOv2' $ProjectSubPath
$OUT = "D:\CBOv2\results\window240_nogrowth_top5_static_bo_cbo_v10_internal6_externalgate_s$Seed"
$KEY = 'reduced7_bo_greedy,reduced7_cbo_lite_pressure_taskmix_counts'

New-Item -ItemType Directory -Force -Path $OUT | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OUT 'logs') | Out-Null

$cases = @(
    @{ lam = '3.0'; probs = '10,40,50'; name = 'lambda3p0_rt10_batch40_ai50' },
    @{ lam = '3.0'; probs = '30,60,10'; name = 'lambda3p0_rt30_batch60_ai10' },
    @{ lam = '3.0'; probs = '10,30,60'; name = 'lambda3p0_rt10_batch30_ai60' },
    @{ lam = '2.6'; probs = '20,10,70'; name = 'lambda2p6_rt20_batch10_ai70' },
    @{ lam = '2.6'; probs = '70,20,10'; name = 'lambda2p6_rt70_batch20_ai10' }
)

Set-Location -LiteralPath $ROOT

Write-Host '============================================================'
Write-Host 'Start BO+CBO top5 rerun | v10 internal6 + external gate + no TR'
Write-Host "ROOT = $ROOT"
Write-Host "OUT  = $OUT"
Write-Host "KEY  = $KEY"
Write-Host "Seed = $Seed"
Write-Host '============================================================'

foreach ($case in $cases) {
    $name = $case.name
    $lam = $case.lam
    $probs = $case.probs

    $caseOut = Join-Path $OUT $name
    $stdout = Join-Path (Join-Path $OUT 'logs') "$name.stdout.log"
    $stderr = Join-Path (Join-Path $OUT 'logs') "$name.stderr.log"

    Write-Host ''
    Write-Host '============================================================'
    Write-Host "Running $name"
    Write-Host "lambda = $lam"
    Write-Host "task_probs = $probs"
    Write-Host "stdout = $stdout"
    Write-Host "stderr = $stderr"
    Write-Host '============================================================'

    $args = @(
        '-m', 'new_tr_split',
        '--mode', 'pressure_scan',
        '--selected-keys', $KEY,
        '--bo-iterations', '500',
        '--bo-interval', '240',
        '--session-duration', '120000',
        '--fixed-rng',
        '--fixed-seed', "$Seed",
        '--reduced7-energy-scale-bounds', '0.5,3.0',
        '--feedback-score', 'task_effective_backlog_violation',
        '--bo-history-mode', 'recent',
        '--bo-recent-window', '80',
        '--cbo-objective-mode', 'normalized_tradeoff',
        '--cbo-reference-mode', 'calibrate',
        '--cbo-shared-reference-policy', 'cbo_first',
        '--cbo-shared-reference-warmup-rounds', '5',
        '--cbo-reference-source-method-key', 'reduced7_cbo_lite_pressure_taskmix_counts',
        '--cbo-backlog-growth-penalty-weight', '0',
        '--scheduler-score-norm-mode', 'candidate_minmax_deadline',
        '--task-adaptation',
        '--lambda-values', $lam,
        '--task-probs', $probs,
        '--output-root', $caseOut
    )

    & python @args > $stdout 2> $stderr

    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host "[FAILED] $name"
        Write-Host "ExitCode = $LASTEXITCODE"
        Write-Host '===== stderr tail ====='
        Get-Content -LiteralPath $stderr -Tail 120
        Write-Host '===== stdout tail ====='
        Get-Content -LiteralPath $stdout -Tail 80
        throw "Case failed: $name"
    }

    Write-Host "[OK] $name"
}

Write-Host ''
Write-Host '============================================================'
Write-Host 'All BO+CBO v10 internal6 external-gate tests finished.'
Write-Host 'Result root:'
Write-Host $OUT
Write-Host '============================================================'

$count = (Get-ChildItem -LiteralPath $OUT -Recurse -Filter '*round_summary*.csv').Count
Write-Host "round_summary count = $count"
