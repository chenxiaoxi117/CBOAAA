cd D:\CBOv2

$ErrorActionPreference = "Continue"

$ROOT = "D:\CBOv2\results\denoise_radius_sweep"
New-Item -ItemType Directory -Force $ROOT | Out-Null

$TESTS = @(
  @{ Name="R025_M2"; Radius="0.25"; MinN="2"; CW="1.0"; TW="1.0" },
  @{ Name="R050_M2"; Radius="0.50"; MinN="2"; CW="1.0"; TW="1.0" },
  @{ Name="R075_M3"; Radius="0.75"; MinN="3"; CW="1.0"; TW="1.0" },
  @{ Name="R100_M3"; Radius="1.00"; MinN="3"; CW="1.0"; TW="1.0" },
  @{ Name="R050_M2_CW03"; Radius="0.50"; MinN="2"; CW="0.3"; TW="1.0" },
  @{ Name="R075_M3_CW03"; Radius="0.75"; MinN="3"; CW="0.3"; TW="1.0" }
)

foreach ($t in $TESTS) {
  $name = $t.Name
  $radius = $t.Radius
  $minN = $t.MinN
  $cw = $t.CW
  $tw = $t.TW

  $out = Join-Path $ROOT $name

  Write-Host "============================================================"
  Write-Host "START $name radius=$radius minN=$minN context_w=$cw theta_w=$tw"
  Write-Host "out=$out"
  Write-Host "============================================================"

  python -W ignore::RuntimeWarning -m new_tr_split `
    --mode scenario `
    --fixed-rng `
    --fixed-seed 42 `
    --lambda-schedule "0:12000:2.6" `
    --session-duration 12000 `
    --task-probs 60,30,10 `
    --bo-iterations 50 `
    --bo-interval 240 `
    --selected-keys reduced6_cbo_lite_pressure_only `
    --feedback-score window_original `
    --cbo-reference-mode off `
    --cbo-objective-mode eval_cost `
    --cbo-history-denoise-mode local_median `
    --cbo-history-denoise-k 7 `
    --cbo-history-denoise-radius $radius `
    --cbo-history-denoise-min-neighbors $minN `
    --cbo-history-denoise-context-weight $cw `
    --cbo-history-denoise-theta-weight $tw `
    --cbo-history-denoise-stat median `
    --output-root $out `
    --export-short-names
}