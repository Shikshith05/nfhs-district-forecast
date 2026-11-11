<#
.SYNOPSIS
  Gate B: pre-flight, confirmation, freeze tag, ONE NFHS-6 evaluation, commit of its outputs.

.DESCRIPTION
  Run from anywhere:   powershell -ExecutionPolicy Bypass -File scripts\gate_b.ps1

  Steps (stops at the first error and never retries anything):
    1. scripts\preflight.py (full); any FAIL stops here
    2. shows what will be frozen and asks you to type YES
    3. git tag -a freeze-v1 on HEAD
    4. scripts\evaluate_nfhs6.py --confirm-frozen   (exactly once)
    5. checks the run log says "completed", then commits outputs\ (including
       the predictions parquet, which .gitignore would otherwise skip)
  It does not push. Push afterwards with:  git push origin main freeze-v1
  See docs\HUMAN_CHECKLIST.md before running this. DECISIONS D36.
#>
param(
  [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$Python = "",
  [string]$Tag = "freeze-v1"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

function Stop-Gate([string]$Message, [int]$Code = 1) {
  Write-Host ""
  Write-Host "GATE B STOPPED: $Message" -ForegroundColor Red
  Write-Host "Nothing will be retried. Read the message above, fix the cause, and decide what to do by hand."
  exit $Code
}

Set-Location -LiteralPath $RepoRoot
if ($Python -eq "") { $Python = Join-Path $RepoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) { Stop-Gate "Python not found at $Python (set up .venv first, see README)." }
$Outputs = Join-Path $RepoRoot "outputs"
$Marker = Join-Path $Outputs "nfhs6_evaluation_run.json"

Write-Host "=== Gate B: one-shot NFHS-6 evaluation ===" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"

# ---- 1. pre-flight --------------------------------------------------------
Write-Host ""
Write-Host "[1/5] Pre-flight checks (takes a minute or two)..." -ForegroundColor Cyan
& $Python (Join-Path $RepoRoot "scripts\preflight.py")
if ($LASTEXITCODE -ne 0) { Stop-Gate "pre-flight failed (exit $LASTEXITCODE)." }
if (Test-Path -LiteralPath $Marker) { Stop-Gate "$Marker already exists: the evaluation has already run." }

# ---- 2. summary and confirmation -----------------------------------------
$head = (& git rev-parse HEAD)
if ($LASTEXITCODE -ne 0) { Stop-Gate "git rev-parse failed." }
$branch = (& git rev-parse --abbrev-ref HEAD)
$manifestPath = Join-Path $Outputs "models\manifest.json"
$trainHash = "(manifest not found)"
if (Test-Path -LiteralPath $manifestPath) {
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  $trainHash = "$($manifest.train_hash); $($manifest.n_core_districts) training units, $($manifest.n_test_units) test units, $($manifest.n_stable_units) stable"
}
Write-Host ""
Write-Host "[2/5] About to freeze and score:" -ForegroundColor Cyan
Write-Host "  commit       $head ($branch)"
Write-Host "  tag          $Tag"
Write-Host "  frozen data  $trainHash"
Write-Host ""
Write-Host "This creates the tag and runs the NFHS-6 evaluation ONCE. It cannot be undone:"
Write-Host "after it runs, any change must be logged in DECISIONS.md and reported."
Write-Host "Type YES (capitals) to continue, anything else to stop:"
$answer = [Console]::In.ReadLine()
if ($null -eq $answer -or $answer.Trim() -cne "YES") { Stop-Gate "not confirmed (you typed '$answer'). Nothing was changed." 10 }

# ---- 3. tag ---------------------------------------------------------------
Write-Host ""
Write-Host "[3/5] Creating tag $Tag on $head" -ForegroundColor Cyan
& git tag -a $Tag -m "Models frozen before NFHS-6 evaluation"
if ($LASTEXITCODE -ne 0) { Stop-Gate "git tag failed; the evaluation was NOT run." }

# ---- 4. evaluation (exactly once) ------------------------------------------
Write-Host ""
Write-Host "[4/5] Running the NFHS-6 evaluation (once)..." -ForegroundColor Cyan
& $Python (Join-Path $RepoRoot "scripts\evaluate_nfhs6.py") --confirm-frozen
$evalCode = $LASTEXITCODE
if ($evalCode -ne 0) {
  if (Test-Path -LiteralPath $Marker) {
    Stop-Gate "the evaluation exited with $evalCode AFTER claiming the run ($Marker exists). It counts as the one run: do not rerun; log what happened in DECISIONS.md." $evalCode
  }
  Stop-Gate "the evaluation exited with $evalCode BEFORE scoring (no run marker). The tag $Tag exists locally; if you fix the cause, delete it with 'git tag -d $Tag' (only if not pushed) and start Gate B again." $evalCode
}
if (-not (Test-Path -LiteralPath $Marker)) { Stop-Gate "the evaluation returned 0 but $Marker is missing." }
$run = Get-Content -LiteralPath $Marker -Raw | ConvertFrom-Json
if ($run.status -ne "completed") { Stop-Gate "run log status is '$($run.status)', not 'completed'." }

# ---- 5. commit outputs ---------------------------------------------------
Write-Host ""
Write-Host "[5/5] Committing the evaluation outputs" -ForegroundColor Cyan
& git add -A -- outputs
if ($LASTEXITCODE -ne 0) { Stop-Gate "git add outputs failed (outputs are on disk; commit them by hand)." }
$pred = Join-Path $Outputs "predictions_nfhs5to6.parquet"
if (Test-Path -LiteralPath $pred) {
  & git add -f -- "outputs/predictions_nfhs5to6.parquet"
  if ($LASTEXITCODE -ne 0) { Stop-Gate "git add of the predictions failed (commit them by hand)." }
}
& git commit -q -m "Gate B: NFHS-6 one-shot evaluation outputs and run log (tag $Tag)"
if ($LASTEXITCODE -ne 0) { Stop-Gate "git commit failed (outputs are on disk; commit them by hand)." }

Write-Host ""
Write-Host "GATE B DONE." -ForegroundColor Green
Write-Host "  tag $Tag -> $head; outputs committed on top."
Write-Host "Next:"
Write-Host "  git push origin $branch $Tag"
Write-Host "  .venv\Scripts\python scripts\post_run.py"
exit 0
