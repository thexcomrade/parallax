<#
.SYNOPSIS
    Distributed k6 orchestrator script for PARALLAX Phase 7C.1.
.DESCRIPTION
    Audits the generator fleet, executes capacity-checked load stages,
    and captures consolidated empirical results.
#>

param(
    [string]$TargetStage = "stage1_50",
    [int]$TargetVUs = 50,
    [string]$Scenario = "distributed_api.js",
    [string]$BaseUrl = "http://staging-api:8000"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " PARALLAX Phase 7C.1: Distributed Load Generator Runner   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Audit Fleet & Staging API
Write-Host "`n[Step 1] Auditing Generator Fleet..." -ForegroundColor Yellow
$audit = & "backend\.venv\Scripts\python.exe" "loadtests\distributed\verify_generators.py"
Write-Host $audit

# 2. Check Allocation
Write-Host "`n[Step 2] Calculating VU Allocation for $TargetVUs VUs..." -ForegroundColor Yellow
try {
    $allocOutput = & "backend\.venv\Scripts\python.exe" -c "from loadtests.distributed.allocate_vus import allocate_vus; import json; cfg=json.load(open('loadtests/distributed/generators.json')); gens=[cfg['local_host']]+cfg.get('external_generators',[]); print(allocate_vus($TargetVUs, gens))"
    Write-Host "Allocation Plan: $allocOutput" -ForegroundColor Green
} catch {
    Write-Warning "Allocation failed: $_"
    exit 2
}

# 3. Target URL Safety Validation
Write-Host "`n[Step 3] Verifying Target Allowlist..." -ForegroundColor Yellow
$allowed = @("staging-api", "localhost", "127.0.0.1", "host.docker.internal")
$isAllowed = $false
foreach ($h in $allowed) {
    if ($BaseUrl.ToLower().Contains($h)) { $isAllowed = $true; break }
}
if (-not $isAllowed) {
    Write-Error "[SECURITY FATAL] Unapproved target URL: $BaseUrl. Refusing execution."
    exit 1
}

# 4. Execute Load Test on Allocated Generator
Write-Host "`n[Step 4] Executing Stage $TargetStage ($TargetVUs VUs)..." -ForegroundColor Yellow
$rawOut = "experiments/phase7c_distributed_rerun/raw_results/${TargetStage}_summary.json"
$runId = "run_" + [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()

docker run --rm `
    -v "${PWD}/loadtests:/loadtests:ro" `
    -v "${PWD}/experiments/phase7c_distributed_rerun/raw_results:/raw_results:rw" `
    --network parallax-staging-network `
    -e "BASE_URL=$BaseUrl" `
    -e "GENERATOR_ID=local-generator-node" `
    -e "RUN_ID=$runId" `
    -e "TARGET_VUS=$TargetVUs" `
    grafana/k6:latest run `
    --summary-export "/raw_results/${TargetStage}_summary.json" `
    "/loadtests/scenarios/$Scenario"

Write-Host "`nStage complete. Raw summary saved to $rawOut" -ForegroundColor Green
